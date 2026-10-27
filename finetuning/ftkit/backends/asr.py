"""Zipformer RNN-T adaptation using the pinned upstream icefall model and k2 loss."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from ..common import (ROOT, fresh_directory, gpu_precision, model_snapshot, normalize,
                      read_jsonl, restore_rng, rng_state, seed_everything, write_json, write_jsonl)
from ..data import safe_audio
from ..metrics import error_rates
from ..training import (barrier, begin_run, checkpoint_fingerprint, distributed,
                        save_torch, schedule, validate_existing_output)


def check_weights(model_state, pretrained):
    """Only auxiliary pruned-loss projections may be absent from inference weights."""
    allowed = ("simple_am_proj.", "simple_lm_proj.")
    missing = [key for key in model_state if key not in pretrained]
    unexpected = [key for key in pretrained if key not in model_state]
    mismatched = [key for key in model_state if key in pretrained and tuple(model_state[key].shape) != tuple(pretrained[key].shape)]
    forbidden = [key for key in missing if not key.startswith(allowed)]
    if forbidden or unexpected or mismatched:
        raise ValueError(f"Incompatible Zipformer checkpoint. Missing inference weights: {forbidden[:12]}; "
                         f"unexpected: {unexpected[:12]}; shape mismatches: {mismatched[:12]}. "
                         "Set architecture options for the replacement checkpoint; partial pretrained loading is prohibited.")
    return missing


def icefall_modules(args, config):
    home = Path(os.environ.get("FINETUNE_HOME", ROOT))
    directory = Path(args.icefall_dir or home / ".cache/icefall").resolve()
    if not (directory / "icefall/__init__.py").is_file():
        raise ValueError(f"Missing icefall checkout at {directory}. Run setup asr or pass --icefall-dir.")
    actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
    if actual != config["icefall_revision"]:
        raise ValueError(f"icefall revision {actual} differs from configured {config['icefall_revision']}; use a matching checkout/config.")
    config["resolved_icefall_revision"] = actual
    for path in (directory, directory / "egs/librispeech/ASR", directory / "egs/librispeech/ASR/zipformer"):
        sys.path.insert(0, str(path))
    # Real lhotse is required. Never import Gipformer's inference-only mock loader.
    import lhotse
    import train as recipe
    from beam_search import greedy_search_batch
    return recipe, greedy_search_batch


def load(args, config, checkpoint=None):
    import torch
    import k2
    import sentencepiece as spm
    import kaldifeat
    directory, revision = model_snapshot(config, ["config.json"])
    if (directory / "config.json").exists():
        published = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if published.get("model_type") not in ("zipformer-transducer", None):
            raise ValueError("ASR backend only supports compatible Zipformer transducer models. Whisper/CTC need another backend.")
    directory, revision = model_snapshot(config, [config["checkpoint_file"], "config.json",
                                                config["tokenizer"]["bpe_file"], config["tokenizer"]["tokens_file"]])
    recipe, decoder = icefall_modules(args, config)
    params = recipe.get_params()
    # Read defaults without invoking a LibriSpeech-specific training CLI.
    for action in recipe.get_parser()._actions:
        if action.dest not in ("help",) and action.default is not None:
            params[action.dest] = action.default
    params.update(config["architecture"])
    tokens = k2.SymbolTable.from_file(str(directory / config["tokenizer"]["tokens_file"]))
    params.blank_id = tokens["<blk>"]
    params.unk_id = tokens["<unk>"]
    params.vocab_size = len([s for s in tokens.symbols if not s.startswith("#")])
    params.use_cr_ctc = False
    processor = spm.SentencePieceProcessor(model_file=str(directory / config["tokenizer"]["bpe_file"]))
    if processor.get_piece_size() != params.vocab_size:
        raise ValueError("BPE vocabulary and token table differ; a compatible original tokenizer is required")
    model = recipe.get_model(params)
    source = Path(checkpoint) if checkpoint else directory / config["checkpoint_file"]
    if source.is_dir():
        source = source / "model.pt"
    state = torch.load(source, map_location="cpu", weights_only=False)
    weights = state["model"] if "model" in state else state
    missing = check_weights(model.state_dict(), weights)
    model.load_state_dict(weights, strict=not missing)
    if missing:
        print("Explicitly initialized training-only loss weights:", ", ".join(missing))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    model.to(device)
    options = kaldifeat.FbankOptions()
    options.device = device
    options.frame_opts.dither = 0
    options.frame_opts.snip_edges = False
    options.frame_opts.samp_freq = 16000
    options.mel_opts.num_bins = 80
    options.mel_opts.high_freq = -400
    fbank = kaldifeat.Fbank(options)
    config["resolved_model_revision"] = revision
    return {"model": model, "processor": processor, "params": params, "decode": decoder, "fbank": fbank,
            "source_directory": directory, "recipe": recipe, "initial_state": state,
            "initialized_training_heads": missing}


def duration_batches(rows, budget, seed=42, epoch=0, world=1, rank=0):
    if any(row["duration"] > budget for row in rows):
        raise ValueError("An utterance exceeds max_audio_seconds; increase the batch duration budget.")
    rng = random.Random(seed + epoch)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    # Length-sort within small random buckets to avoid excessive padding.
    ordered = []
    for start in range(0, len(indices), 100):
        ordered.extend(sorted(indices[start:start + 100], key=lambda i: rows[i]["duration"]))
    batches, batch, seconds = [], [], 0.0
    for index in ordered:
        if batch and seconds + rows[index]["duration"] > budget:
            batches.append(batch)
            batch, seconds = [], 0.0
        batch.append(index)
        seconds += rows[index]["duration"]
    if batch:
        batches.append(batch)
    rng.shuffle(batches)
    if not batches:
        raise ValueError("Empty ASR dataset")
    original = list(batches)
    while len(batches) % world:
        batches.append(list(original[len(batches) % len(original)]))
    return batches[rank::world]


def features(loaded, paths, augment=False):
    import torch
    import soundfile as sf
    from torch.nn.utils.rnn import pad_sequence
    waves = []
    device = next(loaded["model"].parameters()).device
    for path in paths:
        audio, rate = sf.read(str(path), dtype="float32")
        if rate != 16000 or audio.ndim != 1:
            raise ValueError(f"Expected mono 16kHz WAV: {path}")
        waves.append(torch.from_numpy(audio).to(device))
    with torch.no_grad():
        values = loaded["fbank"](waves)
        lengths = torch.tensor([x.shape[0] for x in values], dtype=torch.int64, device=device)
        batch = pad_sequence(values, batch_first=True, padding_value=math.log(1e-10))
        if augment:
            for i, length in enumerate(lengths.tolist()):
                for _ in range(2):
                    width = random.randint(0, min(20, batch.shape[2]))
                    start = random.randint(0, batch.shape[2] - width)
                    batch[i, :length, start:start + width] = 0
                    width = random.randint(0, min(40, max(0, length // 10)))
                    start = random.randint(0, length - width)
                    batch[i, start:start + width] = 0
    return batch, lengths


def loss_for(model, loaded, values, lengths, texts, settings):
    import k2
    targets = k2.RaggedTensor(loaded["processor"].encode(texts, out_type=int))
    result = model(x=values, x_lens=lengths, y=targets, prune_range=settings["prune_range"],
                   am_scale=0.0, lm_scale=0.25, time_warp_factor=0)
    simple, pruned = result[:2]
    return (settings["simple_loss_scale"] * simple + pruned) / len(texts)


def evaluate_loaded(loaded, rows, root):
    import torch
    model = loaded["model"]
    model.eval()
    predictions = []
    started = time.perf_counter()
    with torch.no_grad():
        for row in rows:
            value, length = features(loaded, [safe_audio(root, row["audio"])])
            encoder, encoder_lengths = model.forward_encoder(value, length)
            tokens = loaded["decode"](model=model, encoder_out=encoder, encoder_out_lens=encoder_lengths)
            text = loaded["processor"].decode(tokens[0])
            predictions.append({"id": row["id"], "reference": row["text"], "prediction": text})
    rates = error_rates((p["reference"], p["prediction"]) for p in predictions)
    rates.update({"clips": len(rows), "wall_seconds": time.perf_counter() - started})
    return rates, predictions


def export_loaded(loaded, config, out):
    out = fresh_directory(out)
    save_torch(out / "model.pt", {"model": {k: v.detach().cpu() for k, v in loaded["model"].state_dict().items()}})
    for key in ("bpe_file", "tokens_file"):
        name = config["tokenizer"][key]
        shutil.copy2(loaded["source_directory"] / name, out / name)
    write_json(out / "config.json", {"model_type": "zipformer-transducer", "sample_rate": 16000,
                                     "architecture": config["architecture"]})
    write_json(out / "deployment.json", {"config": config, "format": "zipformer_pytorch"})
    return out


def train(args, config):
    import torch
    from torch.nn.parallel import DistributedDataParallel
    rank, world, local_rank = distributed(config, True)
    validate_existing_output(args, config)
    settings = config["training"]
    seed_everything(settings["seed"])
    dtype = gpu_precision(config)
    loaded = load(args, config, args.resume)
    model = loaded["model"]
    out = begin_run(args, config, rank)
    metadata = json.loads((Path(args.prepared) / "metadata.json").read_text(encoding="utf-8"))
    root = args.data_root or metadata["data_root"]
    train_rows = read_jsonl(Path(args.prepared) / "asr_train.jsonl")
    validation = read_jsonl(Path(args.prepared) / "asr_validation.jsonl")
    mode = "smoke" if args.smoke_test else "benchmark" if args.benchmark else "full"
    identity = checkpoint_fingerprint(config, metadata) + ":" + config["resolved_model_revision"] + f":{world}:{mode}"
    epochs = 2 if args.smoke_test else 1 if args.benchmark else settings["epochs"]
    validation = validation[:2] if args.smoke_test or args.benchmark else validation
    if args.smoke_test:
        # Tiny repeated subset exercises two updates and an actual resumed update quickly.
        train_rows = train_rows[:min(len(train_rows), 8 * world)]
    batches_per_epoch = len(duration_batches(train_rows, settings["max_audio_seconds"], world=world, rank=rank))
    total = batches_per_epoch * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule(total, settings["warmup_ratio"]))
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    start_epoch = start_batch = step = 0
    best = float("inf")
    if args.resume:
        state = loaded["initial_state"]
        if state.get("identity") != identity:
            raise ValueError("ASR resume identity mismatch: model, data, split, revision or training settings changed")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, start_batch, step, best = state["epoch"], state["next_batch"], state["step"], state["best_wer"]
        states = state["rng_by_rank"]
        if len(states) != world:
            raise ValueError("ASR resume requires the same GPU count")
        restore_rng(states[rank])
        if rank == 0 and not (out / "best.pt").exists():
            previous_best = Path(args.resume).parent / "best.pt"
            if previous_best.is_file():
                shutil.copy2(previous_best, out / "best.pt")
            elif math.isfinite(best):
                print("Previous best.pt is absent; selecting a new best from validation after resume.")
                best = float("inf")
        if world > 1:
            best_tensor = torch.tensor([best], device=f"cuda:{local_rank}")
            torch.distributed.broadcast(best_tensor, 0)
            best = best_tensor.item()
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False) if world > 1 else model
    first_name, first_parameter = next((n, p) for n, p in model.named_parameters() if p.requires_grad)
    before = first_parameter.detach().clone() if args.smoke_test else None
    timings, history = [], []
    started = time.perf_counter()

    def save(epoch, next_batch):
        states = [None] * world
        if world > 1:
            torch.distributed.all_gather_object(states, rng_state())
        else:
            states[0] = rng_state()
        if rank == 0:
            save_torch(out / "latest.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                       "next_batch": next_batch, "step": step, "best_wer": best, "identity": identity, "rng_by_rank": states})
        barrier()

    stop = False
    for epoch in range(start_epoch, epochs):
        batches = duration_batches(train_rows, settings["max_audio_seconds"], seed=settings["seed"], epoch=epoch, world=world, rank=rank)
        wrapped.train()
        for position, indices in enumerate(batches):
            if epoch == start_epoch and position < start_batch:
                continue
            torch.cuda.synchronize()
            tick = time.perf_counter()
            rows = [train_rows[i] for i in indices]
            value, lengths = features(loaded, [safe_audio(root, r["audio"]) for r in rows], settings["spec_augment"])
            loaded["recipe"].set_batch_count(model, 4000 + step)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                loss = loss_for(wrapped, loaded, value, lengths, [r["text"] for r in rows], settings)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite ASR loss at step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            torch.cuda.synchronize()
            if step > 2:
                timings.append(time.perf_counter() - tick)
            if rank == 0:
                print(f"epoch={epoch + 1} step={step} loss={float(loss.detach()):.4f}", flush=True)
            if step % settings["save_steps"] == 0:
                save(epoch, position + 1)
            if args.smoke_test:
                save(epoch, position + 1)
                # Restore the saved training state and perform the following update from it.
                saved = torch.load(out / "latest.pt", map_location=f"cuda:{local_rank}", weights_only=False)
                model.load_state_dict(saved["model"], strict=True)
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
                scaler.load_state_dict(saved["scaler"])
                restore_rng(saved["rng_by_rank"][rank])
            if (args.smoke_test and step >= 2) or (args.benchmark and step >= 12):
                stop = True
                break
        barrier()
        if rank == 0:
            rates, predictions = evaluate_loaded(loaded, validation, root)
            history.append({"epoch": epoch + 1, **rates})
            write_json(out / "validation_metrics.json", history)
            write_jsonl(out / "validation_predictions.jsonl", predictions)
            if rates["wer"] is not None and rates["wer"] < best:
                best = rates["wer"]
                save_torch(out / "best.pt", {"model": model.state_dict()})
        if world > 1:
            best_tensor = torch.tensor([best], device=f"cuda:{local_rank}")
            torch.distributed.broadcast(best_tensor, 0)
            best = best_tensor.item()
        # A short run must retain its true next-batch cursor, rather than mark an unfinished epoch complete.
        save(epoch if stop and position + 1 < len(batches) else epoch + 1,
             position + 1 if stop and position + 1 < len(batches) else 0)
        if stop:
            break
    if rank == 0:
        write_json(out / "training_metrics.json", {"steps": step, "wall_seconds": time.perf_counter() - started, "best_wer": best})
        if args.benchmark:
            if not timings:
                raise ValueError("Not enough batches for a steady-state benchmark; increase the dataset or reduce the duration budget")
            seconds = sum(timings) / len(timings)
            write_json(out / "benchmark.json", {"steady_batch_seconds": seconds, "estimated_training_hours": seconds * batches_per_epoch * settings["epochs"] / 3600})
        if args.smoke_test and torch.equal(before, first_parameter.detach()):
            raise RuntimeError(f"Smoke test failed: {first_name} did not update")
        best_state = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(best_state["model"], strict=True)
        # A fully completed run can be resumed for checks without overwriting its existing export.
        export_destination = out / "export"
        if export_destination.exists() and any(export_destination.iterdir()):
            export_destination = out / f"export-step-{step}"
        exported = export_loaded(loaded, config, export_destination)
        if args.smoke_test:
            reloaded = torch.load(exported / "model.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(reloaded["model"], strict=True)
            rates, _ = evaluate_loaded(loaded, validation[:1], root)
            write_json(out / "smoke_test.json", {"finite_loss": True, "weights_updated": True, "checkpoint_resume": True,
                                                "export_reload": True, "reload_wer": rates["wer"]})
        print(f"ASR training complete: {out}")
    barrier()


def evaluate(args, config):
    seed_everything(config["training"]["seed"])
    gpu_precision(config)
    out = fresh_directory(args.output)
    metadata = json.loads((Path(args.prepared) / "metadata.json").read_text(encoding="utf-8"))
    root = args.data_root or metadata["data_root"]
    rows = read_jsonl(Path(args.prepared) / "asr_validation.jsonl")
    if args.limit:
        rows = rows[:args.limit]
    loaded = load(args, config, args.checkpoint)
    rates, predictions = evaluate_loaded(loaded, rows, root)
    write_json(out / "metrics.json", rates)
    write_jsonl(out / "predictions.jsonl", predictions)
    if args.baseline and args.checkpoint:
        import torch
        del loaded
        torch.cuda.empty_cache()
        baseline = load(args, config)
        base_rates, base_predictions = evaluate_loaded(baseline, rows, root)
        write_json(out / "baseline_metrics.json", base_rates)
        write_jsonl(out / "baseline_predictions.jsonl", base_predictions)
    print(json.dumps(rates, indent=2))


def export(args, config):
    gpu_precision(config)
    loaded = load(args, config, args.checkpoint)
    export_loaded(loaded, config, args.output)
    print(f"Exported: {args.output}")


def transcribe(args, config):
    import torch
    if not args.audio:
        raise ValueError("transcribe requires --audio WAV [WAV ...]")
    gpu_precision(config)
    loaded = load(args, config, args.checkpoint)
    loaded["model"].eval()
    with torch.no_grad():
        for path in args.audio:
            value, length = features(loaded, [path])
            encoded, lengths = loaded["model"].forward_encoder(value, length)
            tokens = loaded["decode"](model=loaded["model"], encoder_out=encoded, encoder_out_lens=lengths)
            print(json.dumps({"audio": str(path), "text": loaded["processor"].decode(tokens[0])}, ensure_ascii=False))


def run(args, config):
    {"train-asr": train, "eval-asr": evaluate, "export-asr": export, "transcribe": transcribe}[args.command](args, config)
