"""Unsloth causal/Qwen3-VL text-only LoRA with current-assistant loss masking."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

from ..common import (fresh_directory, gpu_precision, model_snapshot, read_jsonl,
                      seed_everything, write_json, write_jsonl)
from ..data import encode_trajectory, trajectory
from ..training import barrier, begin_run, distributed, validate_existing_output


def load(config, checkpoint=None, *, train=False, merge_export=False):
    # Unsloth must patch the stack before any Torch/Transformers initialization.
    from unsloth import FastLanguageModel, FastVisionModel
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from ..llm_common import tokenizer_for

    if int(config.get("devices", 1)) != 1 or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("LLM training supports one GPU")
    directory, revision = model_snapshot(config, ["config.json"])
    config["resolved_model_revision"] = revision
    model_config = AutoConfig.from_pretrained(directory, trust_remote_code=False)
    handler = config["architecture"].get("handler", "auto")
    vision = model_config.model_type == "qwen3_vl"
    if vision and handler in ("auto", "qwen3_vl"):
        loader = FastVisionModel
    elif type(model_config) in AutoModelForCausalLM._model_mapping and handler in ("auto", "causal_lm"):
        loader = FastLanguageModel
    else:
        raise ValueError(f"Unsupported Unsloth architecture {model_config.model_type!r} / handler {handler!r}")
    directory, revision = model_snapshot(config)
    settings = config["training"]
    dtype = gpu_precision(config)
    model, _ = loader.from_pretrained(model_name=str(directory), max_seq_length=settings["max_seq_length"],
                                      dtype=dtype, load_in_4bit=settings.get("quantize", False) and not merge_export,
                                      trust_remote_code=False, device_map={"": 0})
    source = Path(checkpoint) if checkpoint and (Path(checkpoint) / "tokenizer_config.json").is_file() else directory
    tokenizer = tokenizer_for(source, config)
    tokenizer.padding_side = "right"
    if checkpoint:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=train)
    elif train:
        projections = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        names = [name for name, _ in model.named_modules() if name.endswith(projections)
                 and ".visual." not in name and not name.startswith("visual.")]
        if vision:
            names = [name for name in names if ".language_model." in name or name.startswith("language_model.")]
        if not names:
            raise ValueError("No supported language attention/MLP projections")
        kwargs = dict(r=settings["lora_rank"], lora_alpha=settings["lora_alpha"],
                      lora_dropout=settings["lora_dropout"], target_modules=names, bias="none",
                      use_gradient_checkpointing=settings["gradient_checkpointing"], random_state=settings["seed"])
        if vision:
            kwargs.update(finetune_vision_layers=False, finetune_language_layers=True,
                          finetune_attention_modules=True, finetune_mlp_modules=True)
        model = loader.get_peft_model(model, **kwargs)
    if train:
        loader.for_training(model, use_gradient_checkpointing=settings["gradient_checkpointing"])
        # VLM wrappers and their nested language models can own different configs.
        # Disable all caches, including when activation checkpointing is turned off.
        for module in model.modules():
            nested_config = getattr(module, "config", None)
            if nested_config is not None and hasattr(nested_config, "use_cache"):
                nested_config.use_cache = False
    model.config.use_cache = not train
    if train and any(p.requires_grad for name, p in model.named_parameters() if "visual." in name):
        raise ValueError("Vision parameters unexpectedly trainable")
    config["resolved_model_revision"] = revision
    return model, tokenizer


def prepare(tokenizer, records, max_length):
    encoded, excluded = [], []
    for record in records:
        messages, targets = trajectory(record)
        row = encode_trajectory(tokenizer, messages, targets, max_length)
        if row is None:
            excluded.append({"id": record["id"], "tokens": len(tokenizer.apply_chat_template(messages, tokenize=True))})
        else:
            encoded.append(row)
    if not encoded:
        raise ValueError("No examples fit max_seq_length; increase context or compact schemas. No targets were truncated.")
    return encoded, excluded


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        import torch
        length = max(len(r["input_ids"]) for r in rows)
        # Avoid padding all the way to configured maximum context.
        result = {}
        for key, padding in (("input_ids", self.pad_id), ("attention_mask", 0), ("labels", -100)):
            result[key] = torch.tensor([r[key] + [padding] * (length - len(r[key])) for r in rows], dtype=torch.long)
        return result


def train(args, config):
    import unsloth  # Must precede Torch, distributed() and Transformers.
    import torch
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset
    from ..llm_common import llm_identity
    from ..common import digest
    rank, world, local_rank = distributed(config, True)
    validate_existing_output(args, config)
    seed_everything(config["training"]["seed"])
    model, tokenizer = load(config, train=True)
    out = begin_run(args, config, rank)
    metadata = json.loads((Path(args.prepared) / "metadata.json").read_text(encoding="utf-8"))
    settings = config["training"]
    train_records = read_jsonl(Path(args.prepared) / "llm_train.jsonl")
    val_records = read_jsonl(Path(args.prepared) / "llm_validation.jsonl")
    train_rows, train_excluded = prepare(tokenizer, train_records, settings["max_seq_length"])
    val_rows, val_excluded = prepare(tokenizer, val_records, settings["max_seq_length"])
    identity = digest({"dataset": metadata["fingerprint"], "model": config["model"], "revision": config["resolved_model_revision"],
                       "training": settings, "architecture": config["architecture"], "devices": world,
                       "runtime": llm_identity(tokenizer), "run_mode": "smoke" if args.smoke_test else "benchmark" if args.benchmark else "full"})
    if args.resume:
        resume_meta = Path(args.resume) / "finetune_identity.json"
        if not resume_meta.is_file() or json.loads(resume_meta.read_text())["identity"] != identity:
            raise ValueError("Resume checkpoint identity differs: model, dataset, split or training configuration changed")
    if rank == 0:
        write_json(out / "length_report.json", {"train_kept": len(train_rows), "validation_kept": len(val_rows),
                                               "train_excluded": train_excluded, "validation_excluded": val_excluded})
    effective = settings["effective_batch_size"]
    denominator = settings["microbatch_size"] * world
    if effective % denominator:
        raise ValueError("effective_batch_size must be divisible by microbatch_size * number of GPUs")
    accumulation = effective // denominator
    max_steps = 2 if args.smoke_test else 12 if args.benchmark else -1
    if args.smoke_test or args.benchmark:
        train_rows = train_rows[:max(effective * 2, 8)]
        val_rows = val_rows[:2]

    class IdentityCallback(TrainerCallback):
        def on_save(self, arguments, state, control, **kwargs):
            if state.is_world_process_zero:
                write_json(Path(arguments.output_dir) / f"checkpoint-{state.global_step}" / "finetune_identity.json", {"identity": identity})

    class TimingCallback(TrainerCallback):
        def __init__(self):
            self.samples = []
            self.started = None
        def on_step_begin(self, arguments, state, control, **kwargs):
            torch.cuda.synchronize()
            self.started = time.perf_counter()
        def on_step_end(self, arguments, state, control, **kwargs):
            torch.cuda.synchronize()
            if state.global_step > 2:
                self.samples.append(time.perf_counter() - self.started)

    timing = TimingCallback()
    dtype = gpu_precision(config)
    training_args = SFTConfig(
        dataset_kwargs={"skip_prepare_dataset": True}, packing=False, max_length=settings["max_seq_length"],
        output_dir=str(out), num_train_epochs=settings["epochs"], max_steps=max_steps,
        per_device_train_batch_size=settings["microbatch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=accumulation, learning_rate=settings["learning_rate"],
        weight_decay=settings["weight_decay"], max_grad_norm=settings["max_grad_norm"],
        lr_scheduler_type="cosine", warmup_ratio=settings["warmup_ratio"],
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        gradient_checkpointing=settings["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=1 if args.smoke_test else settings["save_steps"],
        save_strategy="steps", save_steps=1 if args.smoke_test else settings["save_steps"],
        save_total_limit=2, logging_steps=1 if max_steps > 0 else 5,
        report_to=[], seed=settings["seed"], data_seed=settings["seed"],
        ddp_find_unused_parameters=False, remove_unused_columns=False,
        dataloader_num_workers=0, optim="adamw_torch_fused", save_safetensors=True,
    )
    class MaskedTrainer(SFTTrainer):
        def _get_train_sampler(self, train_dataset=None):
            from transformers.trainer_pt_utils import LengthGroupedSampler
            rows = self.train_dataset if train_dataset is None else train_dataset
            return LengthGroupedSampler(batch_size=settings["microbatch_size"] * accumulation,
                                        lengths=[len(row["input_ids"]) for row in rows])

    trainer = MaskedTrainer(model=model, args=training_args, train_dataset=Dataset.from_list(train_rows),
                         eval_dataset=Dataset.from_list(val_rows),
                      data_collator=Collator(tokenizer.pad_token_id), processing_class=tokenizer,
                      callbacks=[IdentityCallback(), timing])
    tracked = [(name, p) for name, p in model.named_parameters() if p.requires_grad and "lora_B" in name]
    before = tracked[0][1].detach().clone() if args.smoke_test else None
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=args.resume)
    if not math.isfinite(result.training_loss):
        raise RuntimeError("Training loss is non-finite")
    trainer.save_model(str(out / "adapter"))
    if rank == 0:
        tokenizer.save_pretrained(out / "adapter")
        write_json(out / "adapter" / "deployment.json", {"config": config, "format": "peft_adapter"})
        write_json(out / "training_metrics.json", {**result.metrics, "wall_seconds": time.perf_counter() - started})
    metrics = trainer.evaluate()
    if rank == 0:
        write_json(out / "validation_loss.json", metrics)
    if args.benchmark and timing.samples and rank == 0:
        seconds = sum(timing.samples) / len(timing.samples)
        total_steps = math.ceil(len(train_records) / effective) * settings["epochs"]
        write_json(out / "benchmark.json", {"steady_optimizer_step_seconds": seconds, "estimated_training_hours": seconds * total_steps / 3600,
                                           "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                                           "training_tokens_per_optimizer_step": sum(len(r["input_ids"]) for r in train_rows) / len(train_rows) * effective,
                                           "training_tokens_per_second": sum(len(r["input_ids"]) for r in train_rows) / len(train_rows) * effective / seconds,
                                           "note": "Approximate non-padding input token throughput; excludes validation/downloads/checkpoints."})
    if args.smoke_test:
        if torch.equal(before, tracked[0][1].detach()):
            raise RuntimeError("Smoke test failed: LoRA weights did not update")
        # Exercise the actual Trainer resume path, including optimizer/scheduler/RNG loading.
        barrier()
        checkpoint = out / "checkpoint-1"
        trainer.args.max_steps = 2
        resumed = trainer.train(resume_from_checkpoint=str(checkpoint))
        if not math.isfinite(resumed.training_loss):
            raise RuntimeError("Smoke resume produced non-finite loss")
        if rank == 0:
            from peft import PeftModel
            # Reload exported adapter onto the already-loaded base without allocating another 4B model.
            base = model.unload()
            reloaded = PeftModel.from_pretrained(base, out / "adapter", is_trainable=False)
            reloaded.eval()
            batch = Collator(tokenizer.pad_token_id)(val_rows[:1])
            batch = {k: v.to(local_rank) for k, v in batch.items()}
            with torch.no_grad():
                loss = reloaded(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Exported adapter reload failed")
            write_json(out / "smoke_test.json", {"finite_loss": True, "weights_updated": True,
                                                "checkpoint_resume": True, "export_reload": True})
    barrier()
    if rank == 0:
        print(f"Training complete: {out / 'adapter'}")


def evaluate(args, config):
    from .vllm import evaluate as evaluate_vllm
    return evaluate_vllm(args, config)


def export(args, config):
    import unsloth  # Import before PEFT/Transformers patches.
    out = fresh_directory(args.output)
    if args.merge:
        model, tokenizer = load(config, args.checkpoint, merge_export=True)
        model = model.merge_and_unload()
        model.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    else:
        # Copy the self-contained adapter, not optimizer state. No model/GPU allocation needed.
        import shutil
        source = Path(args.checkpoint)
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (source / filename).is_file():
                raise ValueError(f"Not a PEFT adapter: missing {filename}")
        for path in source.iterdir():
            if path.is_file() and (path.suffix in (".json", ".safetensors", ".model", ".txt", ".jinja") or path.name == "README.md"):
                shutil.copy2(path, out / path.name)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(source)
    tokenizer.save_pretrained(out)
    write_json(out / "deployment.json", {"config": config, "format": "merged" if args.merge else "peft_adapter"})
    print(f"Exported: {out}")


def predict(args, config):
    from .vllm import predict as predict_vllm
    return predict_vllm(args, config)


def run(args, config):
    dispatch = {"train-llm": train, "eval-llm": evaluate, "export-llm": export, "predict": predict}
    dispatch[args.command](args, config)
