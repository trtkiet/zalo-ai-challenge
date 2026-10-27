"""Private JSON-line protocol for resident model processes. Models never access ToolEnv."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time
import types
import wave

from .common import compact, gpu_precision, model_snapshot, read_config, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["asr", "llm"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--icefall-dir")
    args = parser.parse_args()
    config = read_config(args.config)
    protocol = sys.stdout

    def send(value):
        print(compact(value), file=protocol, flush=True)

    try:
        # All library/model diagnostics go to stderr; stdout is a protocol only.
        with redirect_stdout(sys.stderr):
            import torch
            seed_everything(config.get("training", {}).get("seed", 42))
            gpu_precision(config)
            backend = config["backend"]
            if args.kind == "llm":
                from .backends.llm import load
                model, tokenizer = load(config, args.checkpoint)
                model.eval()
                model.config.use_cache = True
                parameter_count = model.get_nb_trainable_parameters()[1] if hasattr(model, "get_nb_trainable_parameters") else model.num_parameters()
                # Warm kernels once, outside sample timing.
                token = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
                warm = torch.tensor([[token]], device=model.device)
                with torch.inference_mode():
                    model.generate(input_ids=warm, attention_mask=torch.ones_like(warm), max_new_tokens=1,
                                   do_sample=False, pad_token_id=tokenizer.pad_token_id)
            elif backend == "zipformer_rnnt":
                from .backends.asr import load, features
                namespace = types.SimpleNamespace(icefall_dir=args.icefall_dir)
                loaded = load(namespace, config, args.checkpoint)
                model = loaded["model"]
                model.eval()
                parameter_count = sum(p.numel() for p in model.parameters())
                values = loaded["fbank"]([torch.zeros(16000, device=next(model.parameters()).device)])
                with torch.inference_mode():
                    length = torch.tensor([len(values[0])], device=values[0].device)
                    model.forward_encoder(values[0].unsqueeze(0), length)
            elif backend == "hf_whisper":
                from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoProcessor
                if args.checkpoint:
                    config["model"] = args.checkpoint
                    config["revision"] = "main"
                directory, revision = model_snapshot(config, ["config.json"])
                if AutoConfig.from_pretrained(directory).model_type != "whisper":
                    raise ValueError("hf_whisper supports Whisper/PhoWhisper checkpoints only")
                directory, revision = model_snapshot(config)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(directory, torch_dtype=gpu_precision(config)).to("cuda")
                processor = AutoProcessor.from_pretrained(directory)
                model.eval()
                model.generation_config.forced_decoder_ids = None
                config["resolved_model_revision"] = revision
                parameter_count = model.num_parameters()
            else:
                raise ValueError(f"Unsupported inference backend: {backend}")
            torch.cuda.synchronize()
        send({"ready": True, "backend": backend, "revision": config.get("resolved_model_revision"), "parameters": parameter_count})
    except Exception as exc:
        send({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("operation") == "shutdown":
                return
            with redirect_stdout(sys.stderr), torch.inference_mode():
                seconds = max(0.001, float(request["seconds"]))
                started = time.monotonic()
                if request.get("operation", "").endswith("_batch"):
                    from .batched_inference import generate_batch, transcribe_batch
                    requests = request["requests"]
                    if not isinstance(requests, list) or not requests:
                        raise ValueError("A model batch requires a nonempty request list")
                    if args.kind == "llm":
                        results = generate_batch(model, tokenizer, config, requests, seconds)
                    else:
                        results = transcribe_batch(model, config, requests, seconds,
                                                   loaded=loaded if backend == "zipformer_rnnt" else None,
                                                   processor=processor if backend == "hf_whisper" else None)
                    value = {"results": results}
                elif args.kind == "llm":
                    ids = tokenizer.apply_chat_template(request["messages"], tokenize=True, add_generation_prompt=True)
                    maximum = min(16384, int(config["training"]["max_seq_length"]))
                    budget = int(config["training"].get("generation_max_new_tokens", 512))
                    if len(ids) + budget > maximum:
                        raise ValueError(f"Context overflow: {len(ids)} input + {budget} output tokens > {maximum}; no truncation")
                    tensor = torch.tensor([ids], device=model.device)
                    result = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor), do_sample=False,
                                            max_new_tokens=budget, max_time=seconds, pad_token_id=tokenizer.pad_token_id)
                    value = {"text": tokenizer.decode(result[0, len(ids):], skip_special_tokens=True), "input_tokens": len(ids)}
                else:
                    texts = []
                    for path in request["audio"]:
                        if time.monotonic() - started >= seconds:
                            raise TimeoutError("ASR time budget exhausted")
                        if backend == "zipformer_rnnt":
                            value, lengths = features(loaded, [path])
                            encoded, encoded_lengths = model.forward_encoder(value, lengths)
                            tokens = loaded["decode"](model=model, encoder_out=encoded, encoder_out_lens=encoded_lengths)
                            texts.append(loaded["processor"].decode(tokens[0]))
                        else:
                            import numpy as np
                            with wave.open(str(path), "rb") as audio:
                                if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
                                    raise ValueError("Whisper input must be mono 16kHz PCM16 WAV")
                                waveform = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768
                            parts = []
                            chunk_seconds = float(config["architecture"].get("chunk_seconds", 30))
                            if not 0 < chunk_seconds <= 30:
                                raise ValueError("Whisper chunk_seconds must be >0 and <=30")
                            chunk_samples = max(1, int(chunk_seconds * 16000))
                            for offset in range(0, len(waveform), chunk_samples):
                                remaining = seconds - (time.monotonic() - started)
                                if remaining <= 0:
                                    raise TimeoutError("Whisper time budget exhausted")
                                inputs = processor(waveform[offset:offset + chunk_samples], sampling_rate=16000, return_tensors="pt")
                                features_tensor = inputs["input_features"].to(device=model.device, dtype=model.dtype)
                                result = model.generate(input_features=features_tensor, language=config["architecture"].get("language", "vi"),
                                                        task="transcribe", do_sample=False, num_beams=1, max_new_tokens=448, max_time=remaining)
                                parts.append(processor.batch_decode(result, skip_special_tokens=True)[0])
                            texts.append(" ".join(parts).strip())
                    value = {"texts": texts}
                torch.cuda.synchronize()
            send({"ok": True, **value})
        except Exception as exc:
            send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
