"""Neural batch operations. GPU dependencies remain lazy for CPU-only tooling."""
import time
import wave


def left_pad(rows, pad_id):
    """Pad token IDs and masks on the left for decoder-only generation."""
    width = max(map(len, rows))
    return ([([pad_id] * (width - len(row))) + list(row) for row in rows],
            [([0] * (width - len(row))) + [1] * len(row) for row in rows])


def error_result(exc):
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "error_type": type(exc).__name__}


def seconds_left(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("Model batch time budget exhausted")
    return seconds


def generate_batch(model, tokenizer, config, requests, seconds):
    import torch

    deadline = time.monotonic() + seconds
    maximum = min(16384, int(config["training"]["max_seq_length"]))
    budget = int(config["training"].get("generation_max_new_tokens", 512))
    responses = [None] * len(requests)
    indices, rows = [], []
    for index, request in enumerate(requests):
        try:
            ids = tokenizer.apply_chat_template(request["messages"], tokenize=True, add_generation_prompt=True)
            if not ids:
                raise ValueError("Empty tokenized prompt")
            if len(ids) + budget > maximum:
                raise ValueError(f"Context overflow: {len(ids)} input + {budget} output tokens > {maximum}; no truncation")
            indices.append(index)
            rows.append(ids)
        except Exception as exc:
            responses[index] = error_result(exc)
    if rows:
        try:
            padded, mask = left_pad(rows, tokenizer.pad_token_id)
            tensor = torch.tensor(padded, device=model.device)
            attention_mask = torch.tensor(mask, device=model.device)
            generated = model.generate(input_ids=tensor, attention_mask=attention_mask, do_sample=False,
                                       num_beams=1, max_new_tokens=budget, max_time=seconds_left(deadline),
                                       pad_token_id=tokenizer.pad_token_id)
            texts = tokenizer.batch_decode(generated[:, len(padded[0]):], skip_special_tokens=True)
            if len(texts) != len(rows):
                raise ValueError("LLM returned mismatched batch size")
            for index, ids, text in zip(indices, rows, texts):
                responses[index] = {"ok": True, "text": text, "input_tokens": len(ids)}
        except Exception as exc:
            # Do not silently retry an OOM with a different batch/decoding setup.
            for index in indices:
                responses[index] = error_result(exc)
    return responses


def transcribe_batch(model, config, requests, seconds, *, loaded=None, processor=None):
    """Batch clips/chunks across samples; restore their original audio ordering.

    Clip batches are bounded by the number of active samples, so a sample with
    long audio history does not turn a small sample batch into an unbounded one.
    """
    deadline = time.monotonic() + seconds
    width = max(1, len(requests))
    responses = [None] * len(requests)
    clips, jobs = {}, []
    for sample, request in enumerate(requests):
        try:
            if not request["audio"]:
                raise ValueError("No audio supplied")
            for audio_index, path in enumerate(request["audio"]):
                clips[sample, audio_index] = []
                if config["backend"] == "zipformer_rnnt":
                    # Validate each input independently before the shared encoder.
                    with wave.open(str(path), "rb") as audio:
                        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
                            raise ValueError("ASR input must be mono 16kHz PCM16 WAV")
                        if not audio.getnframes():
                            raise ValueError("Empty audio")
                    jobs.append((sample, audio_index, path))
                else:
                    import numpy as np
                    with wave.open(str(path), "rb") as audio:
                        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
                            raise ValueError("Whisper input must be mono 16kHz PCM16 WAV")
                        waveform = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32) / 32768
                    if not len(waveform):
                        raise ValueError("Empty audio")
                    chunk_seconds = float(config["architecture"].get("chunk_seconds", 30))
                    if not 0 < chunk_seconds <= 30:
                        raise ValueError("Whisper chunk_seconds must be >0 and <=30")
                    chunk_samples = max(1, int(chunk_seconds * 16000))
                    for offset in range(0, len(waveform), chunk_samples):
                        jobs.append((sample, audio_index, waveform[offset:offset + chunk_samples]))
        except Exception as exc:
            responses[sample] = error_result(exc)
    jobs = [job for job in jobs if responses[job[0]] is None]
    for offset in range(0, len(jobs), width):
        batch = [job for job in jobs[offset:offset + width] if responses[job[0]] is None]
        if not batch:
            continue
        try:
            seconds_left(deadline)
            if config["backend"] == "zipformer_rnnt":
                from .backends.asr import features
                values, lengths = features(loaded, [job[2] for job in batch])
                encoded, encoded_lengths = model.forward_encoder(values, lengths)
                tokens = loaded["decode"](model=model, encoder_out=encoded, encoder_out_lens=encoded_lengths)
                texts = [loaded["processor"].decode(row) for row in tokens]
            else:
                inputs = processor([job[2] for job in batch], sampling_rate=16000,
                                   return_tensors="pt", return_attention_mask=True)
                values = inputs["input_features"].to(device=model.device, dtype=model.dtype)
                mask = inputs.get("attention_mask")
                kwargs = {"attention_mask": mask.to(model.device)} if mask is not None else {}
                generated = model.generate(input_features=values, language=config["architecture"].get("language", "vi"),
                                           task="transcribe", do_sample=False, num_beams=1, max_new_tokens=448,
                                           max_time=seconds_left(deadline), **kwargs)
                texts = processor.batch_decode(generated, skip_special_tokens=True)
            if len(texts) != len(batch):
                raise ValueError("ASR returned mismatched batch size")
            for (sample, audio_index, _), text in zip(batch, texts):
                clips[sample, audio_index].append(text)
        except Exception as exc:
            for sample, _, _ in batch:
                responses[sample] = error_result(exc)
    for sample, request in enumerate(requests):
        if responses[sample] is None:
            responses[sample] = {"ok": True, "texts": [" ".join(clips[sample, i]).strip()
                                                      for i in range(len(request["audio"]))]}
    return responses
