from __future__ import annotations

import hashlib
import json
import random
import wave
from pathlib import Path

from .common import compact, digest, fresh_directory, normalize, read_jsonl, write_json, write_jsonl


def user_utterances(sample):
    for index, turn in enumerate(sample.get("history", [])):
        if turn.get("role") == "user":
            yield f"{sample['id']}_h{index}", turn["audio"], turn["audio_text"]
    yield sample["id"], sample["audio"], sample["audio_text"]


def grouped_split(samples, validation_fraction=0.1, seed=42):
    """Union complete conversations sharing any normalized utterance, then split groups."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    parent = list(range(len(samples)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, sample in enumerate(samples):
        for _, _, text in user_utterances(sample):
            key = normalize(text)
            if not key:
                raise ValueError(f"Empty transcript in {sample['id']}")
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i in range(len(samples)):
        groups.setdefault(find(i), []).append(samples[i]["id"])
    groups = sorted(groups.values(), key=lambda g: tuple(sorted(g)))
    random.Random(seed).shuffle(groups)
    if len(groups) < 2:
        raise ValueError("Cannot create an isolated validation set: all samples form one connected group.")
    target = max(1, round(len(samples) * validation_fraction))
    validation = []
    # Reserve at least one group for training, even for tiny datasets.
    for group in groups[:-1]:
        if len(validation) < target:
            validation.extend(group)
    val = set(validation)
    return {"train": sorted(s["id"] for s in samples if s["id"] not in val), "validation": sorted(val)}


def safe_audio(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or any(p.endswith("_environment") for p in relative.parts):
        raise ValueError(f"Unsafe audio path: {relative}")
    path = (root / "train" / relative).resolve()
    allowed = (root / "train" / "audio").resolve()
    if not path.is_relative_to(allowed):
        raise ValueError(f"Audio must stay inside train/audio: {relative}")
    return path


def prepare(data_root, output, validation_fraction=0.1, seed=42):
    root = Path(data_root).expanduser().resolve()
    samples = sorted(read_jsonl(root / "train" / "train.jsonl"), key=lambda x: x["id"])
    if len({x["id"] for x in samples}) != len(samples):
        raise ValueError("Duplicate sample IDs")
    catalog_path = root / "train" / "train_tools_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    tools = {t["name"]: t for app in catalog["apps"] for t in app["tools"]}
    split = grouped_split(samples, validation_fraction, seed)
    partition = {sid: name for name, ids in split.items() for sid in ids}
    asr = {name: [] for name in split}
    llm = {name: [] for name in split}
    audio_hashes = {}
    for sample in samples:
        name = partition[sample["id"]]
        if len(sample["label"]["tool_calls"]) != len(sample["label"]["tool_results"]):
            raise ValueError(f"Mismatched calls/results in {sample['id']}")
        for uid, relative, text in user_utterances(sample):
            path = safe_audio(root, relative)
            with wave.open(str(path), "rb") as wav:
                if wav.getnchannels() != 1 or wav.getframerate() != 16000 or wav.getsampwidth() != 2:
                    raise ValueError(f"Expected 16kHz mono PCM16: {path}")
                duration = wav.getnframes() / wav.getframerate()
            if duration <= 0:
                raise ValueError(f"Empty WAV: {path}")
            # Hash audio too; replacing a waveform must invalidate a resume fingerprint.
            file_hash = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    file_hash.update(chunk)
            audio_hashes[relative] = file_hash.hexdigest()
            asr[name].append({"id": uid, "sample_id": sample["id"], "audio": relative,
                              "text": normalize(text), "raw_text": text, "duration": duration})
        installed = [n for app in sample["context"]["installed_apps"].values() for n in app["tools"]]
        if "sys.ask_missing_info" not in installed:
            installed.append("sys.ask_missing_info")
        unknown = set(installed) - tools.keys()
        if unknown:
            raise ValueError(f"Unknown installed tools in {sample['id']}: {sorted(unknown)}")
        # Explicitly omit final_state. Only chronological calls/results are retained.
        llm[name].append({"id": sample["id"], "context": sample["context"],
                          "history": sample.get("history", []), "request": sample["audio_text"],
                          "tools": [tools[n] for n in sorted(set(installed))],
                          "calls": sample["label"]["tool_calls"], "results": sample["label"]["tool_results"]})
    out = fresh_directory(output)
    for name in split:
        write_jsonl(out / f"asr_{name}.jsonl", asr[name])
        write_jsonl(out / f"llm_{name}.jsonl", llm[name])
    write_json(out / "split_ids.json", split)
    source_fingerprint = digest({"samples": samples, "catalog": catalog, "audio": audio_hashes})
    metadata = {"format_version": 1, "data_root": str(root), "seed": seed,
                "validation_fraction": validation_fraction, "source_fingerprint": source_fingerprint,
                "fingerprint": digest({"source": source_fingerprint, "split": split}),
                "normalization": "Unicode NFC; lowercase; punctuation to spaces; collapse whitespace; preserve diacritics and numeric spelling",
                "counts": {n: {"conversations": len(llm[n]), "audio_clips": len(asr[n]),
                               "audio_hours": sum(x["duration"] for x in asr[n]) / 3600} for n in split}}
    write_json(out / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


SYSTEM = """Bạn là trợ lý thực hiện yêu cầu qua công cụ. Chỉ trả về JSON hợp lệ theo dạng
{\"tool_calls\":[{\"name\":\"app.tool\",\"arguments\":{}}]}. Có thể chọn nhiều lời gọi độc lập.
Các lời gọi được thực hiện tuần tự. Chỉ gộp khi mọi tham số và điều kiện thực hiện đã biết.
Nếu cần ID, kết quả đọc hoặc điều kiện từ công cụ, dừng danh sách và chờ kết quả trước khi chọn hành động tiếp.
Khi đã xong hoặc yêu cầu ngoài phạm vi, trả về {\"tool_calls\":[]}.
Chỉ dùng công cụ được cung cấp. Lấy ID từ lịch sử/ngữ cảnh/kết quả công cụ; không tự bịa.
Đọc dữ liệu cần thiết trước khi quyết định hành động. Nếu còn thiếu thông tin, dùng
sys.ask_missing_info với tên công cụ và tên tham số thực sự còn thiếu.
Lịch sử đã được áp dụng; không thực hiện lại các thao tác cũ.
"""


def action(calls):
    return compact({"tool_calls": calls})


def trajectory(record):
    """Return messages and indexes of current-turn assistant targets only."""
    messages = [{"role": "system", "content": SYSTEM + "\nTOOLS:\n" + compact(record["tools"])}]
    for turn in record["history"]:
        if turn["role"] == "user":
            messages.append({"role": "user", "content": turn["audio_text"]})
        elif turn["role"] == "assistant":
            if turn.get("tool_calls"):
                results = turn.get("tool_results", [])
                for i, call in enumerate(turn["tool_calls"]):
                    messages.append({"role": "assistant", "content": action([call])})
                    if i < len(results):
                        messages.append({"role": "tool", "content": compact({"name": call["name"], "result": results[i]})})
            if turn.get("text"):
                messages.append({"role": "assistant", "content": turn["text"]})
    messages.append({"role": "user", "content": "CONTEXT:\n" + compact(record["context"]) + "\nREQUEST:\n" + record["request"]})
    targets = []
    for call, result in zip(record["calls"], record["results"]):
        targets.append(len(messages))
        messages.append({"role": "assistant", "content": action([call])})
        messages.append({"role": "tool", "content": compact({"name": call["name"], "result": result})})
    targets.append(len(messages))
    messages.append({"role": "assistant", "content": action([])})
    return messages, targets


def encode_trajectory(tokenizer, messages, targets, max_length):
    """Mask rendered prefixes; reject templates that rewrite earlier tokens."""
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if len(ids) > max_length:
        return None
    labels = [-100] * len(ids)
    for index in targets:
        before = tokenizer.apply_chat_template(messages[:index], tokenize=True, add_generation_prompt=True)
        through = tokenizer.apply_chat_template(messages[:index + 1], tokenize=True, add_generation_prompt=False)
        if ids[:len(through)] != through or through[:len(before)] != before:
            raise ValueError("Chat template rewrites prefixes; add a masking adapter for this model instead of training incorrect labels.")
        labels[len(before):len(through)] = ids[len(before):len(through)]
    if not any(t != -100 for t in labels):
        raise ValueError("No assistant target tokens")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def generation_cases(tokenizer, record, max_length, max_new_tokens):
    messages, targets = trajectory(record)
    for index in targets:
        ids = tokenizer.apply_chat_template(messages[:index], tokenize=True, add_generation_prompt=True)
        if len(ids) + max_new_tokens <= max_length:
            yield {"sample_id": record["id"], "target_index": index, "input_ids": ids,
                   "expected": json.loads(messages[index]["content"]), "tools": record["tools"]}


def validate_action(value, tools):
    """Check JSON envelope, availability and catalog schema without executing tools."""
    if not isinstance(value, dict) or set(value) != {"tool_calls"} or not isinstance(value["tool_calls"], list):
        return False
    catalog = {t["name"]: t for t in tools}
    for call in value["tool_calls"]:
        if not isinstance(call, dict) or set(call) != {"name", "arguments"} or not isinstance(call["name"], str) or call["name"] not in catalog:
            return False
        args = call["arguments"]
        if not isinstance(args, dict):
            return False
        schema = catalog[call["name"]]["parameters"]
        if not schema_value(args, schema):
            return False
        if any(args.get(k) in (None, "", [], {}) for k in schema.get("required", [])):
            return False
        groups = schema.get("x-one-of", [])
        if groups and not any(all(args.get(k) not in (None, "", [], {}) for k in group) for group in groups):
            return False
    return True


def schema_value(value, schema):
    types = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool}
    kind = schema.get("type")
    if kind in types and (not isinstance(value, types[kind]) or (kind in ("integer", "number") and isinstance(value, bool))):
        return False
    if "enum" in schema and not any(type(value) == type(x) and value == x for x in schema["enum"]):
        return False
    if isinstance(value, dict):
        if any(k not in value for k in schema.get("required", [])):
            return False
        return all(schema_value(v, schema.get("properties", {}).get(k, {})) for k, v in value.items())
    if isinstance(value, list):
        return all(schema_value(x, schema.get("items", {})) for x in value)
    if kind in ("integer", "number"):
        import math
        if not math.isfinite(value):
            return False
        for bound, check in (("minimum", lambda x: value >= x), ("maximum", lambda x: value <= x),
                             ("exclusiveMinimum", lambda x: value > x), ("exclusiveMaximum", lambda x: value < x)):
            if bound in schema and not check(schema[bound]):
                return False
    if kind == "string" and schema.get("format"):
        from datetime import date, datetime, time
        parser = {"date": date.fromisoformat, "date-time": datetime.fromisoformat, "time": time.fromisoformat}.get(schema["format"])
        try:
            if parser:
                parser(value.replace("Z", "+00:00"))
        except ValueError:
            return False
    return True
