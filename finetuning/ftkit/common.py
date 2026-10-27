from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import random
import re
import subprocess
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_config(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("Install PyYAML to read conventional YAML; bundled configs are JSON-compatible YAML.") from exc
        value = yaml.safe_load(text)
        if not isinstance(value, dict):
            raise ValueError("Configuration must be a mapping")
        return value


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def resolve_config(args, kind):
    path = Path(args.config) if args.config else ROOT / "configs" / "models" / ("gipformer.yaml" if kind == "asr" else "qwen_sealion.yaml")
    config = read_config(path)
    # Exports carry their training architecture/base-model identity. Reuse it by
    # default; an explicit --config remains authoritative.
    if not args.config:
        candidate = getattr(args, "model", None) or getattr(args, "checkpoint", None)
        if candidate and Path(candidate).is_dir() and (Path(candidate) / "deployment.json").is_file():
            config = read_config(Path(candidate) / "deployment.json")["config"]
    profile = Path(args.profile)
    if not profile.is_file():
        profile = ROOT / "configs" / "profiles" / f"{args.profile}.yaml"
    config = merge(config, read_config(profile))
    if args.model:
        config["model"] = args.model
        # A source-specific default revision must never follow a model override.
        config["revision"] = "main"
    if args.revision:
        config["revision"] = args.revision
    if getattr(args, "tokenizer", None):
        config["tokenizer"]["model"] = args.tokenizer
    for entry in getattr(args, "set", []) or []:
        key, sep, value = entry.partition("=")
        if not sep:
            raise ValueError("--set requires dotted.key=JSON_VALUE")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    expected = "zipformer_rnnt" if kind == "asr" else "hf_sft"
    if config.get("backend") != expected:
        raise ValueError(f"This command needs backend {expected}; got {config.get('backend')}. Add a backend for another architecture.")
    config["config_path"] = str(path.resolve())
    return config


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(compact(row) + "\n")


def digest(value):
    return hashlib.sha256(compact(value).encode("utf-8")).hexdigest()


def normalize(text):
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split())


def fresh_directory(path):
    path = Path(path).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output already contains files: {path}. Choose a new directory or explicitly resume a training checkpoint.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_snapshot(config, patterns=None):
    source = Path(config["model"]).expanduser()
    if source.is_dir():
        return source.resolve(), "local:" + str(source.resolve())
    from huggingface_hub import snapshot_download
    directory = Path(snapshot_download(config["model"], revision=config.get("revision", "main"), allow_patterns=patterns))
    return directory, directory.name


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # k2 custom CUDA kernels do not guarantee bitwise determinism.
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def gpu_precision(config):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for training/evaluation. CPU-only preparation and unit tests work without torch.")
    value = config.get("precision", "auto")
    bf16 = torch.cuda.is_bf16_supported()
    if value == "bf16" and not bf16:
        raise ValueError("BF16 is unsupported on this GPU; use precision=auto or fp16.")
    if value not in ("auto", "bf16", "fp16"):
        raise ValueError("precision must be auto, bf16, or fp16")
    return torch.bfloat16 if value == "bf16" or (value == "auto" and bf16) else torch.float16


def runtime_manifest(config, prepared=None):
    versions = {}
    for name in ("torch", "torchaudio", "transformers", "peft", "accelerate", "bitsandbytes", "k2", "kaldifeat", "lhotse", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    result = {"config": config, "packages": versions}
    if prepared:
        result["dataset"] = json.loads((Path(prepared) / "metadata.json").read_text(encoding="utf-8"))
    try:
        result["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        result["nvidia_smi"] = "unavailable"
    return result
