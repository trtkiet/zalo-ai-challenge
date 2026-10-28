"""Shared LLM contracts. Importable without Torch, Transformers or CUDA."""
import copy
import importlib.metadata
import json
import sys
from pathlib import Path

from .common import digest
from .data import SYSTEM

PROTOCOL_VERSION = 2
SCHEDULER_VERSION = "continuous_v1"


def llm_identity(tokenizer=None):
    versions = {}
    for name in ("unsloth", "unsloth-zoo", "vllm", "trl", "torch", "transformers", "peft"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"protocol": PROTOCOL_VERSION, "prompt_sha256": digest(SYSTEM), "packages": versions,
            "python_version": list(sys.version_info[:3]),
            "implementation_sha256": digest({str(p.relative_to(Path(__file__).parent)): p.read_text(encoding="utf-8")
                                               for p in sorted(Path(__file__).parent.rglob("*.py"))}),
            "template_sha256": digest(tokenizer.chat_template) if tokenizer is not None else None,
            "tokenizer_vocab_sha256": digest(sorted(tokenizer.get_vocab().items())) if tokenizer is not None and hasattr(tokenizer, "get_vocab") else None,
            "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") if tokenizer is not None else None}


def tokenizer_for(directory, config):
    from transformers import AutoTokenizer
    source = config.get("tokenizer", {}).get("model") or str(directory)
    tokenizer = AutoTokenizer.from_pretrained(source, revision=config.get("tokenizer", {}).get("revision"),
                                             trust_remote_code=False)
    template_path = Path(directory) / "chat_template.json"
    if not tokenizer.chat_template and template_path.is_file():
        template = json.loads(template_path.read_text(encoding="utf-8"))
        tokenizer.chat_template = template["chat_template"] if isinstance(template, dict) else template
    if not tokenizer.chat_template:
        raise ValueError("Model needs a chat template; configure a compatible tokenizer")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs an existing pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def normalize_schema(value):
    """Translate catalog extensions; state-dependent x-ref remains a runtime check."""
    if isinstance(value, list):
        return [normalize_schema(v) for v in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, child in value.items():
        if key.startswith("x-") or key in ("description", "title", "examples", "default"):
            continue
        if key in ("properties", "$defs", "definitions", "patternProperties"):
            result[key] = {name: normalize_schema(schema) for name, schema in child.items()}
        elif key in ("items", "additionalProperties", "anyOf", "allOf", "oneOf", "not", "if", "then", "else", "prefixItems"):
            result[key] = normalize_schema(child)
        else:
            result[key] = copy.deepcopy(child)
    groups = value.get("x-one-of")
    if groups:
        clauses = [{"required": group if isinstance(group, list) else [group]} for group in groups]
        result.setdefault("allOf", []).append({"anyOf": clauses})
    return result


def action_schema(tools, max_calls=12):
    if max_calls < 0:
        raise ValueError("Negative call budget")
    variants = []
    for tool in tools:
        variants.append({"type": "object", "properties": {
            "name": {"type": "string", "const": tool["name"]},
            "arguments": normalize_schema(copy.deepcopy(tool["parameters"]))},
            "required": ["name", "arguments"], "additionalProperties": False})
    array = {"type": "array", "maxItems": max_calls if variants else 0}
    if variants:
        array["items"] = {"anyOf": variants}
    else:
        array["items"] = {"type": "object"}
    return {"type": "object", "properties": {"tool_calls": array},
            "required": ["tool_calls"], "additionalProperties": False}


def inference_settings(config):
    value = {"precision": "bf16", "gpu_memory_utilization": 0.70, "max_num_batched_tokens": 4096,
             "max_num_seqs": 1, "enable_prefix_caching": True, "enable_chunked_prefill": True,
             "max_new_tokens": config["training"].get("generation_max_new_tokens", 512),
             "max_seq_length": min(16384, config["training"]["max_seq_length"]), "max_seconds": 30}
    value.update(config.get("inference", {}))
    if not 0 < value["gpu_memory_utilization"] < 1:
        raise ValueError("inference.gpu_memory_utilization must be between 0 and 1")
    if value["precision"] not in ("bf16", "fp16"):
        raise ValueError("inference.precision must be bf16 or fp16")
    if not 0 < value["max_seq_length"] <= 16384 or value["max_new_tokens"] < 1:
        raise ValueError("Invalid inference context/output budget")
    if value["max_num_seqs"] < 1 or value["max_num_batched_tokens"] < 1:
        raise ValueError("Invalid inference batching limits")
    return value
