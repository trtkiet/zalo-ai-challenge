"""Resident asynchronous vLLM inference; no HTTP service or training dependencies."""
import asyncio
import json
import math
import os
from pathlib import Path
import time

from ..common import model_snapshot
from ..llm_common import action_schema, inference_settings, llm_identity, tokenizer_for


def parameter_count(directory, checkpoint=None):
    """Count the full architecture (including frozen vision) on meta, never a second GPU copy."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, Qwen3VLForConditionalGeneration
    cfg = AutoConfig.from_pretrained(directory, trust_remote_code=False)
    cls = Qwen3VLForConditionalGeneration if cfg.model_type == "qwen3_vl" else AutoModelForCausalLM
    with init_empty_weights():
        model = cls._from_config(cfg) if cfg.model_type == "qwen3_vl" else cls.from_config(cfg, trust_remote_code=False)
    model.tie_weights()
    count = sum(p.numel() for p in model.parameters())
    del model
    if checkpoint:
        from safetensors import safe_open
        with safe_open(str(Path(checkpoint) / "adapter_model.safetensors"), framework="pt", device="cpu") as tensors:
            if any("visual." in name or "vision_tower" in name for name in tensors.keys()):
                raise ValueError("Vision LoRA tensors are unsupported; export a merged model explicitly")
            count += sum(math.prod(tensors.get_slice(name).get_shape()) for name in tensors.keys())
    return count


class Engine:
    def __init__(self, config, checkpoint=None):
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from vllm.lora.request import LoRARequest
        self.config, self.settings = config, inference_settings(config)
        directory, revision = model_snapshot(config)
        config["resolved_model_revision"] = revision
        self.directory = directory
        source = Path(checkpoint) if checkpoint and (Path(checkpoint) / "tokenizer_config.json").is_file() else directory
        self.tokenizer = tokenizer_for(source, config)
        tokenizer_source = config.get("tokenizer", {}).get("model") or str(source)
        tokenizer_revision = (getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
                              or config.get("tokenizer", {}).get("revision"))
        self.lora = None
        adapter_rank = 16
        if checkpoint:
            adapter = json.loads((Path(checkpoint) / "adapter_config.json").read_text(encoding="utf-8"))
            if adapter.get("rank_pattern"):
                raise ValueError("Variable-rank adapters require an explicit merged export")
            targets = adapter.get("target_modules", [])
            if any("visual" in name for name in (targets if isinstance(targets, list) else [targets])):
                raise ValueError("vLLM adapter must target language layers only; use an explicit merged export for other adapters")
            rank = int(adapter["r"])
            if not 1 <= rank <= 512:
                raise ValueError("Unsupported adapter rank; use a merged export")
            adapter_rank = 2 ** math.ceil(math.log2(max(8, rank)))
            self.lora = LoRARequest("finetuned", 1, str(Path(checkpoint).resolve()))
        parameters = parameter_count(directory, checkpoint)
        if parameters >= 5_000_000_000:
            raise ValueError("LLM must have fewer than 5B parameters, including adapters")
        s = self.settings
        args = AsyncEngineArgs(model=str(directory), tokenizer=tokenizer_source, tokenizer_revision=tokenizer_revision,
                               trust_remote_code=False,
                               dtype="bfloat16" if s["precision"] == "bf16" else "half",
                               max_model_len=s["max_seq_length"], tensor_parallel_size=1,
                               gpu_memory_utilization=s["gpu_memory_utilization"],
                               max_num_seqs=s["max_num_seqs"], max_num_batched_tokens=s["max_num_batched_tokens"],
                               enable_prefix_caching=s["enable_prefix_caching"],
                               enable_chunked_prefill=s["enable_chunked_prefill"],
                               seed=config["training"].get("seed", 42), enable_lora=bool(checkpoint),
                               max_lora_rank=adapter_rank, limit_mm_per_prompt={"image": 0, "video": 0},
                               model_impl="vllm")
        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.metadata = {"ready": True, "backend": "vllm", "revision": revision, "parameters": parameters,
                         "identity": llm_identity(self.tokenizer), "inference": s}

    async def generate(self, request_id, payload, deadline):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        s = self.settings
        ids = payload.get("input_ids")
        if ids is None:
            ids = self.tokenizer.apply_chat_template(payload["messages"], tokenize=True, add_generation_prompt=True)
        if not ids or len(ids) + s["max_new_tokens"] > s["max_seq_length"]:
            raise ValueError(f"Context overflow/empty input: {len(ids)} input + {s['max_new_tokens']} output > {s['max_seq_length']}; no truncation")
        schema = payload.get("schema")
        if schema is None and "tools" in payload:
            schema = action_schema(payload["tools"], payload.get("max_calls", 12))
        if schema is None:
            # Generic predict messages still use the JSON action envelope; no tool-name filtering is possible.
            schema = {"type": "object", "properties": {"tool_calls": {"type": "array", "items": {
                "type": "object", "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
                "required": ["name", "arguments"], "additionalProperties": False}, "maxItems": 12}},
                "required": ["tool_calls"], "additionalProperties": False}
        params = SamplingParams(temperature=0, n=1, max_tokens=s["max_new_tokens"],
                                structured_outputs=StructuredOutputsParams(json=schema))
        started, first, last, result = time.monotonic(), None, None, None
        try:
            async with asyncio.timeout(max(0.001, deadline - time.monotonic())):
                async for output in self.engine.generate({"prompt_token_ids": ids}, params, request_id,
                                                         lora_request=self.lora):
                    now = time.monotonic()
                    if output.outputs and output.outputs[0].token_ids:
                        first = now if first is None else first
                        last = now
                    result = output
        except BaseException:
            await self.engine.abort(request_id)
            raise
        if result is None or not result.outputs:
            raise RuntimeError("vLLM returned no output")
        output = result.outputs[0]
        import torch
        free, total = torch.cuda.mem_get_info()
        return {"text": output.text, "input_tokens": len(ids), "output_tokens": len(output.token_ids),
                "ttft_seconds": first - started if first is not None else None,
                "decode_seconds": last - first if first is not None else None,
                "engine_seconds": time.monotonic() - started,
                "cached_tokens": getattr(result, "num_cached_tokens", None), "finish_reason": output.finish_reason,
                "gpu_memory_used_bytes": total - free}

    async def abort(self, request_id):
        await self.engine.abort(request_id)

    def close(self):
        self.engine.shutdown()


async def warm(engine):
    await engine.generate("warmup", {"messages": [{"role": "user", "content": 'Return {"tool_calls":[]}'}],
                                   "tools": [], "max_calls": 0}, time.monotonic() + 120)


async def predict_async(args, config):
    value = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if "messages" not in value:
        from ..data import trajectory
        messages, targets = trajectory(value)
        value = {"messages": messages[:targets[0]], "tools": value["tools"]}
    engine = Engine(config, args.checkpoint)
    try:
        await warm(engine)
        return await engine.generate("predict", value, time.monotonic() + engine.settings["max_seconds"])
    finally:
        engine.close()


def predict(args, config):
    if not args.input:
        raise ValueError("predict requires --input")
    print(asyncio.run(predict_async(args, config))["text"])


def evaluate(args, config):
    from ..common import fresh_directory, read_jsonl, write_json, write_jsonl
    from ..data import generation_cases, validate_action
    out = fresh_directory(args.output)
    records = read_jsonl(Path(args.prepared) / "llm_validation.jsonl")
    if args.limit:
        records = records[:args.limit]

    async def evaluate_all():
        engine = Engine(config, args.checkpoint)
        predictions = []
        counts = {"cases": 0, "json_valid": 0, "schema_valid": 0, "tool_name_correct": 0, "arguments_exact": 0}
        requested = 0
        try:
            await warm(engine)
            for record in records:
                requested += len(record["calls"]) + 1
                for case in generation_cases(engine.tokenizer, record, engine.settings["max_seq_length"], engine.settings["max_new_tokens"]):
                    result = await engine.generate(f"eval-{counts['cases']}", {"input_ids": case["input_ids"], "tools": case["tools"]},
                                                   time.monotonic() + engine.settings["max_seconds"])
                    try:
                        value = json.loads(result["text"])
                        valid_json = True
                    except json.JSONDecodeError:
                        value, valid_json = None, False
                    valid = valid_json and validate_action(value, case["tools"])
                    expected = case["expected"]
                    counts["cases"] += 1
                    counts["json_valid"] += valid_json
                    counts["schema_valid"] += valid
                    counts["tool_name_correct"] += bool(valid and [c["name"] for c in value["tool_calls"]] == [c["name"] for c in expected["tool_calls"]])
                    counts["arguments_exact"] += bool(valid and value == expected)
                    predictions.append({"id": case["sample_id"], "target_index": case["target_index"], "expected": expected, **result})
            return {**counts, "generation_cases_excluded_overlength": requested - counts["cases"],
                    "evaluation_type": "teacher_forced_next_step_vllm; not final app-state accuracy", "backend": engine.metadata}, predictions
        finally:
            engine.close()
    metrics, predictions = asyncio.run(evaluate_all())
    for name in ("json_valid", "schema_valid", "tool_name_correct", "arguments_exact"):
        metrics[name + "_rate"] = metrics[name] / metrics["cases"] if metrics["cases"] else None
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "predictions.jsonl", predictions)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
