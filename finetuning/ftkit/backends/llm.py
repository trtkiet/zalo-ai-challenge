"""HF causal/Qwen3-VL text-only SFT with current-assistant loss masking."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import time

from ..common import (fresh_directory, gpu_precision, model_snapshot, read_jsonl,
                      seed_everything, write_json, write_jsonl)
from ..data import encode_trajectory, generation_cases, trajectory, validate_action
from ..training import barrier, begin_run, distributed, validate_existing_output


def load(config, checkpoint=None, *, train=False, merge_export=False):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    dtype = torch.float32 if merge_export else gpu_precision(config)
    # Reject unsupported families after fetching only metadata, before any large weights.
    directory, revision = model_snapshot(config, ["config.json"])
    model_config = AutoConfig.from_pretrained(directory, trust_remote_code=False)
    handler = config["architecture"].get("handler", "auto")
    if model_config.model_type == "qwen3_vl" and handler in ("auto", "qwen3_vl"):
        cls = Qwen3VLForConditionalGeneration
    elif type(model_config) in AutoModelForCausalLM._model_mapping and handler in ("auto", "causal_lm"):
        cls = AutoModelForCausalLM
    else:
        raise ValueError(f"Unsupported architecture {model_config.model_type!r} with handler {handler!r}. Supported handlers: HF causal_lm and qwen3_vl.")
    directory, revision = model_snapshot(config)
    tokenizer_source = config.get("tokenizer", {}).get("model") or str(directory)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source,
                                             revision=config.get("tokenizer", {}).get("revision"), trust_remote_code=False)
    if not tokenizer.chat_template:
        template_path = directory / "chat_template.json"
        if template_path.is_file():
            template = json.loads(template_path.read_text(encoding="utf-8"))
            tokenizer.chat_template = template["chat_template"] if isinstance(template, dict) else template
    if not tokenizer.chat_template:
        raise ValueError("Model needs a chat template; configure a compatible tokenizer.")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs an existing pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    kwargs = {"torch_dtype": dtype, "trust_remote_code": False, "attn_implementation": "sdpa"}
    quantize = config["training"].get("quantize", True) and not merge_export
    kwargs["device_map"] = "cpu" if merge_export else {"": int(os.environ.get("LOCAL_RANK", "0"))}
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = cls.from_pretrained(directory, **kwargs)
    model.config.use_cache = not train
    if train and quantize:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config["training"]["gradient_checkpointing"])
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=train)
    elif train:
        projections = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        names = [name for name, _ in model.named_modules() if name.endswith(projections) and ".visual." not in name and not name.startswith("visual.")]
        if model_config.model_type == "qwen3_vl":
            names = [name for name in names if ".language_model." in name]
        if not names:
            raise ValueError("No supported language attention/MLP projections; add an architecture adapter.")
        settings = config["training"]
        lora = LoraConfig(r=settings["lora_rank"], lora_alpha=settings["lora_alpha"],
                          lora_dropout=settings["lora_dropout"], target_modules="(?:" + "|".join(re.escape(n) for n in names) + ")",
                          bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
    if train:
        if any(p.requires_grad for name, p in model.named_parameters() if ".visual." in name):
            raise ValueError("Vision parameters unexpectedly trainable")
        if config["training"]["gradient_checkpointing"]:
            model.enable_input_require_grads()
        model.print_trainable_parameters()
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
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments
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
                       "run_mode": "smoke" if args.smoke_test else "benchmark" if args.benchmark else "full"})
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
    training_args = TrainingArguments(
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
        dataloader_num_workers=0, optim="adamw_torch", save_safetensors=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_rows, eval_dataset=val_rows,
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
                                           "note": "Estimate excludes validation, downloads and checkpoint overhead; exclusions can change total steps."})
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
    import torch
    from transformers import Trainer, TrainingArguments
    seed_everything(config["training"]["seed"])
    out = fresh_directory(args.output)
    model, tokenizer = load(config, args.checkpoint)
    records = read_jsonl(Path(args.prepared) / "llm_validation.jsonl")
    if args.limit:
        records = records[:args.limit]
    maximum = config["training"]["max_seq_length"]
    rows, excluded = prepare(tokenizer, records, maximum)
    trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(out / "trainer"), report_to=[],
                      per_device_eval_batch_size=1, remove_unused_columns=False), eval_dataset=rows,
                      data_collator=Collator(tokenizer.pad_token_id))
    metrics = trainer.evaluate()
    model.eval()
    model.config.use_cache = True
    predictions = []
    counts = {"cases": 0, "json_valid": 0, "schema_valid": 0, "tool_name_correct": 0, "arguments_exact": 0}
    requested_cases = 0
    for record in records:
        requested_cases += len(record["calls"]) + 1
        for case in generation_cases(tokenizer, record, maximum, config["training"]["generation_max_new_tokens"]):
            ids = torch.tensor([case["input_ids"]], device=model.device)
            with torch.no_grad():
                output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                        max_new_tokens=config["training"]["generation_max_new_tokens"], pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
            try:
                value = json.loads(text)
                valid_json = True
            except json.JSONDecodeError:
                value, valid_json = None, False
            valid_schema = valid_json and validate_action(value, case["tools"])
            expected = case["expected"]
            names = lambda v: [c["name"] for c in v["tool_calls"]]
            counts["cases"] += 1
            counts["json_valid"] += valid_json
            counts["schema_valid"] += valid_schema
            counts["tool_name_correct"] += bool(valid_schema and names(value) == names(expected))
            counts["arguments_exact"] += bool(valid_schema and value == expected)
            predictions.append({"id": case["sample_id"], "target_index": case["target_index"], "prediction": text,
                                "expected": expected, "schema_valid": bool(valid_schema)})
    metrics.update(counts)
    metrics["generation_cases_excluded_overlength"] = requested_cases - counts["cases"]
    metrics["training_examples_excluded_overlength"] = excluded
    for key in ("json_valid", "schema_valid", "tool_name_correct", "arguments_exact"):
        metrics[key + "_rate"] = counts[key] / counts["cases"] if counts["cases"] else None
    metrics["evaluation_type"] = "teacher_forced_next_step; does not execute tools or measure final app state"
    write_json(out / "metrics.json", metrics)
    write_jsonl(out / "predictions.jsonl", predictions)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def export(args, config):
    from peft import PeftModel
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
    import torch
    if not args.input:
        raise ValueError("predict requires --input JSON with messages or a prepared-format record")
    value = json.loads(Path(args.input).read_text(encoding="utf-8"))
    messages = value.get("messages")
    if messages is None:
        messages, targets = trajectory(value)
        messages = messages[:targets[0]]
    model, tokenizer = load(config, args.checkpoint)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if len(ids) + config["training"]["generation_max_new_tokens"] > config["training"]["max_seq_length"]:
        raise ValueError("Input plus generation budget exceeds context; nothing was truncated")
    tensor = torch.tensor([ids], device=model.device)
    model.eval()
    with torch.no_grad():
        output = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor), do_sample=False,
                                max_new_tokens=config["training"]["generation_max_new_tokens"], pad_token_id=tokenizer.pad_token_id)
    print(tokenizer.decode(output[0, len(ids):], skip_special_tokens=True))


def run(args, config):
    dispatch = {"train-llm": train, "eval-llm": evaluate, "export-llm": export, "predict": predict}
    dispatch[args.command](args, config)
