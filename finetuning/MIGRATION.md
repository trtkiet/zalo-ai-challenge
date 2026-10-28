# vLLM / Unsloth migration

The LLM now trains with Unsloth-backed TRL and generates with a resident async
vLLM engine. The old standalone Transformers loader/generator is removed.
Gipformer training and inference are retained. PhoWhisper inference uses the
new `llm-infer` environment for its Transformers dependencies.

## Install on the Linux GPU server

Use Python 3.11 for both LLM environments. ASR retains its existing Torch/k2
pins and CUDA 12.8 driver requirement. Each LLM framework resolves its own Torch
dependencies; its CUDA wheel must support the installed NVIDIA driver.

```bash
export PYTHON=python3.11
bash finetune.sh setup llm
# Or: bash finetune.sh setup llm-train / setup llm-infer
```

This creates `.venvs/llm-train` and `.venvs/llm-infer` under `FINETUNE_HOME`.
The old `.venvs/llm` is neither used nor deleted. Setup refuses to overwrite an
incomplete or incompatible environment; use a fresh `FINETUNE_HOME` to rebuild.
Compatible new environments can be reused. Install ASR separately if necessary.

`requirements/llm-train.txt` and `llm-infer.txt` are **bootstrap constraints**,
not validated locks. Setup records a resolved freeze and checks imports, CUDA,
BF16 support, and `pip check`. No real model compatibility or speed is established
by installation alone. After successful server checks, `gpu_validate.py
--write-locks` creates separate lock files for review and committing. They are
used by subsequent clean setups. No untested lock is presented as verified.

## Training and exports

```bash
bash finetune.sh train-llm --prepared "$PREPARED" \
  --output "$HOME/llm-smoke" --smoke-test
bash finetune.sh train-llm --prepared "$PREPARED" \
  --output "$HOME/llm-run"
bash finetune.sh export-llm --checkpoint "$HOME/llm-run/adapter" \
  --output "$HOME/llm-adapter"
bash finetune.sh export-llm --checkpoint "$HOME/llm-adapter" \
  --output "$HOME/llm-merged" --merge
```

The default is BF16 where supported, unquantized LoRA, rank 16, alpha 32,
microbatch 1, effective batch 16, and gradient checkpointing. QLoRA remains
available with `--set training.quantize=true`. Vision parameters stay frozen;
only language attention/MLP projections receive adapters. Only current-request
assistant tokens are supervised. Existing singleton gold actions remain valid.
Loss masks are passed unchanged through the custom collator; packing is disabled.
Length grouping reduces padding. Benchmark microbatches 1/2/4 with effective
batch 16; changing settings requires a new run, not an old optimizer resume.

Supported families are Qwen3-VL and text-only causal models supported by both
installed frameworks. Unsupported/custom remote architectures and multi-GPU LLM
profiles fail clearly. SEA-LION and Qwen3-4B-Instruct-2507 are the explicit server
validation targets, not locally GPU-certified models.

Adapters retain standard PEFT files plus tokenizer and deployment metadata.
Merged exports are explicit and use GPU memory during merging. Both work with
vLLM: `--checkpoint`/`--llm-checkpoint` means adapter; `--model`/`--llm-model`
means full weights. No automatic merge or Transformers inference fallback occurs.

## Continuous inference

```bash
bash finetune.sh submit --batch-size 4 --limit 32 --allow-download \
  --output "$HOME/continuous4-preview"
```

`--batch-size` now means maximum **active samples**, default 1. A completed sample
immediately releases a slot for the next input; slow samples do not hold a fixed
group open. Each sample has one ToolEnv and at most one pending LLM request.
Ready ASR requests are combined without waiting for a batch to fill. vLLM accepts
generation requests independently, continuously batching their tokens internally.
Both models share the GPU, so ASR/LLM overlap must be measured for contention.

The sample timer starts immediately before ToolEnv creation. ASR, waiting,
generation, and tools all count against its 30 seconds. Each request has a unique
ID and deadline. Ordinary LLM timeout cancels only that request; late results
cannot execute tools. An engine that fails to acknowledge cancellation within a
two-second grace period is reaped with its process group. Engine failure affects
its pending requests; any reload counts toward affected samples' budgets.

Inference settings are independent of training quantization:

```bash
--llm-set inference.precision=bf16
--llm-set inference.gpu_memory_utilization=0.70
--llm-set inference.max_num_batched_tokens=4096
--llm-set inference.max_seq_length=8192
--llm-set inference.max_new_tokens=512
```

Prefix caching and chunked prefill are enabled. The engine's maximum number of
sequences follows `--batch-size`. Legacy `training.max_seq_length` and
`training.generation_max_new_tokens` still supply budgets when no inference
override is given. Context overflow fails without truncation. Prefix caching
only saves prefill; batching does not guarantee lower per-sample latency.

JSON is constrained to installed tools plus `sys.ask_missing_info`. Multiple
independent calls are permitted, executed in order after validating the entire
list. A result-dependent action must wait for another model generation. On
`ok:false`, `done:false`, or clarification, the unexecuted suffix is discarded;
only actual calls/results enter the next prompt. JSON constraints cannot prove
semantic independence or the correctness of IDs. Official ToolEnv validation
remains authoritative. `[]` stops the agent.

## Journals, resume, and timing

- `inflight.json` is durably written before admission creates a ToolEnv.
- `completed.jsonl` atomically groups each completed result and trace in one
  fsynced journal record, in completion order. A torn/corrupt record fails closed.
- `results.jsonl` and `traces.jsonl` are readable secondary logs; resume can
  rebuild them from the completion journal.
- Final preview/submission lines are restored to input order. Resume skips
  finalized IDs, including those beyond an unfinished input prefix.
- An admitted ID lacking a durable completion prevents resume; no sample is
  silently recreated. A stale marker for an already completed ID is recoverable.
- Ctrl-C stops admissions and finalizes active samples with their actual logs.
  Hard crashes retain markers. Old fixed-group resumes/optimizer checkpoints
  are incompatible; legacy model configs and PEFT artifacts are normalized.

Traces preserve round-trip `generation_seconds` and add output tokens, first-token
latency, decode duration, cached tokens when available, finish reason, and stage
waiting. `gpu_memory_used_bytes` is a device-wide observation including ASR and
other processes, not a guaranteed peak allocator measurement. Summaries include
p50/p95 sample latency, repairs, timeouts, throughput, and peak observed memory.
Training benchmarks report allocator peak memory and approximate non-padding
input-token throughput. Seeds and scheduling identity are recorded; concurrency
and hardware changes may change greedy results numerically.

`eval-llm` reports teacher-context next-step generation metrics without executing
tools. Validation loss is saved during training. Next-step exact-match against
singleton gold targets can penalize a valid multi-call grouping; end-to-end tool
log/state evaluation is needed for agent correctness.

## Real GPU verification

Prime the selected ASR and LLM caches with an `--allow-download` preview first.
Then run, separately on the L40S and A30:

```bash
python3.11 gpu_validate.py --prepared "$PREPARED" --data-root "$DATA_ROOT" \
  --output "$HOME/migration-validation"
# Longer experiment; new directory required:
python3.11 gpu_validate.py --prepared "$PREPARED" --data-root "$DATA_ROOT" \
  --output "$HOME/migration-sweep" --sweep --write-locks
```

The validator runs training smoke/resume/reload, adapter and merged prediction,
and generation evaluation for both model families, then an offline agent preview.
The sweep adds microbatches 1/2/4 and concurrency 1/2/4/8 crossed with token budgets
2048/4096/8192. It records commands, failures, timings, and hardware; it stops on a
failed check, including OOM. Logs and the JSON report are the evidence. Offline
environment flags are checked here; also run the final deployment container with
network disabled to establish container-level isolation.

Compare quality, timeouts, memory, and p95 latency before selecting throughput.
No automatic fastest-only configuration is recommended. GPU speed baselines were
not captured on the Windows development host; use the earlier checkout in an
isolated directory for the Transformers NF4/BF16 comparison. The local 46-test
CPU baseline passed before migration; see VERIFICATION.md for current results.
