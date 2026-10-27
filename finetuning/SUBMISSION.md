# Run chosen ASR + LLM models and produce a submission

Run these commands from the copied `finetuning/` folder on a Linux CUDA host.
The required upload format is **UTF-8 JSON Lines**, `submission.jsonl`, with one
`{id, tool_calls, state_hash}` object per input sample, not a large JSON array.
Fine-tuning manifest preparation is not needed for inference.

The launcher automatically reads `.ENV` from `finetuning/` or the repository
root. With `DATA_ROOT`, `RESULT`, `FINETUNE_HOME` and `HF_HOME` configured there,
you can omit the corresponding shell exports and run:

```bash
bash finetune.sh submit --batch-size 4 --allow-download
bash finetune.sh validate-submission
```

Output defaults to `RESULT`, and validation input to `RESULT/submission.jsonl`.
For a preview or another model run, pass a new `--output` directory. CLI arguments
override exported shell variables, which override `.ENV` defaults. See the
[local settings guide](README.md#local-settings-in-env) for expansion and file selection.

## Environments and a small preview

Use your existing ASR and LLM environments. If not installed:

```bash
export FINETUNE_HOME=/work/lifemate
export HF_HOME=/work/huggingface
bash finetune.sh setup asr
bash finetune.sh setup llm
export DATA_ROOT=/data/train+public_test  # directory containing public_test/

bash finetune.sh submit --data-root "$DATA_ROOT" \
  --output /work/public-preview --limit 5 --allow-download
```

This uses default Gipformer 1.5 and Qwen SEA-LION models. `--allow-download`
permits model download during startup; it never uploads data or uses external
model APIs. Otherwise loading defaults to cached/local **offline** mode. Use
existing caches/local exports for the final offline deployment.

Limited runs write `preview.jsonl`, never `submission.jsonl`. Review
`summary.json`, `traces.jsonl` and worker logs. Preview and full runs must use
separate output directories.

## Batch independent samples on one GPU

`--batch-size` controls how many independent samples share neural inference.
The default is `1`, preserving individual requests. Start with `4` on a GPU
with available memory, then compare throughput, timeout counts and peak VRAM:

```bash
bash finetune.sh submit --data-root "$DATA_ROOT" \
  --batch-size 4 --limit 12 --allow-download \
  --output "$HOME/lifemate/batch4-preview"

# Full submission: omit --limit and use a new directory.
bash finetune.sh submit --data-root "$DATA_ROOT" \
  --batch-size 4 --allow-download \
  --output "$HOME/lifemate/batch4-public"
```

Model/config/checkpoint options work as above. Existing environments can be
reused after copying the updated toolkit; no new dependencies are required.

The runner creates exactly one `ToolEnv` per sample, batches ASR clips and the
next LLM action across samples, and executes each action only in its own
environment. LLM prompts are left padded with attention masks; context overflow
fails that sample without truncating its prompt. Gipformer audio clips and
Whisper/PhoWhisper chunks are batched, retaining history/current-audio order.
Only one copy of each model is loaded; batching does not launch model replicas.

Batches are fixed groups: completed samples leave the active group, and the
next group starts when the current group finishes. Each sample's 30-second
deadline includes environment creation, ASR, waiting in its active group, LLM
generation and tool execution. The earliest deadline bounds a shared worker
request. Increasing batch size can improve throughput while increasing latency
and KV-cache memory. OOMs are reported rather than silently retried with another
batch size. Batch size `4` is a starting experiment, not a memory-fit guarantee.

`summary.json` includes `batch_size`, `processed_this_run`, `inference_seconds`
and `samples_per_second`. Inference timing excludes initial model loading and
the up-front input fingerprinting. Compare the same sample limit/settings and
inspect timeout/error counts alongside throughput. Traces also include ASR and
per-action generation timings, including batch waiting time.

Batch size is part of resume identity: repeat the same size with `--resume`.
A clean Ctrl-C during inference finalizes the active group's actual calls and
hashes before stopping. A hard crash retains an unfinished-group marker and
resume refuses to recreate those sessions. Batched GPU execution and numerical
repeatability still need verification on the target host; batching can change
floating-point results even with greedy decoding.

## Choose models and run the full public test

### Fine-tuned Gipformer + SEA-LION adapter

```bash
bash finetune.sh submit \
  --data-root "$DATA_ROOT" \
  --asr-checkpoint /work/asr-run/best.pt \
  --llm-checkpoint /work/llm-run/adapter \
  --output /work/public-run
```

Upload **`/work/public-run/submission.jsonl`** after reviewing its summary.

### Exported ASR + merged LLM

```bash
bash finetune.sh submit --data-root "$DATA_ROOT" \
  --asr-model /work/asr-export --llm-model /work/llm-merged \
  --output /work/public-merged
```

Use `--llm-checkpoint` for a PEFT **adapter**, and `--llm-model` for a complete
pretrained/merged model. Adapters need the original base model cached or supplied
with `--llm-model`. Export metadata is reused unless an explicit config is given.

### Different pretrained checkpoints

```bash
bash finetune.sh submit --data-root "$DATA_ROOT" \
  --asr-model g-group-ai-lab/gipformer1.5-68M-rnnt --asr-revision ccdda59 \
  --llm-model Qwen/Qwen3-4B-Instruct-2507 --llm-revision main \
  --allow-download --output /work/public-qwen3
```

Pin immutable commit revisions for reproducible comparisons. Model-ID overrides
reset the old revision to `main` unless you specify a replacement revision.
`--asr-config` / `--llm-config` select model configs; `--asr-set` and `--llm-set`
apply dotted JSON-style overrides, for example:

```bash
--llm-set training.max_seq_length=16384
--llm-set training.generation_max_new_tokens=384
```

Context is capped at 16,384 tokens. The loaded LLM must have fewer than 5B
parameters, with fewer than 6B total across both models; quantization does not
reduce this count. Both models remain resident on one GPU. DDP profiles are
rejected; select `gpu16`, `gpu24` (default), or `large`.

## PhoWhisper / Whisper comparison

An **inference-only** `hf_whisper` backend runs in a separate worker using the
existing **LLM environment**, without Gipformer's k2 dependencies:

```bash
bash finetune.sh submit --data-root "$DATA_ROOT" \
  --asr-config configs/models/phowhisper_medium.yaml \
  --asr-model vinai/PhoWhisper-medium \
  --allow-download --output /work/public-phowhisper
```

Compatible `vinai/PhoWhisper-small`, `vinai/PhoWhisper-large`, and
`openai/whisper-large-v3-turbo` IDs or local model folders can be substituted.
Decoding is greedy in Vietnamese; audio longer than 30 seconds is processed in
consecutive chunks. This does **not** add Whisper training, Qwen-ASR, Parakeet or
ChunkFormer support. Those require additional inference adapters.

## Agent behavior and outputs

For each sample the runner:

1. Creates exactly one supplied `ToolEnv` for the selected split.
2. Transcribes historical user audio and the current request.
3. Builds the same JSON action prompt used for fine-tuning, with installed schemas,
   user context and history. Previous history actions are not replayed.
4. Generates one next action, validates it with `env.validate`, executes it with
   `env.call`, and returns the **actual** tool result to the LLM.
5. Stops on an empty list. Clarification is a normal `sys.ask_missing_info` call.
   Invalid JSON/schema responses receive at most two repair opportunities before
   termination; an invalid response is not executed.
6. Uses `env.get_tool_calls()` and `env.get_state_hash()` for the final submission.
   Runner code never reads/copies/parses app state files.

Defaults: **30 seconds per sample**, 12 tool calls, 20 LLM turns, two repairs.
`--max-seconds` can lower but not exceed 30. Timing includes environment creation,
all current/history ASR, generation and tool work. Initial model startup/kernel
warm-up happens once before sample processing. A parent-side timeout kills an
overdue worker; native GPU cancellation is not assumed. Simulator calls finish
before control returns and elapsed time is reported. Worker reloads following
timeout count toward the next affected sample's budget.

These guards do not establish compliance with the A30 runtime limit: **GPU
inference, accuracy and end-to-end latency remain unverified locally**. Long
history or many model rounds can time out.

Files in a run directory:

- `submission.jsonl`: created only after complete full-run coverage and format,
  installed-tool and official schema checks. No debug fields are included.
- `results.jsonl`: durable finalized sample rows, used for resume.
- `traces.jsonl`: transcripts, model responses, actual tool results, status and
  elapsed time. Do not upload this as the submission.
- `summary.json`: coverage/format results and status counts. A structurally valid
  file can still score poorly; inspect timeout/error/limit/invalid-response cases.
- `loaded_models.json`, resolved configs and worker logs: model revisions,
  full parameter counts and diagnostics.

Errors/timeouts retain any calls already executed and their actual final hash.
They never invent successful calls or reset the environment. Public-test output
is for submission, not for constructing training labels or tuning on external labels.

## Validate and resume

```bash
bash finetune.sh validate-submission --data-root "$DATA_ROOT" \
  --input /work/public-run/submission.jsonl

# Repeat the original command/settings and add --resume.
bash finetune.sh submit --data-root "$DATA_ROOT" \
  --asr-checkpoint /work/asr-run/best.pt \
  --llm-checkpoint /work/llm-run/adapter \
  --output /work/public-run --resume
```

Validation checks coverage, order, argument schemas and SHA256 format. It does
**not** replay tools or independently verify hashes. Completed rows are never
rerun on resume. Inputs, audio, catalog, runtime code, local model artifacts and
settings are fingerprinted. Changed runs are rejected.

A hard crash can leave an unfinished-sample marker; resume then refuses to
recreate that environment. Clean Ctrl-C during inference finalizes the current
sample with its actual calls/hash before stopping, allowing later resume to skip it.

Use `--split private_test` with the matching data root for private test. Its
supplied `zaic_tools` is imported in a fresh parent process.

On Kaggle/Colab use the README's path/environment setup, then the same
`!bash finetune.sh submit ...` command. `--asr-python` and `--llm-python` override
the two worker interpreters for custom installations. The orchestrator itself
needs only Python's standard library.
