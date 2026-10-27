# LifeMate portable fine-tuning

Copy **this entire folder** to a Linux x86_64 NVIDIA GPU machine. It is independent
of the original checkout. Python 3.10–3.12 is supported; use 3.11 when possible.
Training requires a driver supporting **CUDA 12.8 or newer**, as required by the
pinned Gipformer/k2 wheels. Setup checks this before installing anything. It does
not change your notebook's global Python packages or install a driver.

The initial models are Gipformer 1.5 (Zipformer RNN-T) and Qwen SEA-LION v4 4B VL.
An offline public/private-test submission runner is included; see [SUBMISSION.md](SUBMISSION.md).
There is no automatic upload or model publication.
Keep the competition data and derived manifests private as required by its README.

## Quick start

Run from the copied `finetuning/` directory. `DATA_ROOT` is the folder containing
`train/`, **not** `train/` itself. Preparation reads only `train.jsonl`, the train
catalog, and WAV files under `train/audio/`. App environment files are never needed.

```bash
export FINETUNE_HOME=/work/lifemate       # writable environments + ASR source cache
export HF_HOME=/work/huggingface          # optional writable model cache
export DATA_ROOT=/data/train+public_test
export PREPARED=/work/prepared

bash finetune.sh setup asr
bash finetune.sh setup llm
bash finetune.sh prepare --data-root "$DATA_ROOT" --output "$PREPARED"

# Always run these in separate, new output directories first.
bash finetune.sh train-asr --prepared "$PREPARED" --output /work/asr-smoke --smoke-test
bash finetune.sh train-llm --prepared "$PREPARED" --output /work/llm-smoke --smoke-test

# Timed pilot runs: warm-up updates are excluded from steady-state timings.
bash finetune.sh train-asr --prepared "$PREPARED" --output /work/asr-benchmark --benchmark
bash finetune.sh train-llm --prepared "$PREPARED" --output /work/llm-benchmark --benchmark

# Full runs (default gpu24 profile).
bash finetune.sh train-asr --prepared "$PREPARED" --output /work/asr-run
bash finetune.sh train-llm --prepared "$PREPARED" --output /work/llm-run
```

Initial downloads and setup need internet. Thereafter use the existing caches or
local model directories. Set `HF_HUB_OFFLINE=1` for cached/offline execution.
Do not use an ephemeral notebook output path for checkpoints you need to retain.
Save/copy notebook outputs through the platform's normal persistent storage.

`setup` intentionally refuses to overwrite an existing environment. If setup was
interrupted, use a fresh `FINETUNE_HOME` and rerun it. Save the printed error if a
CUDA wheel cannot be installed; do not mix k2, PyTorch and CUDA wheel versions.

## Changing a model or training settings

The model configs are JSON-compatible YAML, readable by the standard library;
normal YAML is also accepted in the installed environments. Copy a model config
and edit it, or override fields at the command line:

```bash
# A compatible text-only Qwen model; the backend picks the causal-LM handler.
bash finetune.sh train-llm --prepared "$PREPARED" --output /work/qwen3-run \
  --model Qwen/Qwen3-4B-Instruct-2507 --revision main --profile gpu16 \
  --set training.epochs=2 --set training.learning_rate=0.00005

# Use a previously downloaded/exported compatible checkpoint.
bash finetune.sh train-asr --prepared "$PREPARED" --output /work/asr-other \
  --config configs/models/gipformer.yaml --model /models/compatible-zipformer \
  --set architecture.encoder_dim='"192,256,384,512,384,256"'
```

Precedence is **model config < hardware profile < explicit CLI model/revision/
tokenizer < repeated `--set dotted.key=value` overrides**. Values for `--set` use
JSON syntax, or plain strings. A model-ID override resets the original model's
revision to `main`; use `--revision COMMIT` to pin the replacement. For a local
directory, revision is ignored. Tokenizer/architecture changes are independent
and must remain compatible with the new checkpoint.

Compatibility is explicit:

- `hf_sft`: Hugging Face causal-LM architectures supported by the pinned
  Transformers version, and the `qwen3_vl` handler. Language projection names
  must match the backend's supported attention/MLP targets. Custom remote model
  code is not executed. Qwen3.5 VL and other multimodal families are not promised.
- `zipformer_rnnt`: compatible icefall Zipformer transducer checkpoints with
  `model.pt`, `bpe.model`, and `tokens.txt`. Configure exact architecture dimensions.
  Whisper, CTC-only ASR, a new vocabulary, and different encoder families need
  another backend; changing an ID alone cannot make them compatible.
- ASR loading rejects unexpected weights, missing inference weights and tensor
  shape mismatches. Only missing `simple_am_proj`/`simple_lm_proj` training-loss
  heads may be initialized; their names are printed explicitly.

Each run records the resolved model snapshot and icefall commit, package versions,
training configuration, source fingerprint and split fingerprint in `run.json`.
Default source/checkpoint revisions are pinned in model configs. No model weights
or upstream source are included in this folder.

## Hardware profiles

| `--profile` | GPUs | LLM context | ASR seconds/batch/GPU |
|---|---:|---:|---:|
| `gpu16` | 1 × 16 GB | 4,096 | 40 |
| `gpu24` (default) | 1 × 24 GB | 8,192 | 60 |
| `large` | 1 larger GPU | 8,192 | 120 |
| `dual16` | 2 × 16 GB | 4,096 | 40 |

Profiles are starting configurations, not guarantees of fitting every checkpoint.
For OOM, lower `training.max_seq_length` or `training.max_audio_seconds` in a new
run. LLM microbatch defaults to one; gradient accumulation maintains an effective
batch of 16 across all GPUs. Two GPUs use explicit `torchrun` DDP with one model
replica per GPU; their VRAM is not pooled. ASR sharding pads its batch list so all
ranks perform the same number of updates. This can repeat a small number of batches.

BF16 is selected if supported; otherwise FP16 with gradient scaling is used.
Seeds are fixed and deterministic PyTorch algorithms are requested in warning
mode. k2 kernels and distributed reductions may still differ numerically; run
repeatability checks on the intended inference hardware.

## Data and training details

Preparation creates approximately 90/10 splits with seed 42. Conversations sharing
any normalized current/history utterance are grouped, so the exact validation
count can exceed 10%. The manifests retain relative audio paths and a default
data root. After relocating audio, pass `--data-root /new/root` to ASR commands.
Audio bytes, source records, catalog and split IDs contribute to fingerprints.
Preparation never modifies the source dataset or reads environment state files.

ASR uses every current and historical user audio/transcript pair, normalized with
Unicode NFC, lowercase, punctuation replaced by spaces, and collapsed whitespace.
Diacritics and number spelling are preserved. It retains the original vocabulary,
uses 80-bin filterbanks, pruned RNN-T loss plus auxiliary simple loss, duration
bucketing and training-only SpecAugment. Defaults: 10 epochs, AdamW at 1e-4,
cosine schedule with 5% warm-up and gradient clipping. Validation uses greedy
decoding; `best.pt` and the automatic export are selected by corpus WER.

LLM training uses one complete current-turn trajectory per conversation, with
installed tool schemas, context and transcript history. Prior actions are context,
not new targets. Every current assistant call precedes its tool result. Only
current assistant JSON responses are supervised; history, system, user, tool and
padding tokens are masked. Free-text `final_state` labels are omitted entirely.

The assistant contract is exactly `{"tool_calls":[{"name":"app.tool","arguments":{}}]}`,
at most one call per turn. `{"tool_calls":[]}` terminates a trajectory, including
out-of-scope cases. Clarification is a normal `sys.ask_missing_info` tool call.
This custom JSON contract uses the model's chat template, not its native function
calling markup. Any later agent runner must use the same prompt and contract.

Defaults: text-only 4-bit NF4 QLoRA, frozen vision encoder, language attention/MLP
LoRA rank 16, alpha 32, dropout .05, three epochs, learning rate 1e-4. Full-parameter
LLM training and image training are not implemented. Unquantized **LoRA** is
available with `--set training.quantize=false` if sufficient memory is available.

Overlength examples are **excluded**, with IDs/token counts in `length_report.json`.
There is no silent truncation. Review exclusions before comparing runs; 4K can
exclude substantially more examples than 8K. Chat templates that rewrite earlier
prefixes are rejected rather than produce incorrect loss masks.

**Measured with the initial Qwen tokenizer on this dataset:** 4K excludes
809/1,391 training conversations and 82/154 validation conversations. At 8K,
only one training conversation is excluded and all validation conversations fit.
Prefer an 8K run on a 24 GB GPU when possible. The 16 GB profile is primarily a
memory-conservative starting point; inspect its exclusions before using it for
an accuracy comparison. You can attempt 8K with `--set training.max_seq_length=8192`
on a 16 GB GPU, but memory fit must be established by a smoke test.

## Evaluation and resume

For **audio → live tool execution → submission.jsonl**, use the separate
[`submit` command](SUBMISSION.md). The `eval-llm` command below remains a
teacher-forced model check and is not the submission pipeline.

```bash
bash finetune.sh eval-asr --prepared "$PREPARED" --output /work/asr-eval \
  --checkpoint /work/asr-run/best.pt --baseline

bash finetune.sh eval-llm --prepared "$PREPARED" --output /work/llm-eval \
  --checkpoint /work/llm-run/adapter --limit 30

bash finetune.sh train-asr --prepared "$PREPARED" --output /work/asr-run \
  --resume /work/asr-run/latest.pt

bash finetune.sh train-llm --prepared "$PREPARED" --output /work/llm-run \
  --resume /work/llm-run/checkpoint-50
```

Use the **same config/profile/overrides** when resuming. Resume validates model,
data and training settings and restores optimizer, scheduler, scaler and RNG
states; ASR also restores its exact epoch/batch cursor and per-rank RNG. Keep the
same GPU count. Do not resume a pilot checkpoint as a full run: their schedules
differ. All new run/evaluation/export directories must be empty.

ASR evaluation writes corpus WER/CER and per-clip predictions; `--baseline` also
evaluates pretrained weights on the identical held-out clips. LLM evaluation
writes validation loss and **teacher-forced next-step** JSON/schema/tool-name/exact
argument metrics. It reveals only previous gold tool results, never future ones.
Generation reserves its configured output-token budget; excluded cases are counted.
It does not execute tools or measure final-state/competition score. Its schema
checks cover types, required fields, enums, nested arrays, bounds, date formats and
`x-one-of`; `x-ref` existence requires live app data and is intentionally not checked.

Evaluate the unadapted LLM by omitting `--checkpoint`, with a different output
directory. Compare the same validation records and inspect overlength reports.

## Export and local inference

```bash
bash finetune.sh export-asr --checkpoint /work/asr-run/best.pt --output /work/asr-export
bash finetune.sh transcribe --model /work/asr-export --audio /data/example.wav

bash finetune.sh export-llm --checkpoint /work/llm-run/adapter --output /work/llm-export
bash finetune.sh predict --checkpoint /work/llm-export --input /work/request.json

# Optional: merge adapters into the original unquantized base. Allow ample CPU RAM
# (32 GB+ advisable), disk space, and time. The base must be cached or downloadable.
bash finetune.sh export-llm --checkpoint /work/llm-run/adapter --output /work/llm-merged --merge
bash finetune.sh predict --model /work/llm-merged --input /work/request.json
```

For `predict`, `request.json` is either `{"messages":[{"role":"system","content":...},
{"role":"user","content":...}]}` using the documented contract, or one complete
prepared LLM record. In the latter case only its history/context/request are used
to predict the first current action. It does not execute that action.

Adapter exports need the original base weights; they are not standalone merged
models. Exported ASR directories contain `model.pt`, BPE/token files and exact
architecture configuration. Their decoding requires the pinned icefall checkout.

## Kaggle / Colab cells

Upload the folder and competition data privately or mount them. Replace these
example paths with your actual input locations. Python 3.10–3.12 and a compatible
GPU driver are still required; setup stops clearly on an incompatible runtime.

```python
# First cell: choose writable locations. These environment values persist in
# this notebook process and are inherited by subsequent !bash commands.
import os
os.environ["FINETUNE_HOME"] = "/kaggle/working/lifemate-runtime"  # Colab: /content/lifemate-runtime
os.environ["HF_HOME"] = "/kaggle/working/hf-cache"                # Colab: /content/hf-cache
```

```bash
%cd /kaggle/working/finetuning
!bash finetune.sh setup llm
!bash finetune.sh prepare --data-root /kaggle/input/your-private-data/train+public_test --output /kaggle/working/prepared
!bash finetune.sh train-llm --profile gpu16 --prepared /kaggle/working/prepared --output /kaggle/working/llm-smoke --smoke-test
```

Run ASR with its own `setup asr`, `train-asr` and smoke output. Use `dual16` only
when both T4s are visible and you intend DDP. For Colab use `/content` paths and
your mounted data path. Save checkpoints before a session expires.

## Verification and adding a backend

```bash
python -m unittest discover -s tests -v
python cli.py --help
```

CPU tests require only Python. GPU imports are lazy. Smoke commands load real
weights, train with finite loss, check that weights change, exercise checkpoint
resume, export and reload. Benchmark commands write timings/ETA, not accuracy
claims. Each smoke/benchmark is deliberately a separate disposable run.

Backend interface:

1. `load`: resolve model/tokenizer/revision; reject incompatible architecture.
2. `prepare`: adapt shared manifests to model-specific features/tokens.
3. `train`: apply model config/profile, checkpoints, resume and logging.
4. `evaluate`: report model metrics without leaking future results.
5. `export`: save deployment weights plus tokenizer/config and reload instructions.

Add a module under `ftkit/backends`, a model config, and CLI routing for a new
family. Shared manifest generation and the shell launcher need no model-specific
data changes. The current ASR implementation keeps feature adaptation in
`features`; the LLM implementation exposes `prepare` for tokenization.

Create a portable archive with `python pack.py --output /work/finetuning.zip`.
The packer includes code/configuration/docs only, excluding all dataset manifests,
environments, downloaded models, run outputs and caches.

### Verification status

Local CPU tests and real-data preparation are checked during implementation.
**CUDA training, native k2 wheels, GPU memory fit, distributed execution and
exported-model inference have not been verified on this Windows CPU runtime.**
Run the supplied smoke tests on the target server before committing to full runs.
Successful setup alone is not a training verification.
