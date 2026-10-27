# ZaloAI — LifeMate

Training and inference workspace for the Zalo AI Challenge 2026 LifeMate task:
turn spoken Vietnamese requests into tool calls that leave simulated apps in the
requested state.

The pipeline transcribes the current request and conversation history, then uses
an LLM to select actions, execute them through the supplied `zaic_tools`
environment, and generate a JSON Lines submission.

## Repository layout

```text
.
├── README.md
├── ZaloAI.md                  # Local competition brief
├── finetuning/                # Portable training and inference toolkit
│   ├── README.md              # Setup, training, evaluation, and export guide
│   ├── SUBMISSION.md          # End-to-end submission guide
│   ├── VERIFICATION.md        # Recorded checks and limitations
│   ├── finetune.sh            # Linux launcher with isolated environments
│   ├── cli.py                 # Python command-line entry point
│   ├── configs/               # Model configurations and GPU profiles
│   ├── ftkit/                 # Data, model backends, and agent runner
│   └── tests/                 # CPU tests and tokenizer checks
├── finetuning.zip             # Local portable toolkit archive
└── train+public_test/          # Private competition data package
    ├── README.md              # Organiser-provided data instructions
    ├── VERSION
    ├── train/
    └── public_test/
```

The dataset is restricted to registered teams. Keep it and derived training
manifests private; see the [data package instructions](train+public_test/README.md).
`finetuning.zip` is a packaged copy; use `finetuning/` for the working source.

## Models and capabilities

| Component | Default | Supported workflow |
| --- | --- | --- |
| ASR | Gipformer 1.5, Zipformer RNN-T | Fine-tuning, evaluation, export, transcription |
| LLM | Qwen SEA-LION v4 4B VL | Text-only LoRA/QLoRA, evaluation, export, action generation |
| Alternative ASR | PhoWhisper-medium | Inference only through the `hf_whisper` backend |

The submission runner connects ASR, LLM action generation, and live tool execution.
Standalone `eval-llm` measures teacher-forced next-step predictions; it does not
measure the final-state competition score.

## Quick start

Run training and model inference on **Linux x86_64 with an NVIDIA GPU**. The toolkit
supports Python 3.10–3.12 and recommends 3.11. Gipformer setup requires a driver
supporting CUDA 12.8 or newer. Initial setup and model downloads require internet;
submission inference defaults to cached or local offline loading.

From the repository root on the Linux host:

```bash
export DATA_ROOT="$(pwd)/train+public_test"  # Contains train/ and public_test/
export FINETUNE_HOME=/work/lifemate         # Writable isolated environments/cache
export HF_HOME=/work/huggingface
export PREPARED=/work/prepared
cd finetuning

bash finetune.sh setup asr
bash finetune.sh setup llm
bash finetune.sh prepare --data-root "$DATA_ROOT" --output "$PREPARED"

# Verify each backend on the target GPU before a full run.
bash finetune.sh train-asr --prepared "$PREPARED" --output /work/asr-smoke --smoke-test
bash finetune.sh train-llm --prepared "$PREPARED" --output /work/llm-smoke --smoke-test

# Train in separate output directories; gpu24 is the default profile.
bash finetune.sh train-asr --prepared "$PREPARED" --output /work/asr-run
bash finetune.sh train-llm --prepared "$PREPARED" --output /work/llm-run
```

Preparation uses labelled training records, the training tool catalog, and audio.
It creates grouped train/validation manifests without reading app environment
files. Use new, empty output directories unless explicitly resuming a run.

See the [training guide](finetuning/README.md) for hardware profiles, configuration
overrides, evaluation, resume, model export, and Kaggle/Colab examples.

## Generate a submission

Run from `finetuning/`, using the environments and paths configured above:

```bash
# Optional pretrained preview; permits initial model downloads.
bash finetune.sh submit --data-root "$DATA_ROOT" \
  --output /work/public-preview --limit 5 --allow-download

# Full run with fine-tuned ASR weights and an LLM adapter.
bash finetune.sh submit --data-root "$DATA_ROOT" \
  --asr-checkpoint /work/asr-run/best.pt \
  --llm-checkpoint /work/llm-run/adapter \
  --output /work/public-run

bash finetune.sh validate-submission --data-root "$DATA_ROOT" \
  --input /work/public-run/submission.jsonl
```

Limited runs produce `preview.jsonl`. Full runs produce `submission.jsonl`, with
one `{id, tool_calls, state_hash}` object per sample. Review `summary.json`,
`traces.jsonl`, and worker logs before uploading. Validation checks coverage,
schemas, and hash format; it does not independently verify the final state.

The [submission guide](finetuning/SUBMISSION.md) covers pretrained or exported
models, PhoWhisper, offline loading, interrupted runs, and private-test inference.
Docker packaging follows the organisers' separate instructions.

## Data access and execution rules

The local [competition brief](ZaloAI.md) and data package describe these constraints:

- Access app data only through the supplied tools. Do not directly read, copy, or
  parse files under `<split>_environment/`.
- Create exactly one `ToolEnv` per sample and include every executed call in the
  submission, including failed calls. History actions are already applied.
- Use only the sample's installed app tools plus `sys.ask_missing_info`. Ask for
  missing arguments through that tool; use an empty call list for out-of-scope
  requests.
- Compute the final hash with `env.get_state_hash()` using the matching split's
  environment library.
- The brief specifies fewer than 6B total model parameters, fewer than 5B LLM
  parameters, at most 16,384 context tokens, and a 30-second per-sample limit.
  Inference must run offline with reproducible decoding.
- Do not create public/private-test labels to train or tune the solution.

These are a summary of the checked-in material; consult the organisers' latest
instructions for authoritative competition requirements.

## Development and verification

CPU tests require only Python and can run without GPU dependencies:

```bash
cd finetuning
python -m unittest discover -s tests -v
python cli.py --help
```

See [VERIFICATION.md](finetuning/VERIFICATION.md) for recorded verification.
CUDA training, GPU memory fit, distributed execution, and exported-model inference
still require validation on the target GPU host.

To rebuild a portable archive containing code, configurations, and documentation:

```bash
# Run from finetuning/.
python pack.py --output /work/finetuning.zip
```

The packer excludes competition data, prepared manifests, model weights,
environments, caches, and run outputs.
