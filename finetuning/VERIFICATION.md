# Migration verification (2026-10-28)

- Before migration: 46 dependency-free CPU tests passed on Windows/Python 3.12.
- Current CPU regression suite: **60 tests passed**. Covers continuous admission/refill, per-request
  cancellation, out-of-order IPC and output restoration, durable journal resume,
  no ToolEnv recreation, independent multi-call execution and discarded suffixes,
  schema extension translation, loss-mask preservation, mocked Unsloth loaders,
  and streamed vLLM adapter behavior. See the latest command result for test count.
- The async fake-engine test requires an OS event loop. The Windows sandbox blocked
  its local socket pair; the complete suite was therefore run outside the sandbox.
- Python compilation, Bash syntax, and `git diff --check` passed. The portable
  code-only ZIP was extracted into a fresh directory and its 60 tests passed;
  data, caches, environments, and model weights are excluded.
- `gpu_validate.py` runs real model training/resume/export/reload, both model families,
  generation evaluation, offline previews, and optional training/inference sweeps.
- Follow-up review covers the real worker protocol with a fake async engine,
  cancellation acknowledgements, custom tokenizer parity, and disabled nested
  language-model KV caches with or without training gradient checkpointing.
- Real Unsloth/vLLM installation, CUDA performance, VRAM fit, A30/L40S repeatability,
  quality, and network-disabled container execution are NOT verified on this host.
- Requirements are bootstrap constraints. No Linux/GPU-certified dependency locks
  or speedup claims were fabricated. Generate locks only after server validation.
- The user's reported pre-migration EC2 run (4 samples / 70.825 seconds) is external
  evidence, not a locally reproduced or controlled performance baseline.

The prior verification history follows for context; it does not certify the new
GPU backends. See MIGRATION.md for updated behavior and server commands.

# Verification record

Checked locally on Windows, Python 3.12, without PyTorch/CUDA training libraries.

## Historical checks before the LLM migration

- `.ENV` loading: 46 CPU tests total. Added checks for repository/toolkit file
  selection, explicit file overrides, quoted paths and variable expansion,
  exported-variable and CLI precedence, literal command text, atomic failure on
  invalid settings, CLI path defaults, required arguments without defaults and
  rejection of empty CLI paths that would otherwise resolve to the working directory.

- Independent-sample batching: 39 CPU tests total, including shrinking model
  batches, separate live tool histories, result ordering, repair/error isolation,
  deadline enforcement before actions, batch timeout/interruption preservation,
  left-padding/attention masks/output slicing, context-overflow isolation,
  Gipformer clip ordering with malformed-input isolation, Whisper chunk/audio
  ordering and attention masks, grouped submission
  generation/resume, changed-batch-size rejection, and batch worker IPC.
  These checks use synthetic environments and neural test doubles. Actual
  batched CUDA execution, throughput, VRAM fit and repeatability are unverified.

- 16 dependency-free unit tests: conversation/utterance split isolation,
  deterministic splits, output collision protection, safe audio paths, preparation
  without environment directories, chronological tool results, current-assistant
  masks, empty/clarification/nested actions, invalid JSON structures, overlength
  exclusion, incompatible template rejection, model/config overrides, local export
  configuration, changed-dataset resume rejection, ASR shape guards, duration
  batching and corpus error-rate calculation.
- Bash syntax check and Python source compilation/AST parsing.
- 12 additional submission tests (28 tests total): live-result feedback, history
  transcription, exact executed-call logs, clarification, invalid-response repair,
  installed-tool restrictions, limits, interruption/timeout preservation, coverage,
  resume without recreating sessions, preview naming/coverage, real worker IPC
  and process timeout/termination.
  Neural models and app states are synthetic test doubles for these CPU tests.
- Preparation against the real labeled train set: 1,391 training conversations /
  4,526 clips / 6.13975 hours; 154 validation conversations / 510 clips / 0.68950 hours.
- Actual Qwen SEA-LION tokenizer at revision `d0734aa`: all 1,545 trajectories
  passed prefix-stability and current-assistant loss-mask construction. Every
  reference tool call passed the model-level schema checker.
- Pinned lhotse, kaldialign, PEFT, Accelerate and bitsandbytes versions exist on PyPI.
- The three organizer-supplied example submission rows pass the new runner's
  structural/schema checks using the actual public-test catalog/runtime. No app
  environment was instantiated or replayed for that validation.

## Real tokenizer lengths

| Partition | Mean tokens | Maximum | Excluded at 4,096 | Excluded at 8,192 |
|---|---:|---:|---:|---:|
| Train, 1,391 records | 4,524.6 | 8,356 | 809 | 1 |
| Validation, 154 records | 4,304.5 | 7,419 | 82 | 0 |

This uses full trajectories and the supplied schemas. Generation checks reserve
additional space for the output budget and report their own exclusion counts.

## Not yet verified on a GPU

- Installation/import of the native Linux k2/kaldifeat wheel combination.
- Initial Gipformer tensor compatibility and complete forward/backward execution.
- QLoRA updates, actual VRAM usage, mixed-precision stability and throughput.
- One-/two-GPU training, optimizer/RNG resume and export/reload on CUDA.
- Improvements in ASR WER, tool calling or competition score.
- Public/private-test neural inference, Whisper inference, combined GPU memory,
  and the 30-second end-to-end agent latency on competition hardware.

Run both supplied `--smoke-test` commands on the target host before full training.
They are implemented checks, not claims that those checks already passed locally.
