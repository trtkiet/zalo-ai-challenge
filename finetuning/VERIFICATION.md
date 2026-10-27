# Verification record

Checked locally on Windows, Python 3.12, without PyTorch/CUDA training libraries.

## Passed

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
