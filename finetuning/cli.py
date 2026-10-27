#!/usr/bin/env python3
"""Portable ASR / LLM fine-tuning command line. Preparation needs only Python."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from ftkit.common import resolve_config


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("help")
    prep = sub.add_parser("prepare", help="Build isolated train/validation manifests without reading app states")
    prep.add_argument("--data-root", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--validation-fraction", type=float, default=0.1)
    for command in ("train-asr", "eval-asr", "export-asr", "transcribe", "train-llm", "eval-llm", "export-llm", "predict"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--config", help="Model YAML/JSON config; defaults to the initial model for this task")
        cmd.add_argument("--model", help="Compatible HF checkpoint ID or local directory; resets default revision to main")
        cmd.add_argument("--revision", help="HF revision; pin a commit for reproducibility")
        cmd.add_argument("--tokenizer", help="Optional tokenizer ID/directory override for hf_sft")
        cmd.add_argument("--profile", default="gpu24", help="gpu16, gpu24, large, dual16, or a profile file")
        cmd.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Repeatable dotted config override, e.g. training.epochs=2")
        cmd.add_argument("--prepared", help="Prepared manifest directory")
        cmd.add_argument("--output", help="New output directory, or the same run directory when resuming")
        cmd.add_argument("--data-root", help="Override relocated audio root (folder containing train/)")
        cmd.add_argument("--resume", help="Training checkpoint; restores training and RNG state")
        cmd.add_argument("--checkpoint", help="Checkpoint/adapter for evaluation, inference or export; omit for pretrained evaluation")
        cmd.add_argument("--smoke-test", action="store_true", help="Short training run plus checkpoint resume and export/reload checks")
        cmd.add_argument("--benchmark", action="store_true", help="Short timed training run; writes throughput and ETA")
        cmd.add_argument("--limit", type=int, default=0, help="Limit evaluation conversations/clips; zero means all")
        cmd.add_argument("--icefall-dir", help="ASR: pinned icefall checkout; defaults to FINETUNE_HOME/.cache/icefall")
        cmd.add_argument("--baseline", action="store_true", help="ASR evaluation: also evaluate the original pretrained model")
        cmd.add_argument("--merge", action="store_true", help="LLM export: merge adapter into unquantized base weights")
        cmd.add_argument("--audio", nargs="+", help="ASR transcribe: WAV files")
        cmd.add_argument("--input", help="LLM predict: JSON with messages array, or a prepared-format record")
    return root


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command == "help":
        p.print_help()
        return
    if args.command == "prepare":
        from ftkit.data import prepare
        prepare(args.data_root, args.output, args.validation_fraction, args.seed)
        return
    asr = args.command.endswith("asr") or args.command == "transcribe"
    kind = "asr" if asr else "llm"
    config = resolve_config(args, kind)
    training = args.command.startswith("train-")
    if training or args.command.startswith("eval-"):
        if not args.prepared or not args.output:
            p.error("--prepared and --output are required")
    if args.command.startswith("export-") and (not args.checkpoint or not args.output):
        p.error("Export requires --checkpoint and --output")
    if args.smoke_test and args.benchmark:
        p.error("Choose --smoke-test or --benchmark, not both")
    devices = int(config.get("devices", 1))
    if training and devices > 1 and "LOCAL_RANK" not in os.environ:
        subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={devices}",
                        str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])], check=True)
        return
    if asr:
        from ftkit.backends.asr import run
    else:
        from ftkit.backends.llm import run
    run(args, config)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
