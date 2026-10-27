"""Runtime helpers shared by the two GPU backends."""
import json
import math
import os
from pathlib import Path

from .common import digest, fresh_directory, runtime_manifest, write_json


def distributed(config, training):
    import torch
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if training and world != int(config.get("devices", 1)):
        raise ValueError("WORLD_SIZE differs from the selected hardware profile")
    if world > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    return int(os.environ.get("RANK", "0")), world, local_rank


def barrier():
    import torch
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def validate_existing_output(args, config):
    path = Path(args.output).expanduser().resolve()
    if path.exists() and any(path.iterdir()) and not args.resume:
        raise ValueError(f"Output already contains files: {path}; choose a new output or explicitly --resume.")
    if args.resume and not Path(args.resume).exists():
        raise FileNotFoundError(args.resume)
    manifest_path = path / "run.json"
    if args.resume and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        old = existing["config"]
        for key in ("model", "revision", "backend", "architecture", "training"):
            if old.get(key) != config.get(key):
                raise ValueError(f"Resume config mismatch for {key}; use the original configuration.")
        current = json.loads((Path(args.prepared) / "metadata.json").read_text(encoding="utf-8"))
        if existing["dataset"]["fingerprint"] != current["fingerprint"]:
            raise ValueError("Resume dataset/split fingerprint changed")
    return path


def begin_run(args, config, rank):
    out = Path(args.output).expanduser().resolve()
    if rank == 0:
        if not args.resume:
            fresh_directory(out)
        else:
            out.mkdir(parents=True, exist_ok=True)
        if not (out / "run.json").exists():
            write_json(out / "run.json", runtime_manifest(config, args.prepared))
    barrier()
    return out


def checkpoint_fingerprint(config, metadata):
    return digest({"model": config["model"], "revision": config["revision"], "architecture": config["architecture"],
                   "training": config["training"], "dataset": metadata["fingerprint"]})


def schedule(total, warmup_ratio):
    warmup = max(1, round(total * warmup_ratio))
    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return factor


def save_torch(path, value):
    import torch
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)
