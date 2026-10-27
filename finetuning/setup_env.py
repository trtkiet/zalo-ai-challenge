#!/usr/bin/env python3
"""Install isolated Linux/CUDA training environments; never change notebook packages."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import venv

from ftkit.env import load_env_file

ROOT = Path(__file__).resolve().parent


def run(*args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=["asr", "llm"])
    args = parser.parse_args()
    try:
        load_env_file()
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        parser.error("Training setup requires Linux x86_64 with NVIDIA CUDA. On Windows use WSL2 or copy the folder to a GPU server.")
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        parser.error("Use Python 3.10–3.12 (3.11 recommended) for the pinned CUDA wheels.")
    try:
        info = subprocess.check_output(["nvidia-smi"], text=True)
    except (OSError, subprocess.CalledProcessError):
        parser.error("nvidia-smi failed. Enable a GPU runtime/install the NVIDIA driver first.")
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", info)
    if not match or tuple(map(int, match.groups())) < (12, 8):
        parser.error("The pinned stack needs driver support for CUDA 12.8+. Use a compatible server/runtime; setup will not replace your driver.")
    home = Path(os.environ.get("FINETUNE_HOME", ROOT)).expanduser().resolve()
    destination = home / ".venvs" / args.environment
    py = destination / "bin" / "python"
    if destination.exists():
        parser.error(f"Environment already exists: {destination}. Reuse it, or set FINETUNE_HOME to a new location to rebuild.")
    venv.EnvBuilder(with_pip=True).create(destination)
    run(py, "-m", "pip", "install", "pip==25.2")
    packages = ["torch==2.8.0"]
    if args.environment == "asr":
        packages.append("torchaudio==2.8.0")
    run(py, "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cu128", *packages)
    run(py, "-m", "pip", "install", "-r", ROOT / "requirements" / f"{args.environment}.txt")
    if args.environment == "asr":
        config = json.loads((ROOT / "configs/models/gipformer.yaml").read_text())
        checkout = home / ".cache" / "icefall"
        revision = config["icefall_revision"]
        if not checkout.exists():
            checkout.mkdir(parents=True)
            run("git", "-C", checkout, "init")
            run("git", "-C", checkout, "remote", "add", "origin", "https://github.com/k2-fsa/icefall.git")
            run("git", "-C", checkout, "fetch", "--depth", "1", "origin", revision)
            run("git", "-C", checkout, "checkout", "--detach", "FETCH_HEAD")
        actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise RuntimeError(f"Existing icefall checkout is {actual}, expected {revision}; use a new FINETUNE_HOME or --icefall-dir.")
        run(py, "-c", "import torch,k2,kaldifeat,lhotse; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name())")
    else:
        run(py, "-c", "import torch,transformers,peft,bitsandbytes; assert torch.cuda.is_available(); print(torch.__version__,transformers.__version__)")
    freeze = subprocess.check_output([str(py), "-m", "pip", "freeze"], text=True)
    (destination / "installed-packages.txt").write_text(freeze)
    print(f"Ready: {py}. Run a smoke test before a full training run.")


if __name__ == "__main__":
    main()
