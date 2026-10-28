"""Run real Linux/CUDA migration checks; never substitutes mocks for GPU evidence."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from ftkit.common import fresh_directory, read_jsonl, write_json
from ftkit.env import load_env_file
from setup_env import llm_lock_path

ROOT = Path(__file__).resolve().parent


def main():
    load_env_file()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT"), required=not bool(os.environ.get("DATA_ROOT")))
    parser.add_argument("--output", required=True, help="New directory for actual server evidence")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--sweep", action="store_true", help="Also benchmark microbatches 1/2/4 and concurrency 1/2/4/8 x token budgets 2048/4096/8192")
    parser.add_argument("--write-locks", action="store_true", help="Copy resolved freezes into requirements only after every requested check passes")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run on the Linux GPU server with both isolated LLM runtimes installed")
    if args.limit < 1:
        parser.error("Use a positive preview limit")
    out = fresh_directory(args.output)
    environment = dict(os.environ)
    home = Path(environment.get("FINETUNE_HOME", ROOT)).expanduser().resolve()
    report = {"status": "running", "hardware": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip(),
        "checks": [], "notes": ["Results apply only to this GPU and package environment.",
                                  "Preview validity is not held-out agent correctness or leaderboard accuracy."]}

    def execute(name, command, offline=False):
        env = dict(environment)
        if offline:
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1")
        started = time.monotonic()
        with (out / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(["bash", str(ROOT / "finetune.sh"), *command], env=env,
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        report["checks"].append({"name": name, "command": command, "offline": offline,
                                 "exit_code": result.returncode, "seconds": time.monotonic() - started})
        write_json(out / "report.json", report)
        if result.returncode:
            raise RuntimeError(f"Check {name} failed; inspect {out / (name + '.log')}")

    try:
        sample = read_jsonl(Path(args.prepared) / "llm_validation.jsonl")[0]
        request = out / "request.json"
        write_json(request, sample)
        for name, model_args in (("sealion", []), ("qwen3", ["--model", "Qwen/Qwen3-4B-Instruct-2507"])):
            trained = out / f"{name}-smoke"
            execute(f"{name}-train", ["train-llm", "--prepared", args.prepared, "--output", str(trained), "--smoke-test", *model_args])
            adapter, merged = out / f"{name}-adapter", out / f"{name}-merged"
            execute(f"{name}-export", ["export-llm", "--checkpoint", str(trained / "adapter"), "--output", str(adapter)])
            execute(f"{name}-merge", ["export-llm", "--checkpoint", str(adapter), "--output", str(merged), "--merge"])
            execute(f"{name}-adapter-predict", ["predict", "--checkpoint", str(adapter), "--input", str(request)], offline=True)
            execute(f"{name}-merged-predict", ["predict", "--model", str(merged), "--input", str(request)], offline=True)
            for kind in ("adapter", "merged"):
                text = (out / f"{name}-{kind}-predict.log").read_text()
                predictions = []
                for index, char in enumerate(text):
                    if char != "{":
                        continue
                    try:
                        value, _ = json.JSONDecoder().raw_decode(text[index:])
                        if isinstance(value, dict) and set(value) == {"tool_calls"}:
                            predictions.append(value)
                    except json.JSONDecodeError:
                        pass
                if not predictions or not isinstance(predictions[-1]["tool_calls"], list):
                    raise RuntimeError("Prediction did not preserve the JSON action contract")
            execute(f"{name}-eval", ["eval-llm", "--prepared", args.prepared, "--checkpoint", str(adapter),
                                      "--output", str(out / f"{name}-eval"), "--limit", "8"], offline=True)
        execute("offline-agent", ["submit", "--data-root", args.data_root, "--limit", str(args.limit),
                                  "--batch-size", "2", "--output", str(out / "offline-agent")], offline=True)
        agent_summary = json.loads((out / "offline-agent" / "summary.json").read_text())
        report["offline_agent_statuses"] = agent_summary["statuses"]
        if agent_summary["statuses"].get("error", 0) or agent_summary["statuses"].get("timeout", 0):
            raise RuntimeError("Offline preview has engine errors/timeouts; inspect traces before certifying this runtime")
        if args.sweep:
            for micro in (1, 2, 4):
                execute(f"train-micro{micro}", ["train-llm", "--prepared", args.prepared, "--benchmark",
                                               "--set", f"training.microbatch_size={micro}",
                                               "--output", str(out / f"train-micro{micro}")])
            sweep = []
            for concurrency in (1, 2, 4, 8):
                for budget in (2048, 4096, 8192):
                    name = f"infer-c{concurrency}-t{budget}"
                    execute(name, ["submit", "--data-root", args.data_root, "--limit", str(args.limit),
                                   "--batch-size", str(concurrency), "--llm-set", f"inference.max_num_batched_tokens={budget}",
                                   "--output", str(out / name)], offline=True)
                    sweep.append({"concurrency": concurrency, "token_budget": budget,
                                  **json.loads((out / name / "summary.json").read_text())})
            write_json(out / "sweep.json", sweep)
            report["notes"].append("Choose a configuration only after matching held-out correctness as well as latency/timeouts; no automatic speed-only recommendation.")
        if args.write_locks:
            for runtime in ("llm-train", "llm-infer"):
                py = home / ".venvs" / runtime / "bin/python"
                subprocess.run([str(py), "-m", "pip", "check"], check=True)
                freeze = subprocess.check_output([str(py), "-m", "pip", "freeze"], text=True)
                version = json.loads(subprocess.check_output(
                    [str(py), "-c", "import json,sys; print(json.dumps(list(sys.version_info[:2])))"], text=True))
                target = llm_lock_path(ROOT, runtime, version)
                if target.exists():
                    raise ValueError(f"Refusing to overwrite existing lock {target}")
                cuda = subprocess.check_output([str(py), "-c", "import torch; print(torch.version.cuda)"], text=True).strip()
                index = f"--extra-index-url https://download.pytorch.org/whl/cu{cuda.replace('.', '')}\n" if cuda != "None" else ""
                target.write_text(f"# Python {version[0]}.{version[1]}; validated by {out / 'report.json'} on {report['hardware']}\n" + index + freeze)
                import hashlib
                marker = home / ".venvs" / runtime / "setup.json"
                state = json.loads(marker.read_text())
                state.update(requirements_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                             source=target.name, python_version=version,
                             gpu_model_smoke_verified=True, evidence=str(out / "report.json"))
                write_json(marker, state)
        report["status"] = "passed_on_this_gpu"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(out / "report.json", report)


if __name__ == "__main__":
    main()
