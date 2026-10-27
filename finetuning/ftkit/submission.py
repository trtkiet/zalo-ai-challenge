"""Offline audio/tool agent and official-format submission writer.

The orchestrator uses stdlib only. Model processes use their existing isolated
environments. Only the supplied ToolEnv opens app state; no state file is parsed here.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import types

from .common import ROOT, compact, digest, fresh_directory, read_jsonl, resolve_config, write_json
from .data import action, trajectory


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def artifact_signature(source):
    if not source:
        return None
    path = Path(source).expanduser()
    if path.is_file():
        return {"file_sha256": file_hash(path)}
    if path.is_dir():
        selected = sorted(p for p in path.iterdir() if p.is_file() and
                          (p.suffix in (".json", ".safetensors", ".model", ".txt", ".jinja") or
                           p.name == "model.pt" or p.name.startswith("pytorch_model")))
        return {p.name: file_hash(p) for p in selected}
    return {"hub_id": source}


def audio_path(root, split, relative):
    relative = Path(relative)
    if relative.is_absolute():
        raise ValueError("Audio paths must be relative to the split")
    path = (Path(root) / split / relative).resolve()
    if not path.is_relative_to((Path(root) / split / "audio").resolve()):
        raise ValueError(f"Audio must remain inside {split}/audio: {relative}")
    return path


def inputs(root, split):
    directory = Path(root).expanduser().resolve() / split
    rows = read_jsonl(directory / f"{split}.jsonl")
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Empty split or duplicate input IDs")
    catalog = json.loads((directory / f"{split}_tools_catalog.json").read_text(encoding="utf-8"))
    tools = {t["name"]: t for app in catalog["apps"] for t in app["tools"]}
    return rows, catalog, tools


def installed_tools(item, tools):
    names = {n for app in item["context"]["installed_apps"].values() for n in app["tools"]}
    names.add("sys.ask_missing_info")
    if names - tools.keys():
        raise ValueError(f"Input references tools absent from catalog: {sorted(names - tools.keys())}")
    return [tools[name] for name in sorted(names)]


def runtime(root, split):
    """Import the exact supplied split package without importing another split's runtime."""
    directory = (Path(root) / split / "zaic_tools").resolve()
    initial = directory / "__init__.py"
    if not initial.is_file():
        raise FileNotFoundError(f"Missing supplied execution environment: {initial}")
    existing = sys.modules.get("zaic_tools")
    if existing is not None:
        if Path(existing.__file__).resolve() != initial:
            raise ValueError("A different split's zaic_tools is already imported; run this split in a fresh process")
        return existing
    spec = importlib.util.spec_from_file_location("zaic_tools", initial, submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["zaic_tools"] = module
    spec.loader.exec_module(module)
    return module


def parse_response(text):
    """Accept the trained JSON contract, optionally surrounded by a Markdown fence."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() != "```":
            raise ValueError("Unclosed JSON fence")
        text = "\n".join(lines[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"tool_calls"} or not isinstance(value["tool_calls"], list):
        raise ValueError('Expected {"tool_calls":[...]}')
    if len(value["tool_calls"]) > 1:
        raise ValueError("Return at most one tool call so later actions can use actual results")
    for call in value["tool_calls"]:
        if not isinstance(call, dict) or set(call) != {"name", "arguments"} or not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
            raise ValueError("Tool calls require string name and object arguments")
    return value["tool_calls"]


class Worker:
    def __init__(self, python, kind, config_path, log_path, checkpoint=None, icefall_dir=None, allow_download=False):
        self.command = [str(python), "-m", "ftkit.inference_worker", "--kind", kind, "--config", str(config_path)]
        if checkpoint:
            self.command += ["--checkpoint", str(checkpoint)]
        if icefall_dir:
            self.command += ["--icefall-dir", str(icefall_dir)]
        self.log_path = Path(log_path)
        self.environment = dict(os.environ)
        self.environment["PYTHONPATH"] = str(ROOT) + os.pathsep + self.environment.get("PYTHONPATH", "")
        self.environment["PYTHONUNBUFFERED"] = "1"
        self.environment["USE_TF"] = "0"
        if not allow_download:
            self.environment["HF_HUB_OFFLINE"] = "1"
            self.environment["TRANSFORMERS_OFFLINE"] = "1"
        self.process = None
        self.log = None
        self.ready = None

    def start(self, deadline):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        self.messages = queue.Queue()
        self.log = self.log_path.open("a", encoding="utf-8")
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                        text=True, encoding="utf-8", bufsize=1, env=self.environment)
        process, messages = self.process, self.messages

        def reader():
            try:
                for line in process.stdout:
                    try:
                        value = json.loads(line)
                        if isinstance(value, dict) and ("ready" in value or "ok" in value):
                            messages.put(value)
                    except json.JSONDecodeError:
                        # Ignore an upstream native diagnostic accidentally written to stdout.
                        continue
            finally:
                messages.put(None)
        threading.Thread(target=reader, daemon=True).start()
        response = self.receive(deadline)
        if not response.get("ready"):
            self.close()
            raise RuntimeError(f"Model startup failed: {response.get('error')}. See {self.log_path}")
        self.ready = response

    def receive(self, deadline):
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            self.close()
            raise TimeoutError("Sample deadline exceeded")
        try:
            response = self.messages.get(timeout=seconds)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError("Model exceeded its wall-clock deadline") from exc
        if response is None:
            self.close()
            raise RuntimeError(f"Model process exited. See {self.log_path}")
        return response

    def request(self, operation, payload, deadline):
        self.start(deadline)  # Reload after a killed process counts toward this sample's budget.
        payload = {"operation": operation, **payload, "seconds": max(0.001, deadline - time.monotonic())}
        try:
            self.process.stdin.write(compact(payload) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise RuntimeError(f"Model pipe failed; see {self.log_path}") from exc
        response = self.receive(deadline)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Model request failed"))
        return response

    def close(self):
        if self.process is not None:
            process, self.process = self.process, None
            if process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()
        if self.log is not None:
            self.log.close()
            self.log = None


def infer_sample(item, tools, env, asr, llm, root, split, *, max_seconds=30, max_tool_calls=12, max_turns=20, max_repairs=2, started=None):
    started = time.monotonic() if started is None else started
    deadline = started + max_seconds
    trace = {"id": item["id"], "status": "error", "steps": []}

    def remaining():
        if time.monotonic() >= deadline:
            raise TimeoutError("Sample deadline exceeded")

    try:
        history = copy.deepcopy(item.get("history", []))
        user_turns = [h for h in history if h["role"] == "user"]
        paths = [str(audio_path(root, split, h["audio"])) for h in user_turns]
        paths.append(str(audio_path(root, split, item["audio"])))
        texts = asr.request("transcribe", {"audio": paths}, deadline)["texts"]
        if len(texts) != len(paths) or not all(isinstance(t, str) and t.strip() for t in texts):
            raise ValueError("ASR returned empty or mismatched transcripts")
        for turn, text in zip(user_turns, texts):
            turn["audio_text"] = text
        trace["transcripts"] = texts
        available = installed_tools(item, tools)
        allowed = {t["name"] for t in available}
        record = {"history": history, "context": item["context"], "request": texts[-1], "tools": available, "calls": [], "results": []}
        messages, targets = trajectory(record)
        messages = messages[:targets[0]]
        repairs = 0
        for _ in range(max_turns):
            remaining()
            response = llm.request("generate", {"messages": messages}, deadline)
            text = response["text"]
            entry = {"response": text, "input_tokens": response.get("input_tokens")}
            trace["steps"].append(entry)
            try:
                calls = parse_response(text)
                if calls:
                    call = calls[0]
                    if call["name"] not in allowed:
                        raise ValueError("Tool is not installed for this sample")
                    validation = env.validate(call["name"], call["arguments"])
                    if not validation["ok"]:
                        raise ValueError(validation["error"])
            except (ValueError, TypeError, KeyError) as exc:
                entry["validation_error"] = str(exc)
                if repairs >= max_repairs:
                    trace["status"] = "invalid_response"
                    break
                repairs += 1
                messages.extend([{"role": "assistant", "content": text},
                                 {"role": "user", "content": "Chưa thực hiện công cụ nào từ câu trả lời vừa rồi. Sửa câu trả lời JSON theo schema. Lỗi: " + str(exc)}])
                continue
            if not calls:
                trace["status"] = "stopped"
                break
            if len(env.get_tool_calls()) >= max_tool_calls:
                trace["status"] = "tool_limit"
                break
            remaining()
            result = env.call(call["name"], call["arguments"])
            entry["tool_result"] = result
            messages.extend([{"role": "assistant", "content": action(calls)},
                             {"role": "tool", "content": compact({"name": call["name"], "result": result})}])
        else:
            trace["status"] = "turn_limit"
        if time.monotonic() >= deadline:
            trace["status"] = "timeout"
    except KeyboardInterrupt:
        trace.update(status="interrupted", error="Interrupted; all calls already executed are retained")
    except TimeoutError as exc:
        trace.update(status="timeout", error=str(exc))
    except Exception as exc:
        trace.update(status="error", error=f"{type(exc).__name__}: {exc}")
    # The submitted calls/hash are always obtained from the single live environment.
    result = {"id": item["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash()}
    trace["elapsed_seconds"] = time.monotonic() - started
    return result, trace


def validate_records(rows, items, catalog, package):
    expected = [x["id"] for x in items]
    if [x.get("id") for x in rows] != expected:
        raise ValueError("Output IDs must match input order exactly, with one row per sample and no duplicates")
    tools = {t["name"]: t for app in catalog["apps"] for t in app["tools"]}
    for row, item in zip(rows, items):
        if set(row) != {"id", "tool_calls", "state_hash"} or not isinstance(row["tool_calls"], list):
            raise ValueError(f"Malformed submission row: {row.get('id')}")
        if not isinstance(row["state_hash"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["state_hash"]):
            raise ValueError(f"Invalid SHA256 state hash: {row['id']}")
        allowed = {t["name"] for t in installed_tools(item, tools)}
        for call in row["tool_calls"]:
            if not isinstance(call, dict) or set(call) != {"name", "arguments"} or not isinstance(call["name"], str) or not isinstance(call["arguments"], dict):
                raise ValueError(f"Malformed tool call: {row['id']}")
            if call["name"] not in allowed:
                raise ValueError(f"Uninstalled tool {call['name']} in {row['id']}")
        for check in package.schema_validate(row["tool_calls"], catalog):
            if not check["ok"]:
                raise ValueError(f"Schema error in {row['id']}: {check}")
    return {"valid": True, "samples": len(rows), "hashes_replayed": False,
            "note": "Coverage, order, schemas and hash format checked; state hashes are not independently replayed."}


def validate_file(root, split, path):
    items, catalog, _ = inputs(root, split)
    return validate_records(read_jsonl(path), items, catalog, runtime(root, split))


def model_config(args, kind):
    proxy = types.SimpleNamespace(config=getattr(args, f"{kind}_config"), model=getattr(args, f"{kind}_model"),
                                  revision=getattr(args, f"{kind}_revision"), checkpoint=getattr(args, f"{kind}_checkpoint"),
                                  tokenizer=None, profile=args.profile, set=getattr(args, f"{kind}_set"))
    return resolve_config(proxy, kind, allowed_backends=("zipformer_rnnt", "hf_whisper") if kind == "asr" else None)


def worker_python(args, kind, config):
    value = getattr(args, f"{kind}_python")
    if value:
        return Path(value).expanduser().resolve()
    environment = "llm" if kind == "llm" or config["backend"] == "hf_whisper" else "asr"
    home = Path(os.environ.get("FINETUNE_HOME", ROOT))
    path = home / ".venvs" / environment / "bin" / "python"
    if not path.is_file():
        raise ValueError(f"Missing {path}; run bash finetune.sh setup {environment}, or set --{kind}-python.")
    return path


def run(args):
    if not 0 < args.max_seconds <= 30:
        raise ValueError("--max-seconds must be > 0 and <= the competition's 30-second budget")
    if args.limit < 0 or args.max_tool_calls < 1 or args.max_turns < 1 or args.max_repairs < 0:
        raise ValueError("Invalid sample/call/turn/repair limits")
    root = Path(args.data_root).expanduser().resolve()
    items, catalog, tools = inputs(root, args.split)
    if args.limit:
        items = items[:args.limit]
    package = runtime(root, args.split)
    configs = {kind: model_config(args, kind) for kind in ("asr", "llm")}
    if any(int(c.get("devices", 1)) != 1 for c in configs.values()):
        raise ValueError("Submission runs use one GPU; choose gpu16/gpu24/large, not a DDP profile")
    maximum = configs["llm"]["training"]["max_seq_length"]
    if not 0 < maximum <= 16384:
        raise ValueError("LLM context must be between 1 and 16,384 tokens")
    interpreters = {kind: worker_python(args, kind, config) for kind, config in configs.items()}
    print("Checking dataset and selected local checkpoint fingerprints ...", flush=True)
    audio_hashes = {}
    for item in items:
        paths = [item["audio"]] + [h["audio"] for h in item.get("history", []) if h["role"] == "user"]
        for relative in paths:
            if relative not in audio_hashes:
                audio_hashes[relative] = file_hash(audio_path(root, args.split, relative))
    artifacts = {kind: {"model": artifact_signature(config["model"]),
                        "checkpoint": artifact_signature(getattr(args, f"{kind}_checkpoint"))}
                 for kind, config in configs.items()}
    identity_configs = copy.deepcopy(configs)
    for config in identity_configs.values():
        config.pop("config_path", None)
    # Fingerprint only documented inputs/catalog and supplied runtime code, never app states.
    runtime_files = {str(p.relative_to(root / args.split)): p.read_text(encoding="utf-8")
                     for p in sorted((root / args.split / "zaic_tools").rglob("*.py"))}
    identity = digest({"items": items, "audio_sha256": audio_hashes, "catalog": catalog, "runtime": runtime_files,
                       "configs": identity_configs, "artifacts": artifacts,
                       "settings": [args.split, args.limit, args.max_seconds, args.max_tool_calls, args.max_turns, args.max_repairs]})
    out = Path(args.output).expanduser().resolve()
    ledger = out / "results.jsonl"
    if args.resume:
        saved = json.loads((out / "submission_run.json").read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError("Resume identity changed: inputs, catalog, runtime, models or generation settings differ")
        if (out / "inflight.json").exists():
            raise ValueError("An unfinished sample marker remains. Refusing to recreate its ToolEnv or silently replay actions; inspect the interrupted run.")
        results = read_jsonl(ledger) if ledger.exists() else []
        validate_records(results, items[:len(results)], catalog, package)
    else:
        out = fresh_directory(out)
        results = []
        write_json(out / "submission_run.json", {"identity": identity, "split": args.split, "preview": bool(args.limit),
                                                "configs": configs, "max_seconds": args.max_seconds})
    workers = {}
    try:
        for kind, config in configs.items():
            config_path = out / f"{kind}_config.json"
            write_json(config_path, config)
            workers[kind] = Worker(interpreters[kind], kind, config_path, out / f"{kind}_worker.log",
                                   getattr(args, f"{kind}_checkpoint"), args.icefall_dir, args.allow_download)
            print(f"Loading {kind}: {config['model']} ...", flush=True)
            workers[kind].start(time.monotonic() + args.startup_timeout)
        metadata = {k: w.ready for k, w in workers.items()}
        prior_models = out / "loaded_models.json"
        if args.resume and prior_models.exists() and json.loads(prior_models.read_text()) != metadata:
            raise ValueError("Resolved worker model revisions/counts changed since the original run")
        write_json(prior_models, metadata)
        if metadata["llm"]["parameters"] >= 5_000_000_000 or sum(m["parameters"] for m in metadata.values()) >= 6_000_000_000:
            raise ValueError("Selected checkpoints exceed the LLM <5B / whole-system <6B parameter limits")
        with ledger.open("a", encoding="utf-8") as stream, (out / "traces.jsonl").open("a", encoding="utf-8") as traces:
            for item in items[len(results):]:
                # A marker guards against silently resetting a half-executed sample after a hard crash.
                write_json(out / "inflight.json", {"id": item["id"]})
                started = time.monotonic()
                env = package.ToolEnv(args.split, item["id"], root=root)  # exactly once for this sample
                result, trace = infer_sample(item, tools, env, workers["asr"], workers["llm"], root, args.split,
                                             max_seconds=args.max_seconds, max_tool_calls=args.max_tool_calls,
                                             max_turns=args.max_turns, max_repairs=args.max_repairs, started=started)
                stream.write(compact(result) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                traces.write(compact(trace) + "\n")
                traces.flush()
                results.append(result)
                (out / "inflight.json").unlink()
                print(f"{len(results)}/{len(items)} {item['id']}: {trace['status']}, {len(result['tool_calls'])} calls, {trace['elapsed_seconds']:.2f}s", flush=True)
                if trace["status"] == "interrupted":
                    raise KeyboardInterrupt
        report = validate_records(results, items, catalog, package)
        filename = "preview.jsonl" if args.limit else "submission.jsonl"
        destination = out / filename
        if destination.exists():
            if read_jsonl(destination) != results:
                raise ValueError(f"Refusing to overwrite differing existing output: {destination}")
        else:
            temporary = out / (filename + ".tmp")
            with temporary.open("x", encoding="utf-8") as stream:
                for result in results:
                    stream.write(compact(result) + "\n")
            temporary.replace(destination)
        statuses = read_jsonl(out / "traces.jsonl")
        counts = {}
        for trace in statuses:
            counts[trace["status"]] = counts.get(trace["status"], 0) + 1
        report.update({"complete_split": not bool(args.limit), "statuses": counts, "output": str(destination)})
        write_json(out / "summary.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        for worker in workers.values():
            worker.close()
