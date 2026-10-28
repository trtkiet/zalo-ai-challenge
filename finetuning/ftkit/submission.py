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
import signal
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
    source = directory / f"{split}.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}. DATA_ROOT/--data-root must point to the folder "
                                f"containing {split}/. Check .ENV and any exported DATA_ROOT override.")
    rows = read_jsonl(source)
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
        self.environment["VLLM_NO_USAGE_STATS"] = "1"
        self.environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
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
                                        text=True, encoding="utf-8", bufsize=1, env=self.environment,
                                        start_new_session=os.name != "nt")
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

    def request_batch(self, operation, payloads, deadline):
        response = self.request(operation + "_batch", {"requests": payloads}, deadline)
        results = response.get("results")
        if (not isinstance(results, list) or len(results) != len(payloads)
                or any(not isinstance(result, dict) or type(result.get("ok")) is not bool for result in results)):
            raise RuntimeError("Worker returned mismatched batch results")
        return results

    def close(self):
        if self.process is not None:
            process, self.process = self.process, None
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.kill()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            for stream in (process.stdin, process.stdout):
                if stream:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass
        if self.log is not None:
            self.log.close()
            self.log = None


def sample_steps(item, tools, env, root, split, *, max_seconds=30, max_tool_calls=12, max_turns=20, max_repairs=2, started=None):
    """Yield model requests, preserving one environment and one deadline per sample."""
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
        asr_started = time.monotonic()
        asr_response = yield ("asr", "transcribe", {"audio": paths}, deadline)
        texts = asr_response["texts"]
        trace["asr_seconds"] = time.monotonic() - asr_started
        trace["asr_scheduler_wait_seconds"] = asr_response.get("scheduler_wait_seconds", 0)
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
            generation_started = time.monotonic()
            response = yield ("llm", "generate", {"messages": messages, "tools": available,
                                                   "max_calls": max(0, max_tool_calls - len(env.get_tool_calls()))}, deadline)
            remaining()  # Never execute a response returned after this sample's deadline.
            text = response["text"]
            entry = {"response": text, "input_tokens": response.get("input_tokens"),
                     "generation_seconds": time.monotonic() - generation_started}
            for key in ("output_tokens", "ttft_seconds", "decode_seconds", "engine_seconds", "cached_tokens",
                        "finish_reason", "gpu_memory_used_bytes", "scheduler_wait_seconds", "worker_queue_seconds"):
                if key in response:
                    entry[key] = response[key]
            trace["steps"].append(entry)
            try:
                calls = parse_response(text)
                for call in calls:
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
            entry["executed"] = []
            for call in calls:
                remaining()
                if len(env.get_tool_calls()) >= max_tool_calls:
                    trace["status"] = "tool_limit"
                    break
                result = env.call(call["name"], call["arguments"])
                entry["executed"].append({"call": call, "result": result})
                entry["tool_result"] = result  # Backward-compatible last-result field.
                messages.extend([{"role": "assistant", "content": action([call])},
                                 {"role": "tool", "content": compact({"name": call["name"], "result": result})}])
                if result.get("ok") is False or result.get("done") is False or call["name"] == "sys.ask_missing_info":
                    entry["discarded_calls"] = len(calls) - len(entry["executed"])
                    break
            if trace["status"] == "tool_limit":
                break
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


def infer_sample(item, tools, env, asr, llm, root, split, **settings):
    """Drive the same sample state machine using individual model requests."""
    steps = sample_steps(item, tools, env, root, split, **settings)
    workers = {"asr": asr, "llm": llm}
    response, error = None, None
    while True:
        try:
            request = steps.throw(error) if error is not None else steps.send(response)
        except StopIteration as done:
            return done.value
        kind, operation, payload, deadline = request
        try:
            response = workers[kind].request(operation, payload, deadline)
            error = None
        except (Exception, KeyboardInterrupt) as exc:
            error = exc


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
    environment = "llm-infer" if kind == "llm" or config["backend"] == "hf_whisper" else "asr"
    home = Path(os.environ.get("FINETUNE_HOME", ROOT))
    path = home / ".venvs" / environment / "bin" / "python"
    if not path.is_file():
        raise ValueError(f"Missing {path}; run bash finetune.sh setup {environment}, or set --{kind}-python.")
    return path


def run(args):
    from .continuous import AsyncWorker, continuous_samples
    from .llm_common import SCHEDULER_VERSION, llm_identity, inference_settings
    if not 0 < args.max_seconds <= 30:
        raise ValueError("--max-seconds must be > 0 and <= the competition's 30-second budget")
    if args.limit < 0 or args.max_tool_calls < 1 or args.max_turns < 1 or args.max_repairs < 0:
        raise ValueError("Invalid sample/call/turn/repair limits")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    root = Path(args.data_root).expanduser().resolve()
    items, catalog, tools = inputs(root, args.split)
    if args.limit:
        items = items[:args.limit]
    package = runtime(root, args.split)
    configs = {kind: model_config(args, kind) for kind in ("asr", "llm")}
    if any(int(c.get("devices", 1)) != 1 for c in configs.values()):
        raise ValueError("Submission runs use one GPU; choose gpu16/gpu24/large, not a DDP profile")
    configs["llm"].setdefault("inference", {})["max_num_seqs"] = args.batch_size
    maximum = inference_settings(configs["llm"])["max_seq_length"]
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
    batching = {"batch_size": args.batch_size, "scheduler": SCHEDULER_VERSION, "llm_contract": llm_identity()}
    identity = digest({"items": items, "audio_sha256": audio_hashes, "catalog": catalog, "runtime": runtime_files,
                       "configs": identity_configs, "artifacts": artifacts,
                       "settings": [args.split, args.limit, args.max_seconds, args.max_tool_calls, args.max_turns, args.max_repairs],
                       **batching})
    out = Path(args.output).expanduser().resolve()
    ledger = out / "results.jsonl"
    journal = out / "completed.jsonl"
    finalized = {}
    old_traces = []
    if args.resume:
        saved = json.loads((out / "submission_run.json").read_text(encoding="utf-8"))
        if saved["identity"] != identity:
            raise ValueError("Resume identity changed: inputs, catalog, runtime, models or generation settings differ")
        records = read_jsonl(journal) if journal.exists() else []
        for row in records:
            sid = row["result"]["id"]
            if sid in finalized:
                raise ValueError("Duplicate finalized sample in completion journal")
            finalized[sid] = row["result"]
            old_traces.append(row["trace"])
        expected = {item["id"]: item for item in items}
        if finalized.keys() - expected.keys():
            raise ValueError("Journal contains unknown sample IDs")
        for sid, result in finalized.items():
            validate_records([result], [expected[sid]], catalog, package)
        marker = out / "inflight.json"
        if marker.exists():
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
            if not isinstance(marker_data.get("ids"), list):
                raise ValueError("An unfinished sample marker remains in an incompatible format")
            unfinished = set(marker_data["ids"]) - finalized.keys()
            if unfinished:
                raise ValueError("An unfinished sample marker remains. Refusing to recreate its ToolEnv or replay actions")
            marker.unlink()  # A crash after durable completion but before marker cleanup is recoverable.
        # Secondary readable ledgers can be reconstructed after an interrupted append.
        from .common import write_jsonl
        write_jsonl(ledger, finalized.values())
        write_jsonl(out / "traces.jsonl", old_traces)
    else:
        out = fresh_directory(out)
        write_json(out / "submission_run.json", {"identity": identity, "split": args.split, "preview": bool(args.limit),
                                                "configs": configs, "max_seconds": args.max_seconds,
                                                "batch_size": args.batch_size, "scheduler": SCHEDULER_VERSION})
    workers = {}
    previous_sigint = None
    try:
        for kind, config in configs.items():
            config_path = out / f"{kind}_config.json"
            write_json(config_path, config)
            workers[kind] = AsyncWorker(interpreters[kind], kind, config_path, out / f"{kind}_worker.log",
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
        inference_started, prior_count = time.monotonic(), len(finalized)
        run_id = str(time.time_ns())
        inflight = set()
        stop_event = threading.Event()
        if threading.current_thread() is threading.main_thread():
            previous_sigint = signal.signal(signal.SIGINT, lambda *unused: stop_event.set())

        def admit(sid):
            inflight.add(sid)
            write_json(out / "inflight.json", {"ids": sorted(inflight)})

        remaining_items = [item for item in items if item["id"] not in finalized]
        with journal.open("a", encoding="utf-8") as durable, ledger.open("a", encoding="utf-8") as stream, (out / "traces.jsonl").open("a", encoding="utf-8") as traces:
            settings = dict(max_seconds=args.max_seconds, max_tool_calls=args.max_tool_calls,
                            max_turns=args.max_turns, max_repairs=args.max_repairs)
            for result, trace in continuous_samples(remaining_items, tools, package, workers, root, args.split,
                                                     capacity=args.batch_size, on_admit=admit,
                                                     should_stop=stop_event.is_set, **settings):
                trace["completion_index"] = len(finalized)
                trace["run_id"] = run_id
                trace["admitted_at_seconds"] = trace.pop("admitted_at_monotonic") - inference_started
                trace["completed_at_seconds"] = time.monotonic() - inference_started
                durable.write(compact({"result": result, "trace": trace}) + "\n")
                durable.flush()
                os.fsync(durable.fileno())
                stream.write(compact(result) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                traces.write(compact(trace) + "\n")
                traces.flush()
                os.fsync(traces.fileno())
                finalized[result["id"]] = result
                inflight.discard(result["id"])
                if inflight:
                    write_json(out / "inflight.json", {"ids": sorted(inflight)})
                else:
                    (out / "inflight.json").unlink(missing_ok=True)
                print(f"{len(finalized)}/{len(items)} {result['id']}: {trace['status']}, {len(result['tool_calls'])} calls, {trace['elapsed_seconds']:.2f}s", flush=True)
        inference_seconds = time.monotonic() - inference_started
        results = [finalized[item["id"]] for item in items if item["id"] in finalized]
        if len(results) != len(items):
            raise KeyboardInterrupt("Active samples finalized; unadmitted samples can be resumed")
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
        report.update({"complete_split": not bool(args.limit), "statuses": counts, "output": str(destination),
                       "scheduler": SCHEDULER_VERSION, "batch_size": args.batch_size, "processed_this_run": len(results) - prior_count,
                       "inference_seconds": inference_seconds,
                       "samples_per_second": (len(results) - prior_count) / max(inference_seconds, 1e-9)})
        latencies = sorted(t["elapsed_seconds"] for t in statuses)
        report["latency_seconds"] = {"p50": percentile(latencies, 0.5), "p95": percentile(latencies, 0.95)}
        report["timeout_rate"] = counts.get("timeout", 0) / len(statuses) if statuses else 0
        report["repairs"] = sum("validation_error" in step for t in statuses for step in t["steps"])
        report["peak_observed_gpu_memory_bytes"] = max((step["gpu_memory_used_bytes"] for t in statuses for step in t["steps"]
                                                       if step.get("gpu_memory_used_bytes") is not None), default=None)
        write_json(out / "summary.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if previous_sigint is not None:
            signal.signal(signal.SIGINT, previous_sigint)
        for worker in workers.values():
            worker.close()


def percentile(values, fraction):
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)
