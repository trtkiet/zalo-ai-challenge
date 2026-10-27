import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from cli import parser
from ftkit.common import read_jsonl, write_json
from ftkit.data import action, validate_action
from ftkit.submission import (Worker, audio_path, infer_batch, infer_sample, parse_response, run,
                              validate_records, worker_python)
from ftkit.batched_inference import generate_batch, left_pad, transcribe_batch


TOOLS = {
    "contacts.search": {"name": "contacts.search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    "sms.send": {"name": "sms.send", "parameters": {"type": "object", "properties": {"contact_id": {"type": "string"}, "content": {"type": "string"}}, "required": ["contact_id", "content"]}},
    "sys.ask_missing_info": {"name": "sys.ask_missing_info", "parameters": {"type": "object", "properties": {"tool_name": {"type": "string"}, "params": {"type": "array", "items": {"type": "string"}}}, "required": ["tool_name", "params"]}}
}


def item(sid="synthetic_1"):
    return {"id": sid, "audio": f"audio/{sid}/current.wav", "context": {"installed_apps": {
        "contacts": {"tools": ["contacts.search"]}, "sms": {"tools": ["sms.send"]}}},
        "history": [{"role": "user", "audio": f"audio/{sid}/history.wav"},
                    {"role": "assistant", "text": "Earlier action already applied", "tool_calls": [{"name": "sms.send", "arguments": {"contact_id": "past", "content": "past"}}],
                     "tool_results": [{"ok": True}]}]}


class FakeEnv:
    def __init__(self):
        self.calls = []
    def validate(self, name, arguments):
        valid = validate_action({"tool_calls": [{"name": name, "arguments": arguments}]}, list(TOOLS.values()))
        return {"ok": valid, "error": None if valid else "schema mismatch"}
    def call(self, name, arguments):
        self.calls.append({"name": name, "arguments": copy.deepcopy(arguments)})
        return {"contacts": [{"id": "LIVE_ID"}]} if name == "contacts.search" else {"ok": True}
    def get_tool_calls(self):
        return copy.deepcopy(self.calls)
    def get_state_hash(self):
        return hashlib.sha256(json.dumps(self.calls, sort_keys=True).encode()).hexdigest()


class FakeASR:
    def request(self, operation, payload, deadline):
        self.paths = payload["audio"]
        return {"texts": ["history transcribed", "request transcribed"]}


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []
    def request(self, operation, payload, deadline):
        self.prompts.append(copy.deepcopy(payload["messages"]))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return {"text": response}


class BatchTests(unittest.TestCase):
    def workers(self, scripts):
        class ASR:
            def request_batch(self, operation, payloads, deadline):
                self.sizes = getattr(self, "sizes", []) + [len(payloads)]
                return [{"ok": True, "texts": [Path(p).parent.name for p in payload["audio"]]}
                        for payload in payloads]
        class LLM:
            def __init__(self):
                self.positions = {sid: 0 for sid in scripts}
                self.prompts = []
                self.sizes = []
            def request_batch(self, operation, payloads, deadline):
                self.sizes.append(len(payloads))
                responses = []
                for payload in payloads:
                    self.prompts.append(copy.deepcopy(payload["messages"]))
                    sid = next(sid for sid in scripts if sid in payload["messages"][1]["content"])
                    response = scripts[sid][self.positions[sid]]
                    self.positions[sid] += 1
                    responses.append(response if isinstance(response, dict) else {"ok": True, "text": response})
                return responses
        return {"asr": ASR(), "llm": LLM()}

    def package(self):
        class Package:
            initialized = []
            @classmethod
            def ToolEnv(cls, split, sid, root):
                cls.initialized.append(sid)
                return FakeEnv()
        return Package

    def test_independent_trajectories_shrink_batches_and_keep_order(self):
        first = {"name": "contacts.search", "arguments": {"query": "mẹ"}}
        second = {"name": "sms.send", "arguments": {"contact_id": "LIVE_ID", "content": "hello"}}
        workers = self.workers({"synthetic_1": [action([first]), action([second]), action([])],
                                "synthetic_2": [action([])]})
        package = self.package()
        rows = infer_batch([item("synthetic_1"), item("synthetic_2")], TOOLS, package, workers, Path.cwd(), "public_test")
        self.assertEqual(package.initialized, ["synthetic_1", "synthetic_2"])
        self.assertEqual(workers["asr"].sizes, [2])
        self.assertEqual(workers["llm"].sizes, [2, 1, 1])
        self.assertEqual([row[0]["id"] for row in rows], package.initialized)
        self.assertEqual(rows[0][0]["tool_calls"], [first, second])
        self.assertEqual(rows[1][0]["tool_calls"], [])
        self.assertEqual([row[1]["status"] for row in rows], ["stopped", "stopped"])
        self.assertIn("LIVE_ID", json.dumps(workers["llm"].prompts[2]))
        self.assertNotIn("LIVE_ID", json.dumps(workers["llm"].prompts[1]))

    def test_batch_repairs_and_model_errors_are_isolated(self):
        workers = self.workers({"synthetic_1": ["invalid", action([])],
                                "synthetic_2": [{"ok": False, "error": "context overflow"}]})
        rows = infer_batch([item("synthetic_1"), item("synthetic_2")], TOOLS, self.package(), workers, Path.cwd(), "public_test")
        self.assertEqual(rows[0][1]["status"], "stopped")
        self.assertEqual(rows[1][1]["status"], "error")
        self.assertIn("context overflow", rows[1][1]["error"])
        self.assertTrue(all(not row[0]["tool_calls"] for row in rows))

    def test_timeout_or_interrupt_preserves_each_samples_executed_calls(self):
        call = {"name": "contacts.search", "arguments": {"query": "mẹ"}}
        for exception, status in [(TimeoutError("deadline"), "timeout"), (KeyboardInterrupt(), "interrupted")]:
            workers = self.workers({"synthetic_1": [action([call])], "synthetic_2": [action([call])]})
            original = workers["llm"].request_batch
            def request(operation, payloads, deadline):
                if workers["llm"].sizes:
                    raise exception
                return original(operation, payloads, deadline)
            workers["llm"].request_batch = request
            rows = infer_batch([item("synthetic_1"), item("synthetic_2")], TOOLS, self.package(), workers, Path.cwd(), "public_test")
            self.assertEqual([row[1]["status"] for row in rows], [status, status])
            self.assertEqual([row[0]["tool_calls"] for row in rows], [[call], [call]])

    def test_late_model_response_never_executes_tools(self):
        workers = self.workers({"synthetic_1": []})
        def request(operation, payloads, deadline):
            time.sleep(0.03)
            return [{"ok": True, "text": action([{"name": "contacts.search", "arguments": {"query": "mẹ"}}])}]
        workers["llm"].request_batch = request
        rows = infer_batch([item()], TOOLS, self.package(), workers, Path.cwd(), "public_test", max_seconds=0.02)
        self.assertEqual(rows[0][1]["status"], "timeout")
        self.assertEqual(rows[0][0]["tool_calls"], [])

    def test_left_padding_preserves_prompts_and_masks(self):
        ids, masks = left_pad([[8, 9], [1, 2, 3, 4]], 0)
        self.assertEqual(ids, [[0, 0, 8, 9], [1, 2, 3, 4]])
        self.assertEqual(masks, [[0, 0, 1, 1], [1, 1, 1, 1]])

    def test_batched_llm_padding_output_slicing_and_overflow_isolation(self):
        # Exercise the generation wrapper without importing GPU libraries.
        import types
        class Tensor:
            def __init__(self, rows):
                self.rows = rows
            def __getitem__(self, key):
                return [row[key[1]] for row in self.rows[key[0]]]
        class Model:
            device = "test"
            def generate(self, **kwargs):
                self.kwargs = kwargs
                return Tensor([row + [99 + i] for i, row in enumerate(kwargs["input_ids"].rows)])
        class Tokenizer:
            pad_token_id = 0
            def apply_chat_template(self, messages, **kwargs):
                return messages
            def batch_decode(self, rows, **kwargs):
                return [str(row[0]) for row in rows]
        model = Model()
        config = {"training": {"max_seq_length": 6, "generation_max_new_tokens": 2}}
        with patch.dict(sys.modules, {"torch": types.SimpleNamespace(tensor=lambda rows, **kwargs: Tensor(rows))}):
            results = generate_batch(model, Tokenizer(), config,
                                     [{"messages": [1]}, {"messages": [2, 3, 4]}, {"messages": list(range(6))}], 5)
        self.assertEqual([r.get("text") for r in results], ["99", "100", None])
        self.assertIn("Context overflow", results[2]["error"])
        self.assertEqual(model.kwargs["input_ids"].rows, [[0, 0, 1], [2, 3, 4]])
        self.assertEqual(model.kwargs["attention_mask"].rows, [[0, 0, 1], [1, 1, 1]])
        self.assertEqual([r.get("input_tokens") for r in results], [1, 3, None])

    def test_gipformer_batches_clips_and_restores_sample_history_order(self):
        import wave
        import types
        class Model:
            def forward_encoder(self, values, lengths):
                return values, lengths
        sizes = []
        def features(loaded, paths):
            sizes.append(len(paths))
            return [Path(p).stem for p in paths], [1] * len(paths)
        loaded = {"decode": lambda **kw: kw["encoder_out"],
                  "processor": types.SimpleNamespace(decode=lambda row: row)}
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for name in ("a_history", "a_current", "b_current"):
                path = Path(td) / f"{name}.wav"
                with wave.open(str(path), "wb") as audio:
                    audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    audio.writeframes(b"\x00\x00" * 16)
                paths.append(str(path))
            with patch("ftkit.backends.asr.features", features):
                rows = transcribe_batch(Model(), {"backend": "zipformer_rnnt"},
                                        [{"audio": paths[:2]}, {"audio": paths[2:]}, {"audio": [str(Path(td) / "missing.wav")]}],
                                        5, loaded=loaded)
        self.assertEqual(rows[0]["texts"], ["a_history", "a_current"])
        self.assertEqual(rows[1]["texts"], ["b_current"])
        self.assertFalse(rows[2]["ok"])
        self.assertEqual(sizes, [3])

    def test_whisper_batches_chunks_and_restores_audio_order(self):
        import types
        import wave
        class Array(list):
            def astype(self, dtype):
                return self
            def __truediv__(self, value):
                return self
        class Tensor:
            def __init__(self, values):
                self.values = values
            def to(self, *args, **kwargs):
                return self
        class Processor:
            def __call__(self, waveforms, **kwargs):
                return {"input_features": Tensor([len(w) for w in waveforms]), "attention_mask": Tensor([])}
            def batch_decode(self, tokens, **kwargs):
                return list(map(str, tokens))
        class Model:
            device, dtype = "fake_cuda", "float16"
            sizes = []
            def generate(self, **kwargs):
                self.assert_mask = "attention_mask" in kwargs
                self.sizes.append(len(kwargs["input_features"].values))
                return kwargs["input_features"].values
        model = Model()
        numpy = types.SimpleNamespace(frombuffer=lambda data, **kwargs: Array([0] * (len(data) // 2)), float32="float32")
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i, count in enumerate((16000, 31 * 16000, 8000)):
                path = Path(td) / f"clip{i}.wav"
                with wave.open(str(path), "wb") as audio:
                    audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    audio.writeframes(b"\x00\x00" * count)
                paths.append(str(path))
            with patch.dict(sys.modules, {"numpy": numpy}):
                rows = transcribe_batch(model, {"backend": "hf_whisper", "architecture": {"language": "vi"}},
                                        [{"audio": paths[:2]}, {"audio": paths[2:]}], 5, processor=Processor())
        self.assertEqual(rows[0]["texts"], ["16000", "480000 16000"])
        self.assertEqual(rows[1]["texts"], ["8000"])
        self.assertEqual(model.sizes, [2, 2])
        self.assertTrue(model.assert_mask)


class SubmissionTests(unittest.TestCase):
    def test_live_results_history_and_exact_log(self):
        env, asr = FakeEnv(), FakeASR()
        calls = [{"name": "contacts.search", "arguments": {"query": "mẹ"}},
                 {"name": "sms.send", "arguments": {"contact_id": "LIVE_ID", "content": "con về muộn"}}]
        llm = FakeLLM([action([calls[0]]), action([calls[1]]), action([])])
        result, trace = infer_sample(item(), TOOLS, env, asr, llm, Path.cwd(), "public_test")
        self.assertEqual(result["tool_calls"], calls)
        self.assertEqual(result["state_hash"], env.get_state_hash())
        self.assertEqual(trace["status"], "stopped")
        self.assertNotIn("LIVE_ID", json.dumps(llm.prompts[0]))
        self.assertIn("LIVE_ID", json.dumps(llm.prompts[1]))
        self.assertIn("history transcribed", json.dumps(llm.prompts[0]))
        self.assertEqual(len(asr.paths), 2)

    def test_invalid_output_is_repaired_before_execution(self):
        env = FakeEnv()
        llm = FakeLLM(["bad JSON", action([{"name": "sms.send", "arguments": {}}]), action([])])
        result, trace = infer_sample(item(), TOOLS, env, FakeASR(), llm, Path.cwd(), "public_test")
        self.assertEqual(result["tool_calls"], [])
        self.assertEqual(trace["status"], "stopped")
        self.assertEqual(sum("validation_error" in s for s in trace["steps"]), 2)

    def test_uninstalled_call_is_never_executed(self):
        llm = FakeLLM([action([{"name": "banking.transfer", "arguments": {}}])])
        result, trace = infer_sample(item(), TOOLS, FakeEnv(), FakeASR(), llm, Path.cwd(), "public_test", max_repairs=0)
        self.assertEqual(trace["status"], "invalid_response")
        self.assertEqual(result["tool_calls"], [])

    def test_clarification_and_empty_calls(self):
        call = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["content"]}}
        result, trace = infer_sample(item(), TOOLS, FakeEnv(), FakeASR(), FakeLLM([action([call]), action([])]), Path.cwd(), "public_test")
        self.assertEqual(result["tool_calls"], [call])
        self.assertEqual(trace["status"], "stopped")

    def test_timeout_and_interrupt_keep_actual_calls(self):
        call = {"name": "contacts.search", "arguments": {"query": "mẹ"}}
        for exception, status in [(TimeoutError("test"), "timeout"), (KeyboardInterrupt(), "interrupted")]:
            result, trace = infer_sample(item(), TOOLS, FakeEnv(), FakeASR(), FakeLLM([action([call]), exception]), Path.cwd(), "public_test")
            self.assertEqual(trace["status"], status)
            self.assertEqual(result["tool_calls"], [call])

    def test_limits_do_not_add_unexecuted_calls(self):
        call = {"name": "contacts.search", "arguments": {"query": "mẹ"}}
        result, trace = infer_sample(item(), TOOLS, FakeEnv(), FakeASR(), FakeLLM([action([call]), action([call])]), Path.cwd(), "public_test", max_tool_calls=1)
        self.assertEqual(trace["status"], "tool_limit")
        self.assertEqual(len(result["tool_calls"]), 1)

    def test_parser_and_paths(self):
        self.assertEqual(parse_response('```json\n{"tool_calls":[]}\n```'), [])
        for value in ('[]', '{"tool_calls":null}', '{"tool_calls":[{"name":[],"arguments":{}}]}'):
            with self.assertRaises((ValueError, TypeError)):
                parse_response(value)
        with self.assertRaises(ValueError):
            audio_path(Path.cwd(), "public_test", "../public_test_environment/a/state.json")

    def test_validation_coverage_and_shapes(self):
        class Package:
            @staticmethod
            def schema_validate(calls, catalog):
                return [{"ok": True} for c in calls]
        source = [item()]
        catalog = {"apps": [{"tools": list(TOOLS.values())}]}
        rows = [{"id": source[0]["id"], "tool_calls": [], "state_hash": FakeEnv().get_state_hash()}]
        self.assertTrue(validate_records(rows, source, catalog, Package)["valid"])
        with self.assertRaises(ValueError):
            validate_records(rows + rows, source, catalog, Package)
        rows[0]["state_hash"] = "invented"
        with self.assertRaises(ValueError):
            validate_records(rows, source, catalog, Package)


class OrchestratorTests(unittest.TestCase):
    def fixture(self, root, count=2):
        split = root / "data/public_test"
        split.mkdir(parents=True)
        records = [item(f"synthetic_{i + 1}") for i in range(count)]
        for record in records:
            for relative in (record["audio"], record["history"][0]["audio"]):
                path = split / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic audio; fake worker never decodes it")
        (split / "public_test.jsonl").write_text("\n".join(json.dumps(x) for x in records), encoding="utf-8")
        write_json(split / "public_test_tools_catalog.json", {"apps": [{"tools": list(TOOLS.values())}]})
        args = parser().parse_args(["submit", "--data-root", str(root / "data"), "--output", str(root / "run"),
                                    "--asr-python", sys.executable, "--llm-python", sys.executable])
        return args

    def test_batched_run_resume_and_settings_fingerprint(self):
        initialized, batch_sizes = [], []
        class Package:
            @staticmethod
            def ToolEnv(split, sid, root):
                initialized.append(sid)
                return FakeEnv()
            @staticmethod
            def schema_validate(calls, catalog):
                return [{"ok": True} for c in calls]
        class ModelWorker:
            def __init__(self, python, kind, *args):
                self.kind = kind
                self.ready = {"ready": True, "parameters": 1, "revision": "test"}
            def start(self, deadline):
                pass
            def request_batch(self, operation, payloads, deadline):
                batch_sizes.append((self.kind, len(payloads)))
                return [{"ok": True, "texts": ["history", "current"]} if self.kind == "asr"
                        else {"ok": True, "text": action([])} for payload in payloads]
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as td, patch("ftkit.submission.runtime", return_value=Package), patch("ftkit.submission.Worker", ModelWorker):
            root = Path(td)
            args = self.fixture(root, count=5)
            args.batch_size = 2
            run(args)
            rows = read_jsonl(root / "run/submission.jsonl")
            self.assertEqual([row["id"] for row in rows], initialized)
            self.assertEqual(len(initialized), 5)
            self.assertEqual(batch_sizes, [(kind, size) for size in (2, 2, 1) for kind in ("asr", "llm")])
            self.assertFalse((root / "run/inflight.json").exists())
            args.resume = True
            run(args)
            self.assertEqual(len(initialized), 5)
            args.batch_size = 3
            with self.assertRaisesRegex(ValueError, "Resume identity changed"):
                run(args)

    def test_batch_interrupt_finalizes_group_then_resumes_without_recreating_it(self):
        initialized = []
        class Package:
            @staticmethod
            def ToolEnv(split, sid, root):
                initialized.append(sid)
                return FakeEnv()
            @staticmethod
            def schema_validate(calls, catalog):
                return [{"ok": True} for c in calls]
        class ModelWorker:
            interrupt_once = True
            def __init__(self, python, kind, *args):
                self.kind = kind
                self.ready = {"ready": True, "parameters": 1, "revision": "test"}
            def start(self, deadline):
                pass
            def request_batch(self, operation, payloads, deadline):
                if self.kind == "llm" and ModelWorker.interrupt_once:
                    ModelWorker.interrupt_once = False
                    raise KeyboardInterrupt
                return [{"ok": True, "texts": ["history", "current"]} if self.kind == "asr"
                        else {"ok": True, "text": action([])} for payload in payloads]
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as td, patch("ftkit.submission.runtime", return_value=Package), patch("ftkit.submission.Worker", ModelWorker):
            root = Path(td)
            args = self.fixture(root, count=3)
            args.batch_size = 2
            with self.assertRaises(KeyboardInterrupt):
                run(args)
            self.assertFalse((root / "run/inflight.json").exists())
            self.assertEqual(len(read_jsonl(root / "run/results.jsonl")), 2)
            args.resume = True
            run(args)
            self.assertEqual(initialized, ["synthetic_1", "synthetic_2", "synthetic_3"])
            self.assertEqual(len(read_jsonl(root / "run/submission.jsonl")), 3)

    def test_batched_worker_transport(self):
        code = ("import sys,json; print(json.dumps({'ready':True}),flush=True);\n"
                "for line in sys.stdin: print(json.dumps({'ok':True,'results':[{'ok':True,'text':p['tag']} for p in json.loads(line)['requests']]}),flush=True)")
        with tempfile.TemporaryDirectory() as td:
            worker = Worker(sys.executable, "llm", "unused", Path(td) / "log.txt")
            worker.command = [sys.executable, "-u", "-c", code]
            try:
                worker.start(time.monotonic() + 10)
                rows = worker.request_batch("generate", [{"tag": "A"}, {"tag": "B"}], time.monotonic() + 10)
                self.assertEqual([row["text"] for row in rows], ["A", "B"])
            finally:
                worker.close()

    def test_complete_run_and_resume_do_not_recreate_sessions(self):
        initialized = []
        class Package:
            @staticmethod
            def ToolEnv(split, sid, root):
                initialized.append(sid)
                return FakeEnv()
            @staticmethod
            def schema_validate(calls, catalog):
                return [{"ok": True} for c in calls]
        class ModelWorker:
            def __init__(self, python, kind, *args):
                self.kind = kind
                self.ready = {"ready": True, "parameters": 1, "revision": "test"}
            def start(self, deadline):
                pass
            def request(self, operation, payload, deadline):
                return {"texts": ["history", "current"]} if self.kind == "asr" else {"text": action([])}
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as td, patch("ftkit.submission.runtime", return_value=Package), patch("ftkit.submission.Worker", ModelWorker):
            root = Path(td)
            args = self.fixture(root)
            run(args)
            rows = read_jsonl(root / "run/submission.jsonl")
            self.assertEqual(len(rows), 2)
            self.assertEqual(initialized, ["synthetic_1", "synthetic_2"])
            args.resume = True
            run(args)
            self.assertEqual(initialized, ["synthetic_1", "synthetic_2"])
            write_json(root / "run/inflight.json", {"id": "synthetic_3"})
            with self.assertRaisesRegex(ValueError, "unfinished"):
                run(args)

    def test_real_worker_transport_without_models(self):
        code = "import sys,json; print(json.dumps({'ready':True}),flush=True);\nfor line in sys.stdin: print(json.dumps({'ok':True,'text':'response'}),flush=True)"
        with tempfile.TemporaryDirectory() as td:
            worker = Worker(sys.executable, "llm", "unused", Path(td) / "log.txt")
            worker.command = [sys.executable, "-u", "-c", code]
            try:
                worker.start(time.monotonic() + 10)
                self.assertEqual(worker.request("generate", {}, time.monotonic() + 10)["text"], "response")
            finally:
                worker.close()

    def test_preview_is_not_named_submission(self):
        class Package:
            @staticmethod
            def ToolEnv(split, sid, root):
                return FakeEnv()
            @staticmethod
            def schema_validate(calls, catalog):
                return [{"ok": True} for c in calls]
        class ModelWorker:
            def __init__(self, python, kind, *args):
                self.kind = kind
                self.ready = {"ready": True, "parameters": 1, "revision": "test"}
            def start(self, deadline):
                pass
            def request(self, operation, payload, deadline):
                return {"texts": ["history", "current"]} if self.kind == "asr" else {"text": action([])}
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as td, patch("ftkit.submission.runtime", return_value=Package), patch("ftkit.submission.Worker", ModelWorker):
            root = Path(td)
            args = self.fixture(root)
            args.limit = 1
            run(args)
            self.assertFalse((root / "run/submission.jsonl").exists())
            self.assertEqual(len(read_jsonl(root / "run/preview.jsonl")), 1)
            self.assertFalse(json.loads((root / "run/summary.json").read_text())["complete_split"])

    def test_hard_worker_timeout(self):
        code = "import sys,json,time; print(json.dumps({'ready':True}),flush=True);\nfor line in sys.stdin: time.sleep(30)"
        with tempfile.TemporaryDirectory() as td:
            worker = Worker(sys.executable, "llm", "unused", Path(td) / "log.txt")
            worker.command = [sys.executable, "-u", "-c", code]
            try:
                worker.start(time.monotonic() + 10)
                with self.assertRaises(TimeoutError):
                    worker.request("generate", {}, time.monotonic() + 0.1)
                self.assertIsNone(worker.process)
            finally:
                worker.close()


if __name__ == "__main__":
    unittest.main()
