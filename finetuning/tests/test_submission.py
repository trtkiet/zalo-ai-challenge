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
from ftkit.submission import (Worker, audio_path, infer_sample, parse_response, run,
                              validate_records, worker_python)


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
    def fixture(self, root):
        split = root / "data/public_test"
        split.mkdir(parents=True)
        records = [item("synthetic_1"), item("synthetic_2")]
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
