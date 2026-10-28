import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from ftkit.common import read_jsonl, write_json
from ftkit.continuous import AsyncWorker, continuous_samples
from ftkit.data import action, validate_action
from ftkit.llm_common import action_schema, inference_settings, llm_identity
from ftkit.submission import infer_sample, parse_response, run
from test_submission import FakeASR, FakeEnv, FakeLLM, TOOLS, item
import test_submission as fixtures


class ScheduledWorker:
    def __init__(self, kind, scripts=None, interrupt=False):
        self.kind, self.scripts = kind, copy.deepcopy(scripts or {})
        self.ready = {"ready": True, "parameters": 1, "revision": "test"}
        self.jobs, self.cancelled, self.requests = [], [], []
        self.counter = 0
        self.interrupt = interrupt
    def start(self, deadline):
        pass
    def submit(self, operation, payload, deadline):
        self.counter += 1
        rid = f"{self.kind}-{self.counter}"
        self.requests.append((rid, copy.deepcopy(payload), deadline))
        if self.kind == "asr":
            result = {"ok": True, "results": [{"ok": True, "texts": [Path(p).parent.name for p in req["audio"]]}
                                                for req in payload["requests"]]}
            delay = 0
        else:
            content = json.dumps(payload["messages"])
            sid = next((sid for sid in self.scripts if sid in content), None)
            delay, result = self.scripts[sid].pop(0) if sid else (0, {"text": action([])})
            result = {"ok": True, **result}
        self.jobs.append((time.monotonic() + delay, {"request_id": rid, **result}))
        return rid
    def poll(self):
        if self.interrupt and self.jobs:
            self.interrupt = False
            raise KeyboardInterrupt
        ready = [v for due, v in self.jobs if due <= time.monotonic()]
        self.jobs = [(due, v) for due, v in self.jobs if due > time.monotonic()]
        return ready
    def cancel(self, rid):
        self.cancelled.append(rid)
        # Intentionally keep the late response to exercise cancellation races.
    def close(self):
        self.jobs.clear()


def package():
    class Package:
        initialized = []
        @classmethod
        def ToolEnv(cls, split, sid, root):
            if sid in cls.initialized:
                raise AssertionError("ToolEnv recreated")
            cls.initialized.append(sid)
            return FakeEnv()
        @staticmethod
        def schema_validate(calls, catalog):
            return [{"ok": True} for _ in calls]
    return Package


class ContinuousTests(unittest.TestCase):
    def test_fast_sample_releases_slot_and_completion_is_not_input_order(self):
        pkg = package()
        workers = {"asr": ScheduledWorker("asr"), "llm": ScheduledWorker("llm", {
            "synthetic_1": [(0.18, {"text": action([])})],
            "synthetic_2": [(0.01, {"text": action([])})],
            "synthetic_3": [(0.01, {"text": action([])})]})}
        rows = list(continuous_samples([item(f"synthetic_{i}") for i in (1, 2, 3)], TOOLS, pkg, workers,
                                       Path.cwd(), "public_test", capacity=2))
        self.assertEqual([r[0]["id"] for r in rows], ["synthetic_2", "synthetic_3", "synthetic_1"])
        self.assertEqual(pkg.initialized, ["synthetic_1", "synthetic_2", "synthetic_3"])
        self.assertTrue(all(r[1]["status"] == "stopped" for r in rows))
        self.assertEqual(len({r[0] for r in workers["llm"].requests}), 3)

    def test_one_deadline_cancels_only_its_request(self):
        pkg = package()
        workers = {"asr": ScheduledWorker("asr"), "llm": ScheduledWorker("llm", {
            "synthetic_1": [(0.3, {"text": action([])})],
            "synthetic_2": [(0.01, {"text": action([])})],
            "synthetic_3": [(0.01, {"text": action([])})]})}
        rows = list(continuous_samples([item(f"synthetic_{i}") for i in (1, 2, 3)], TOOLS, pkg, workers,
                                       Path.cwd(), "public_test", capacity=2, max_seconds=0.12))
        self.assertEqual({r[0]["id"]: r[1]["status"] for r in rows},
                         {"synthetic_1": "timeout", "synthetic_2": "stopped", "synthetic_3": "stopped"})
        self.assertEqual(workers["llm"].cancelled, ["llm-1"])

    def test_graceful_interrupt_finalizes_active_only(self):
        pkg = package()
        workers = {"asr": ScheduledWorker("asr"), "llm": ScheduledWorker("llm", interrupt=True)}
        rows = list(continuous_samples([item(f"synthetic_{i}") for i in (1, 2, 3)], TOOLS, pkg, workers,
                                       Path.cwd(), "public_test", capacity=2))
        self.assertEqual(len(rows), 2)
        self.assertEqual([r[1]["status"] for r in rows], ["interrupted", "interrupted"])
        self.assertEqual(len(pkg.initialized), 2)

    def test_empty_input_and_invalid_capacity(self):
        self.assertEqual(list(continuous_samples([], TOOLS, package(), {}, Path.cwd(), "public_test")), [])
        with self.assertRaises(ValueError):
            list(continuous_samples([], TOOLS, package(), {}, Path.cwd(), "public_test", capacity=0))

    def test_journal_restores_nonprefix_results_and_input_order(self):
        pkg = package()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = fixtures.OrchestratorTests().fixture(root, count=3)
            args.batch_size = 2
            def factory(py, kind, *unused):
                return ScheduledWorker(kind, {"synthetic_1": [(0.10, {"text": action([])})]})
            with patch("ftkit.submission.runtime", return_value=pkg), patch("ftkit.continuous.AsyncWorker", factory):
                run(args)
                rows = read_jsonl(root / "run/submission.jsonl")
                self.assertEqual([r["id"] for r in rows], ["synthetic_1", "synthetic_2", "synthetic_3"])
                journal = read_jsonl(root / "run/completed.jsonl")
                self.assertEqual(journal[0]["result"]["id"], "synthetic_2")
                # Simulate a crash after durable completion but before marker cleanup.
                write_json(root / "run/inflight.json", {"ids": ["synthetic_2"]})
                args.resume = True
                run(args)
                self.assertEqual(len(pkg.initialized), 3)
                write_json(root / "run/inflight.json", {"ids": ["unknown_inflight"]})
                with self.assertRaisesRegex(ValueError, "unfinished"):
                    run(args)


class MultiCallTests(unittest.TestCase):
    def test_independent_actions_execute_in_order_without_repair(self):
        calls = [{"name": "contacts.search", "arguments": {"query": "mẹ"}},
                 {"name": "contacts.search", "arguments": {"query": "Hương"}}]
        env = FakeEnv()
        result, trace = infer_sample(item(), TOOLS, env, FakeASR(), FakeLLM([action(calls), action([])]), Path.cwd(), "public_test")
        self.assertEqual(result["tool_calls"], calls)
        self.assertEqual(len(trace["steps"][0]["executed"]), 2)
        self.assertTrue(validate_action({"tool_calls": calls}, list(TOOLS.values())))
        self.assertEqual(parse_response(action(calls)), calls)

    def test_prevalidate_all_before_any_side_effect(self):
        calls = [{"name": "contacts.search", "arguments": {"query": "mẹ"}},
                 {"name": "sms.send", "arguments": {}}]
        result, trace = infer_sample(item(), TOOLS, FakeEnv(), FakeASR(), FakeLLM([action(calls)]), Path.cwd(),
                                     "public_test", max_repairs=0)
        self.assertEqual(result["tool_calls"], [])
        self.assertEqual(trace["status"], "invalid_response")

    def test_failure_discards_suffix_and_history_contains_only_executed_prefix(self):
        class NoChange(FakeEnv):
            def call(self, name, arguments):
                super().call(name, arguments)
                return {"ok": True, "done": False}
        first = {"name": "contacts.search", "arguments": {"query": "mẹ"}}
        suffix = {"name": "contacts.search", "arguments": {"query": "SHOULD_NOT_EXECUTE"}}
        llm = FakeLLM([action([first, suffix]), action([])])
        result, trace = infer_sample(item(), TOOLS, NoChange(), FakeASR(), llm, Path.cwd(), "public_test")
        self.assertEqual(result["tool_calls"], [first])
        self.assertEqual(trace["steps"][0]["discarded_calls"], 1)
        self.assertNotIn("SHOULD_NOT_EXECUTE", json.dumps(llm.prompts[1]))

    def test_budget_applies_between_calls(self):
        call = {"name": "contacts.search", "arguments": {"query": "mẹ"}}
        result, trace = infer_sample(item(), TOOLS, FakeEnv(), FakeASR(), FakeLLM([action([call, call])]), Path.cwd(),
                                     "public_test", max_tool_calls=1)
        self.assertEqual(result["tool_calls"], [call])
        self.assertEqual(trace["status"], "tool_limit")


class SchemaTests(unittest.TestCase):
    def test_names_groups_nested_types_and_zero_budget(self):
        schema = action_schema([{"name": "contacts.update", "parameters": {
            "type": "object", "properties": {"phone": {"type": "string", "x-ref": "contacts"}, "email": {"type": "string"}},
            "x-one-of": [["phone"], ["email"]]}}], 3)
        args = schema["properties"]["tool_calls"]["items"]["anyOf"][0]["properties"]["arguments"]
        self.assertEqual(args["allOf"], [{"anyOf": [{"required": ["phone"]}, {"required": ["email"]}]}])
        self.assertNotIn("x-ref", json.dumps(schema))
        self.assertEqual(action_schema([], 0)["properties"]["tool_calls"]["maxItems"], 0)

    def test_inference_precision_is_independent_of_qlora(self):
        cfg = {"training": {"quantize": True, "max_seq_length": 8192}}
        self.assertEqual(inference_settings(cfg)["precision"], "bf16")
        self.assertEqual(llm_identity()["protocol"], 2)

    def test_argument_names_matching_schema_annotations_survive(self):
        schema = action_schema([{"name": "calendar.create", "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "description": "Event title"},
            "description": {"type": "string"}}, "required": ["title"]}}])
        args = schema["properties"]["tool_calls"]["items"]["anyOf"][0]["properties"]["arguments"]
        self.assertEqual(set(args["properties"]), {"title", "description"})


class TransportTests(unittest.TestCase):
    def test_stalled_worker_stdin_does_not_block_submission_or_cancellation(self):
        code = "import json,time; print(json.dumps({'ready':True}),flush=True); time.sleep(20)"
        with tempfile.TemporaryDirectory() as td:
            worker = AsyncWorker(sys.executable, "llm", "unused", Path(td) / "log")
            worker.command = [sys.executable, "-u", "-c", code, "--kind", "llm"]
            try:
                worker.start(time.monotonic() + 5)
                started = time.monotonic()
                rid = worker.submit("generate", {"text": "x" * 1_000_000}, time.monotonic() + 5)
                worker.cancel(rid)
                self.assertLess(time.monotonic() - started, 1)
                # A stalled native reader cannot acknowledge cancellation: reap the group.
                worker.cancelled_at[rid] = time.monotonic() - 3
                worker.poll()
                self.assertIsNone(worker.process)
            finally:
                worker.close()

    def test_correlated_out_of_order_and_cancelled_response_ignored(self):
        code = """import sys,json
print(json.dumps({'ready':True}),flush=True)
requests=[]
for line in sys.stdin:
 r=json.loads(line)
 if r['operation']=='cancel': continue
 requests.append(r)
 if len(requests)==2:
  for r in reversed(requests): print(json.dumps({'request_id':r['request_id'],'ok':True,'text':r['tag']}),flush=True)
  requests=[]
"""
        with tempfile.TemporaryDirectory() as td:
            worker = AsyncWorker(sys.executable, "llm", "unused", Path(td) / "log")
            worker.command = [sys.executable, "-u", "-c", code, "--kind", "llm"]
            try:
                worker.start(time.monotonic() + 5)
                first = worker.submit("generate", {"tag": "A"}, time.monotonic() + 5)
                second = worker.submit("generate", {"tag": "B"}, time.monotonic() + 5)
                values = []
                deadline = time.monotonic() + 5
                while len(values) < 2 and time.monotonic() < deadline:
                    values.extend(worker.poll())
                    time.sleep(0.005)
                self.assertEqual([r["request_id"] for r in values], [second, first])
                third = worker.submit("generate", {"tag": "C"}, deadline)
                fourth = worker.submit("generate", {"tag": "D"}, deadline)
                worker.cancel(third)
                values = []
                while not values and time.monotonic() < deadline:
                    values.extend(worker.poll())
                    time.sleep(0.005)
                self.assertEqual([r["request_id"] for r in values], [fourth])
                self.assertIsNone(worker.process.poll())
            finally:
                worker.close()


if __name__ == "__main__":
    unittest.main()
