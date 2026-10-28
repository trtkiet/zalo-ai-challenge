"""Exercise the real concurrent JSON-line worker without loading CUDA libraries."""
import asyncio
import io
import json
import queue
import time
import unittest
from unittest.mock import patch

from ftkit.llm_worker import serve


class WorkerProtocolTests(unittest.TestCase):
    def test_concurrent_completion_cancellation_and_unknown_cancel_acknowledged(self):
        async def scenario():
            inputs, protocol = queue.Queue(), io.StringIO()
            slow_started = asyncio.Event()
            instances = []

            class Engine:
                metadata = {"ready": True, "backend": "fake-vllm"}
                def __init__(self, *args):
                    self.closed = False
                    instances.append(self)
                async def generate(self, rid, request, deadline):
                    if rid == "slow":
                        slow_started.set()
                        await asyncio.sleep(10)
                    return {"text": '{"tool_calls":[]}'}
                def close(self):
                    self.closed = True

            async def warm(engine):
                pass

            def messages():
                return [json.loads(line) for line in protocol.getvalue().splitlines()]

            async def wait_for(predicate):
                async with asyncio.timeout(3):
                    while not predicate():
                        await asyncio.sleep(0.001)

            async def feed():
                try:
                    await wait_for(lambda: any(m.get("ready") for m in messages()))
                    for rid in ("slow", "fast"):
                        inputs.put(json.dumps({"operation": "generate", "request_id": rid,
                                              "deadline": time.monotonic() + 5}) + "\n")
                    await slow_started.wait()
                    await wait_for(lambda: any(m.get("request_id") == "fast" for m in messages()))
                    inputs.put('{"operation":"cancel","request_id":"slow"}\n')
                    inputs.put('{"operation":"cancel","request_id":"already_finished"}\n')
                    await wait_for(lambda: {"slow", "fast", "already_finished"}.issubset(
                        {m.get("request_id") for m in messages()}))
                finally:
                    inputs.put('{"operation":"shutdown"}\n')
                    inputs.put("")  # Unblock the stdin thread even if a test assertion fails.

            with patch("ftkit.backends.vllm.Engine", Engine), patch("ftkit.backends.vllm.warm", warm), \
                    patch("sys.stdout", protocol), patch("sys.stdin", type("Input", (), {"readline": lambda self: inputs.get()})()):
                await asyncio.wait_for(asyncio.gather(serve({}), feed()), timeout=5)
            values = messages()
            responses = [m for m in values if "request_id" in m]
            self.assertEqual(responses[0]["request_id"], "fast")
            self.assertTrue(responses[0]["ok"])
            self.assertEqual(sum(m["request_id"] == "slow" for m in responses), 1)
            self.assertEqual(next(m for m in responses if m["request_id"] == "slow")["error_type"], "TimeoutError")
            self.assertEqual(next(m for m in responses if m["request_id"] == "already_finished")["error_type"], "TimeoutError")
            self.assertTrue(instances[0].closed)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
