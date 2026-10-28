"""Adapter contract tests with a streamed fake engine; real CUDA checks live in gpu_validate.py."""
import asyncio
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch

from ftkit.backends.vllm import Engine


class FakeNative:
    def __init__(self):
        self.aborted = []
        self.requests = []
        self.slow = False
    async def generate(self, prompt, params, rid, **kwargs):
        self.requests.append((prompt, params, rid, kwargs))
        if self.slow:
            await asyncio.sleep(10)
        for tokens, finished in (([1], None), ([1, 2], "stop")):
            await asyncio.sleep(0)
            yield types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=tokens, text='{"tool_calls":[]}', finish_reason=finished)],
                                        num_cached_tokens=2)
    async def abort(self, rid):
        self.aborted.append(rid)
    def shutdown(self):
        pass


class VllmTests(unittest.TestCase):
    def test_exact_prompt_ids_settings_streaming_metrics_and_timeout(self):
        native = FakeNative()
        class Args:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
        class AsyncEngine:
            @staticmethod
            def from_engine_args(args):
                native.args = args.kwargs
                return native
        tokenizer = types.SimpleNamespace(chat_template="original", apply_chat_template=lambda *a, **kw: [7, 8, 9])
        modules = {"vllm": types.SimpleNamespace(AsyncEngineArgs=Args, AsyncLLMEngine=AsyncEngine, SamplingParams=Args),
                   "vllm.lora.request": types.SimpleNamespace(LoRARequest=Args),
                   "vllm.sampling_params": types.SimpleNamespace(StructuredOutputsParams=Args),
                   "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(mem_get_info=lambda: (100, 200)))}
        config = {"model": "unused", "training": {"max_seq_length": 20, "generation_max_new_tokens": 4},
                  "tokenizer": {"model": "custom-tokenizer", "revision": "tokenizer-revision"},
                  "inference": {"max_num_seqs": 4}}
        with patch.dict(sys.modules, modules), patch("ftkit.backends.vllm.model_snapshot", return_value=(Path.cwd(), "abc")), \
                patch("ftkit.backends.vllm.parameter_count", return_value=123), patch("ftkit.backends.vllm.tokenizer_for", return_value=tokenizer):
            engine = Engine(config)
            self.assertEqual(native.args["max_num_seqs"], 4)
            self.assertEqual(native.args["tokenizer"], "custom-tokenizer")
            self.assertEqual(native.args["tokenizer_revision"], "tokenizer-revision")
            self.assertTrue(native.args["enable_prefix_caching"])
            self.assertTrue(native.args["enable_chunked_prefill"])
            self.assertEqual(native.args["dtype"], "bfloat16")
            result = asyncio.run(engine.generate("r1", {"messages": [], "tools": []}, time.monotonic() + 3))
            self.assertEqual(native.requests[0][0], {"prompt_token_ids": [7, 8, 9]})
            self.assertEqual(result["output_tokens"], 2)
            self.assertEqual(result["cached_tokens"], 2)
            self.assertEqual(result["finish_reason"], "stop")
            self.assertGreaterEqual(result["decode_seconds"], 0)
            self.assertEqual(result["gpu_memory_used_bytes"], 100)
            native.slow = True
            with self.assertRaises(TimeoutError):
                asyncio.run(engine.generate("r2", {"input_ids": [1], "tools": []}, time.monotonic() + 0.02))
            self.assertEqual(native.aborted, ["r2"])
            with self.assertRaisesRegex(ValueError, "overflow"):
                asyncio.run(engine.generate("r3", {"input_ids": list(range(18)), "tools": []}, time.monotonic() + 1))
            self.assertEqual(len(native.requests), 2)


if __name__ == "__main__":
    unittest.main()
