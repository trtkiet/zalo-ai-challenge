import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
import wave

from ftkit.common import ROOT, fresh_directory, normalize, resolve_config, write_json
from ftkit.data import (action, encode_trajectory, generation_cases, grouped_split, prepare,
                       safe_audio, trajectory, validate_action)
from ftkit.metrics import error_rates
from ftkit.backends.asr import check_weights, duration_batches
from ftkit.training import validate_existing_output


TOOLS = [
    {"name": "contacts.search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "sms.send", "parameters": {"type": "object", "properties": {"contact_id": {"type": "string"}, "content": {"type": "string"}}, "required": ["contact_id", "content"]}},
    {"name": "sys.ask_missing_info", "parameters": {"type": "object", "properties": {"tool_name": {"type": "string"}, "params": {"type": "array", "items": {"type": "string"}}}, "required": ["tool_name", "params"]}}
]


def record():
    return {"id": "train_1", "context": {"installed_apps": {"sms": {"tools": ["sms.send"]}}},
            "history": [{"role": "user", "audio_text": "Gửi cho mẹ", "audio": "unused.wav"},
                        {"role": "assistant", "text": "Nội dung gì?"}],
            "request": "Nhắn mẹ là con về muộn", "tools": TOOLS,
            "calls": [{"name": "contacts.search", "arguments": {"query": "mẹ"}},
                      {"name": "sms.send", "arguments": {"contact_id": "c_SECRET_RESULT", "content": "con về muộn"}}],
            "results": [{"contacts": [{"id": "c_SECRET_RESULT"}]}, {"ok": True}]}


class CharTokenizer:
    """A prefix-stable chat template; target IDs are inspectable without GPU packages."""
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>" + m["content"] + f"</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(map(ord, text)) if tokenize else text


class DataTests(unittest.TestCase):
    def test_normalization_preserves_vietnamese(self):
        self.assertEqual(normalize("  Chuyển: MẸ 500 nghìn! "), "chuyển mẹ 500 nghìn")

    def test_split_keeps_shared_history_together(self):
        samples = [{"id": f"s{i}", "audio": f"audio/s{i}.wav", "audio_text": text, "history": []}
                   for i, text in enumerate(["Nhắc uống thuốc", "NHẮC UỐNG THUỐC!", "Mở nhạc", "Xem thời tiết", "Tắt đèn"])]
        samples[2]["history"] = [{"role": "user", "audio": "audio/h.wav", "audio_text": "Nhắc uống thuốc"}]
        split = grouped_split(samples, 0.4)
        self.assertEqual(split, grouped_split(list(reversed(samples)), 0.4))
        location = {sid: name for name, ids in split.items() for sid in ids}
        self.assertEqual(location["s0"], location["s1"])
        self.assertEqual(location["s0"], location["s2"])
        self.assertEqual(set(location), {f"s{i}" for i in range(5)})

    def test_paths_reject_escape_and_environment(self):
        with tempfile.TemporaryDirectory() as td:
            for path in ("../train_environment/x/state.json", "audio/../../state.json", "/tmp/file.wav"):
                with self.assertRaises(ValueError):
                    safe_audio(td, path)
            self.assertTrue(safe_audio(td, "audio/x/x.wav").is_relative_to(Path(td).resolve()))

    def test_chronological_results_and_masks(self):
        item = record()
        messages, targets = trajectory(item)
        self.assertNotIn("c_SECRET_RESULT", json.dumps(messages[:targets[0]], ensure_ascii=False))
        self.assertIn("c_SECRET_RESULT", json.dumps(messages[:targets[1]], ensure_ascii=False))
        encoded = encode_trajectory(CharTokenizer(), messages, targets, 100000)
        supervised = "".join(chr(x) for x in encoded["labels"] if x != -100)
        self.assertNotIn("Nội dung gì?", supervised)
        self.assertNotIn("TOOLS", supervised)
        self.assertNotIn('"result"', supervised)
        self.assertIn(action(item["calls"][:1]), supervised)
        self.assertIn(action([]), supervised)

    def test_no_action_and_clarification(self):
        item = record()
        item["calls"], item["results"] = [], []
        messages, targets = trajectory(item)
        self.assertEqual(len(targets), 1)
        self.assertEqual(messages[-1]["content"], action([]))
        item["calls"] = [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "sms.send", "params": ["content"]}}]
        item["results"] = [{"ok": True}]
        messages, targets = trajectory(item)
        self.assertEqual(len(targets), 2)

    def test_overlength_is_excluded_not_truncated(self):
        messages, targets = trajectory(record())
        self.assertIsNone(encode_trajectory(CharTokenizer(), messages, targets, 20))
        self.assertEqual(list(generation_cases(CharTokenizer(), record(), 20, 5)), [])

    def test_unsafe_chat_template_fails(self):
        class Rewriter(CharTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return [len(messages)] + super().apply_chat_template(messages, **kwargs)
        messages, targets = trajectory(record())
        with self.assertRaises(ValueError):
            encode_trajectory(Rewriter(), messages, targets, 100000)

    def test_nested_and_schema_checks(self):
        tools = [{"name": "food.order", "parameters": {"type": "object", "properties": {"items": {"type": "array", "items": {
            "type": "object", "properties": {"quantity": {"type": "integer", "minimum": 1}}, "required": ["quantity"]}}}, "required": ["items"]}}]
        value = {"tool_calls": [{"name": "food.order", "arguments": {"items": [{"quantity": 2}]}}]}
        self.assertTrue(validate_action(value, tools))
        value["tool_calls"][0]["arguments"]["items"][0]["quantity"] = True
        self.assertFalse(validate_action(value, tools))
        self.assertFalse(validate_action({"tool_calls": [{"name": "not.installed", "arguments": {}}]}, tools))
        self.assertFalse(validate_action({"tool_calls": [{"name": [], "arguments": {}}]}, tools))
        self.assertTrue(validate_action({"tool_calls": []}, tools))

    def test_preparation_without_environment_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train = root / "data/train"
            train.mkdir(parents=True)
            samples = []
            for i in range(10):
                relative = f"audio/train_{i}/train_{i}.wav"
                path = train / relative
                path.parent.mkdir(parents=True)
                with wave.open(str(path), "wb") as wav:
                    wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    wav.writeframes(b"\0\0" * 160)
                samples.append({"id": f"train_{i}", "audio": relative, "audio_text": f"Lệnh số {i}", "history": [],
                                "context": {"installed_apps": {"sms": {"tools": ["sms.send"]}}},
                                "label": {"tool_calls": [], "tool_results": [], "final_state": {"secret": "NEVER_COPY"}}})
            (train / "train.jsonl").write_text("\n".join(json.dumps(s) for s in samples), encoding="utf-8")
            write_json(train / "train_tools_catalog.json", {"apps": [{"tools": TOOLS}]})
            metadata = prepare(root / "data", root / "prepared")
            self.assertEqual(metadata["counts"]["train"]["conversations"], 9)
            content = (root / "prepared/llm_train.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("NEVER_COPY", content)
            with self.assertRaises(ValueError):
                prepare(root / "data", root / "prepared")


class ConfigTests(unittest.TestCase):
    def test_local_export_configuration(self):
        with tempfile.TemporaryDirectory() as td:
            write_json(Path(td) / "deployment.json", {"config": {"backend": "zipformer_rnnt", "model": "original",
                       "revision": "pinned", "tokenizer": {}, "architecture": {"decoder_dim": 256}, "training": {"epochs": 4}}})
            args = types.SimpleNamespace(config=None, profile="gpu16", model=td, revision=None, tokenizer=None, set=[])
            config = resolve_config(args, "asr")
            self.assertEqual(config["architecture"]["decoder_dim"], 256)
            self.assertEqual(config["model"], td)

    def test_resume_rejects_changed_dataset(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "run"
            checkpoint = output / "checkpoint"
            checkpoint.mkdir(parents=True)
            prepared = root / "prepared"
            prepared.mkdir()
            config = {"model": "m", "revision": "r", "backend": "hf_sft", "architecture": {}, "training": {}}
            write_json(output / "run.json", {"config": config, "dataset": {"fingerprint": "old"}})
            write_json(prepared / "metadata.json", {"fingerprint": "new"})
            args = types.SimpleNamespace(output=output, resume=checkpoint, prepared=prepared)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                validate_existing_output(args, config)

    def test_model_revision_and_cli_precedence(self):
        args = types.SimpleNamespace(config=None, profile="gpu16", model="another/qwen", revision=None,
                                     tokenizer=None, set=["training.epochs=2", "training.max_seq_length=3072"])
        config = resolve_config(args, "llm")
        self.assertEqual(config["revision"], "main")
        self.assertEqual(config["model"], "another/qwen")
        self.assertEqual(config["training"]["epochs"], 2)
        self.assertEqual(config["training"]["max_seq_length"], 3072)
        args.revision = "abc123"
        self.assertEqual(resolve_config(args, "llm")["revision"], "abc123")

    def test_outputs_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            fresh_directory(Path(td) / "run")
            (Path(td) / "run/file").write_text("existing")
            with self.assertRaises(ValueError):
                fresh_directory(Path(td) / "run")


class ASRTests(unittest.TestCase):
    def test_weight_guard(self):
        tensor = lambda shape: types.SimpleNamespace(shape=shape)
        model = {"encoder.weight": tensor((2, 2)), "simple_am_proj.weight": tensor((2, 2))}
        weights = {"encoder.weight": tensor((2, 2))}
        self.assertEqual(check_weights(model, weights), ["simple_am_proj.weight"])
        with self.assertRaises(ValueError):
            check_weights(model, {})
        with self.assertRaises(ValueError):
            check_weights(model, {"encoder.weight": tensor((3, 2))})

    def test_duration_buckets_and_distributed_lengths(self):
        rows = [{"duration": d} for d in [1, 5, 2, 7, 4, 3, 9]]
        batches = duration_batches(rows, 10)
        self.assertEqual(sorted(i for b in batches for i in b), list(range(len(rows))))
        self.assertTrue(all(sum(rows[i]["duration"] for i in b) <= 10 for b in batches))
        self.assertEqual(len(duration_batches(rows, 10, world=2, rank=0)), len(duration_batches(rows, 10, world=2, rank=1)))
        with self.assertRaises(ValueError):
            duration_batches(rows, 2)

    def test_corpus_rates(self):
        rates = error_rates([("mở đèn phòng khách", "mở đèn"), ("xin chào", "xin chào")])
        self.assertAlmostEqual(rates["wer"], 2 / 6)


if __name__ == "__main__":
    unittest.main()
