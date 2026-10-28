import sys
import types
import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from ftkit.backends.llm import Collator, load, export


class UnslothTests(unittest.TestCase):
    def test_language_only_loader_and_qlora_selection(self):
        class Config:
            model_type = "qwen3_vl"
        class Model:
            config = types.SimpleNamespace(use_cache=True)
            language_config = types.SimpleNamespace(use_cache=True)
            def modules(self):
                return [self, types.SimpleNamespace(config=self.language_config)]
            def named_modules(self):
                return [("model.language_model.layers.0.q_proj", None), ("visual.layers.0.q_proj", None)]
            def named_parameters(self):
                return [("visual.weight", types.SimpleNamespace(requires_grad=False))]
        class Loader:
            @classmethod
            def from_pretrained(cls, **kwargs):
                cls.loaded = kwargs
                return Model(), object()
            @classmethod
            def get_peft_model(cls, model, **kwargs):
                cls.lora = kwargs
                return model
            @classmethod
            def for_training(cls, model, **kwargs):
                cls.training_mode = kwargs
        modules = {"unsloth": types.SimpleNamespace(FastLanguageModel=Loader, FastVisionModel=Loader),
                   "torch": types.SimpleNamespace(),
                   "transformers": types.SimpleNamespace(AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **kw: Config()),
                                                          AutoModelForCausalLM=types.SimpleNamespace(_model_mapping={}))}
        cfg = {"model": "test", "architecture": {"handler": "auto"}, "devices": 1,
               "training": {"max_seq_length": 8192, "quantize": False, "lora_rank": 16, "lora_alpha": 32,
                            "lora_dropout": .05, "gradient_checkpointing": True, "seed": 42}}
        tokenizer = types.SimpleNamespace(padding_side="left")
        with patch.dict(sys.modules, modules), patch("ftkit.backends.llm.model_snapshot", return_value=(Path.cwd(), "revision")), \
                patch("ftkit.backends.llm.gpu_precision", return_value="bf16"), \
                patch("ftkit.llm_common.tokenizer_for", return_value=tokenizer):
            model, returned = load(cfg, train=True)
            self.assertFalse(Loader.loaded["load_in_4bit"])
            self.assertFalse(Loader.lora["finetune_vision_layers"])
            self.assertEqual(Loader.lora["target_modules"], ["model.language_model.layers.0.q_proj"])
            self.assertEqual(Loader.lora["lora_alpha"], 32)
            self.assertFalse(model.config.use_cache)
            self.assertFalse(model.language_config.use_cache)
            self.assertEqual(Loader.training_mode, {"use_gradient_checkpointing": True})
            self.assertIs(returned, tokenizer)
            cfg["training"]["quantize"] = True
            load(cfg, train=True)
            self.assertTrue(Loader.loaded["load_in_4bit"])
            load(cfg, merge_export=True)
            self.assertFalse(Loader.loaded["load_in_4bit"])
            cfg["training"]["gradient_checkpointing"] = False
            Model.language_config.use_cache = True
            model, _ = load(cfg, train=True)
            self.assertFalse(model.language_config.use_cache)
            self.assertEqual(Loader.training_mode, {"use_gradient_checkpointing": False})

    def test_collator_preserves_existing_loss_masks_and_dynamic_padding(self):
        rows = [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
                {"input_ids": [3], "attention_mask": [1], "labels": [3]}]
        fake_torch = types.SimpleNamespace(tensor=lambda value, **kwargs: value, long=object())
        with patch.dict(sys.modules, {"torch": fake_torch}):
            result = Collator(9)(rows)
        self.assertEqual(result["labels"], [[-100, 2], [3, -100]])
        self.assertEqual(result["input_ids"], [[1, 2], [3, 9]])
        self.assertEqual(result["attention_mask"], [[1, 1], [1, 0]])

    def test_adapter_export_is_portable_and_does_not_load_weights(self):
        with tempfile.TemporaryDirectory() as td:
            source, target = Path(td) / "source", Path(td) / "export"
            source.mkdir()
            (source / "adapter_config.json").write_text('{"r":16}')
            (source / "adapter_model.safetensors").write_bytes(b"synthetic adapter")
            tokenizer = types.SimpleNamespace(save_pretrained=lambda path: (path / "tokenizer_config.json").write_text('{}'))
            modules = {"unsloth": types.SimpleNamespace(), "transformers": types.SimpleNamespace(
                AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda path: tokenizer))}
            args = types.SimpleNamespace(output=target, checkpoint=source, merge=False)
            with patch.dict(sys.modules, modules), patch("ftkit.backends.llm.load", side_effect=AssertionError("must not load weights")):
                export(args, {"backend": "unsloth_sft"})
            self.assertEqual((target / "adapter_model.safetensors").read_bytes(), b"synthetic adapter")
            self.assertEqual(json.loads((target / "deployment.json").read_text())["format"], "peft_adapter")


if __name__ == "__main__":
    unittest.main()
