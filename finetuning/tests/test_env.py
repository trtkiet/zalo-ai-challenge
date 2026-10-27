import contextlib
import io
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

from cli import parser
from ftkit.env import find_env_file, load_env_file, main


class EnvTests(unittest.TestCase):
    def test_loads_parent_env_expands_paths_and_preserves_shell_overrides(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {"HOME": "/home/test user", "HF_HOME": "/override"}, clear=True):
            root = Path(td) / "finetuning"
            root.mkdir()
            (root.parent / ".ENV").write_text(
                'RESULT="$HOME/project/results"\n'
                'export FINETUNE_HOME="${HOME}/project/runtime"\n'
                'HF_HOME="$FINETUNE_HOME/cache"\n'
                'DATA_ROOT="$HOME/project/data"  # dataset folder\n'
                'CACHE_FROM_OVERRIDE="${HF_HOME}/models"\n'
                "LITERAL='$HOME/literal#value'\n", encoding="utf-8")
            added = load_env_file(root)
            self.assertEqual(os.environ["RESULT"], "/home/test user/project/results")
            self.assertEqual(os.environ["FINETUNE_HOME"], "/home/test user/project/runtime")
            self.assertEqual(os.environ["DATA_ROOT"], "/home/test user/project/data")
            self.assertEqual(os.environ["HF_HOME"], "/override")
            self.assertNotIn("HF_HOME", added)
            self.assertEqual(os.environ["CACHE_FROM_OVERRIDE"], "/override/models")
            self.assertEqual(os.environ["LITERAL"], "$HOME/literal#value")

    def test_toolkit_file_takes_precedence_and_explicit_file_is_supported(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            root = Path(td) / "finetuning"
            root.mkdir()
            parent = root.parent / ".ENV"
            local = root / ".ENV"
            parent.write_text("VALUE=parent", encoding="utf-8")
            local.write_text("VALUE=local", encoding="utf-8")
            self.assertEqual(find_env_file(root), local)
            os.environ["LIFEMATE_ENV_FILE"] = str(parent)
            self.assertEqual(find_env_file(root), parent)
            os.environ["LIFEMATE_ENV_FILE"] = str(root / "missing")
            with self.assertRaisesRegex(ValueError, "does not exist"):
                load_env_file(root)

    def test_missing_file_is_optional_and_bad_file_does_not_partially_apply(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            root = Path(td) / "finetuning"
            root.mkdir()
            self.assertEqual(load_env_file(root), {})
            for bad_line in ('BAD="unclosed', "not an assignment", 'BAD="$UNDEFINED_VAR/path"'):
                (root / ".ENV").write_text("FIRST=value\n" + bad_line, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_env_file(root)
                self.assertNotIn("FIRST", os.environ)

    def test_shell_exports_quote_literals_without_executing_commands(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            path = Path(td) / ".ENV"
            path.write_text("PAYLOAD='$(touch NEVER_EXECUTE); echo unsafe'\n", encoding="utf-8")
            os.environ["LIFEMATE_ENV_FILE"] = str(path)
            output = io.StringIO()
            with patch("sys.argv", ["env.py", "--shell"]), contextlib.redirect_stdout(output):
                self.assertEqual(main(), 0)
            self.assertEqual(shlex.split(output.getvalue()),
                             ["export", "PAYLOAD=$(touch NEVER_EXECUTE); echo unsafe"])

    def test_cli_defaults_and_explicit_argument_precedence(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            path = Path(td) / ".ENV"
            path.write_text("DATA_ROOT=/dataset\nRESULT=/results\n", encoding="utf-8")
            os.environ["LIFEMATE_ENV_FILE"] = str(path)
            defaults = parser().parse_args(["submit", "--batch-size", "4"])
            self.assertEqual(defaults.data_root, "/dataset")
            self.assertEqual(defaults.output, "/results")
            explicit = parser().parse_args(["submit", "--data-root", "/other-data", "--output", "/other-output"])
            self.assertEqual(explicit.data_root, "/other-data")
            self.assertEqual(explicit.output, "/other-output")
            validation = parser().parse_args(["validate-submission"])
            self.assertEqual(validation.data_root, "/dataset")
            self.assertEqual(Path(validation.input), Path("/results/submission.jsonl"))

    def test_cli_still_requires_paths_when_no_env_defaults_exist(self):
        with patch.dict(os.environ, {}, clear=True), patch("ftkit.env.find_env_file", return_value=None):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser().parse_args(["submit"])


if __name__ == "__main__":
    unittest.main()
