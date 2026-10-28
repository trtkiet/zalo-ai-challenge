import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import setup_env


class SetupPythonTests(unittest.TestCase):
    def fixture(self, root):
        kit, home = root / "kit", root / "runtime"
        (kit / "requirements").mkdir(parents=True)
        for environment in ("llm-train", "llm-infer"):
            (kit / "requirements" / f"{environment}.txt").write_text("fake-package>=1\n")
        return kit, home

    def create(self, kit, home, version, environment="llm-train"):
        with patch.object(setup_env, "ROOT", kit), \
                patch.object(setup_env.sys, "version_info", (*version, 1, "final", 0)), \
                patch.object(setup_env.sys, "version", f"{version[0]}.{version[1]}.1 (test)"), \
                patch.object(setup_env.venv, "EnvBuilder") as builder, \
                patch.object(setup_env, "run") as commands, \
                patch.object(setup_env.subprocess, "check_output", return_value="fake-package==1\n"), \
                contextlib.redirect_stdout(io.StringIO()):
            builder.return_value.create.side_effect = lambda target: target.mkdir(parents=True)
            setup_env.setup_llm_environment(environment, home)
            return builder.call_count, commands.call_args_list

    def test_both_versions_create_both_llm_environments(self):
        for version in ((3, 11), (3, 12)):
            for environment in ("llm-train", "llm-infer"):
                with self.subTest(version=version, environment=environment), tempfile.TemporaryDirectory() as td:
                    kit, home = self.fixture(Path(td))
                    count, commands = self.create(kit, home, version, environment)
                    self.assertEqual(count, 1)
                    state = json.loads((home / ".venvs" / environment / "setup.json").read_text())
                    self.assertEqual(state["python_version"], list(version))
                    self.assertFalse(state["gpu_model_smoke_verified"])
                    self.assertTrue(any("check" in list(map(str, call.args)) for call in commands))

    def test_matching_environment_reused_without_installation(self):
        with tempfile.TemporaryDirectory() as td:
            kit, home = self.fixture(Path(td))
            self.create(kit, home, (3, 12))
            count, commands = self.create(kit, home, (3, 12))
            self.assertEqual(count, 0)
            self.assertEqual(commands, [])

    def test_switching_minor_version_refuses_to_reuse_or_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            kit, home = self.fixture(Path(td))
            self.create(kit, home, (3, 11))
            marker = home / ".venvs/llm-train/setup.json"
            before = marker.read_bytes()
            with self.assertRaisesRegex(ValueError, "Refusing to reuse.*Python"):
                self.create(kit, home, (3, 12))
            self.assertEqual(marker.read_bytes(), before)

    def test_locks_are_python_specific_and_legacy_locks_only_apply_to_311(self):
        with tempfile.TemporaryDirectory() as td:
            kit, home = self.fixture(Path(td))
            legacy = kit / "requirements/llm-train.lock.txt"
            legacy.write_text("legacy-package==1\n")
            self.create(kit, home, (3, 12))
            marker = json.loads((home / ".venvs/llm-train/setup.json").read_text())
            self.assertEqual(marker["source"], "llm-train.txt")
            self.create(kit, Path(td) / "runtime311", (3, 11))
            marker = json.loads((Path(td) / "runtime311/.venvs/llm-train/setup.json").read_text())
            self.assertEqual(marker["source"], legacy.name)
            lock = setup_env.llm_lock_path(kit, "llm-train", (3, 12))
            lock.write_text("python312-package==2\n")
            self.create(kit, Path(td) / "runtime312locked", (3, 12))
            marker = json.loads((Path(td) / "runtime312locked/.venvs/llm-train/setup.json").read_text())
            self.assertEqual(marker["source"], "llm-train.py312.lock.txt")
            self.assertEqual(marker["requirements_sha256"], hashlib.sha256(lock.read_bytes()).hexdigest())

    def test_legacy_marker_python_string_is_recognized(self):
        self.assertEqual(setup_env.recorded_python_version({"python": "3.11.9 (main, build)"}), (3, 11))
        self.assertEqual(setup_env.recorded_python_version({"python": "3.12.11 (main, build)"}), (3, 12))
        self.assertIsNone(setup_env.recorded_python_version({}))

    def test_unsupported_python_rejected_before_environment_creation(self):
        for version in ((3, 10), (3, 13)):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as td:
                kit, home = self.fixture(Path(td))
                with self.assertRaisesRegex(ValueError, "3.11 or 3.12"):
                    self.create(kit, home, version)
                self.assertFalse(home.exists())


if __name__ == "__main__":
    unittest.main()
