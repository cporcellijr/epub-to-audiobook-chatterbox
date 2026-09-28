"""Pure-Python tests for chatterbox/config.py: the deep-merge helper and the
atomic (temp file + os.replace) config save (F-52). No GPU or running server needed.

Run inside the image (has PyYAML etc.):
    docker run --rm --entrypoint python3 -v <repo>/chatterbox:/app -w /app \
        chatterbox-tts-server:local -m unittest discover -s tests
"""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from threading import Lock

# tests/ is not a package here; make the chatterbox root (this file's parent's
# parent) importable regardless of how the test runner set up sys.path/top-level-dir.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as config_module
import yaml


class TestDeepMergeDicts(unittest.TestCase):
    def test_nested_merge_source_overwrites_destination(self) -> None:
        destination = {"a": 1, "b": {"x": 1, "y": 2}, "c": 3}
        source = {"b": {"y": 20, "z": 30}, "d": 4}
        result = config_module._deep_merge_dicts(source, destination)
        self.assertEqual(
            result, {"a": 1, "b": {"x": 1, "y": 20, "z": 30}, "c": 3, "d": 4}
        )

    def test_non_dict_source_value_overwrites_dict_destination_node(self) -> None:
        destination = {"a": {"nested": True}}
        source = {"a": "now a plain string"}
        result = config_module._deep_merge_dicts(source, destination)
        self.assertEqual(result, {"a": "now a plain string"})

    def test_destination_is_modified_in_place_and_returned(self) -> None:
        destination = {"a": 1}
        result = config_module._deep_merge_dicts({"b": 2}, destination)
        self.assertIs(result, destination)
        self.assertEqual(destination, {"a": 1, "b": 2})


class TestAtomicConfigSave(unittest.TestCase):
    """F-52: saves go through a same-directory temp file plus os.replace, so a
    concurrent reader never sees a truncated file, and no .tmp/.bak is left behind."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="chatterbox_cfg_test_")
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)
        self._original_config_path = config_module.CONFIG_FILE_PATH
        config_module.CONFIG_FILE_PATH = Path(self.tmp_dir) / "config.yaml"
        self.addCleanup(self._restore_config_path)

    def _restore_config_path(self) -> None:
        config_module.CONFIG_FILE_PATH = self._original_config_path

    @staticmethod
    def _bare_manager(config: dict) -> "config_module.YamlConfigManager":
        # Bypass __init__ (device detection, default-path creation) entirely; the
        # save path under test only needs self._lock and self.config.
        manager = config_module.YamlConfigManager.__new__(config_module.YamlConfigManager)
        manager._lock = Lock()
        manager.config = config
        return manager

    def _temp_and_backup_paths(self):
        temp_file = config_module.CONFIG_FILE_PATH.with_suffix(
            config_module.CONFIG_FILE_PATH.suffix + ".tmp"
        )
        backup_file = config_module.CONFIG_FILE_PATH.with_suffix(
            config_module.CONFIG_FILE_PATH.suffix + ".bak"
        )
        return temp_file, backup_file

    def test_save_writes_valid_yaml_and_leaves_no_temp_file(self) -> None:
        manager = self._bare_manager(
            {"server": {"port": 8004}, "model": {"repo_id": "chatterbox"}}
        )
        self.assertTrue(manager.save_config_yaml())

        self.assertTrue(config_module.CONFIG_FILE_PATH.exists())
        temp_file, _ = self._temp_and_backup_paths()
        self.assertFalse(temp_file.exists())

        with open(config_module.CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        self.assertEqual(loaded["server"]["port"], 8004)
        self.assertEqual(loaded["model"]["repo_id"], "chatterbox")

    def test_second_save_replaces_content_and_cleans_up_tmp_and_bak(self) -> None:
        manager = self._bare_manager({"model": {"repo_id": "chatterbox"}})
        self.assertTrue(manager.save_config_yaml())

        manager.config = {"model": {"repo_id": "chatterbox-turbo"}}
        self.assertTrue(manager.save_config_yaml())

        with open(config_module.CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
        self.assertEqual(loaded["model"]["repo_id"], "chatterbox-turbo")

        temp_file, backup_file = self._temp_and_backup_paths()
        self.assertFalse(temp_file.exists())
        self.assertFalse(backup_file.exists())


if __name__ == "__main__":
    unittest.main()
