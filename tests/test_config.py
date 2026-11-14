from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from paperpilot import config


class LoadSettingsTest(unittest.TestCase):
    def test_defaults_with_localappdata(self) -> None:
        env = {"LOCALAPPDATA": r"C:\Users\demo\AppData\Local"}
        settings = config.load_settings(env=env)

        self.assertEqual(
            settings.data_dir,
            Path(r"C:\Users\demo\AppData\Local") / "PaperPilot",
        )
        self.assertEqual(settings.graph_mode, "off")
        self.assertEqual(settings.max_steps, 8)
        self.assertEqual(settings.max_tool_calls, 12)
        self.assertEqual(settings.timeout_seconds, 240)

    def test_explicit_data_dir_wins(self) -> None:
        env = {
            "PAPERPILOT_DATA_DIR": r"D:\tmp\pp-data",
            "LOCALAPPDATA": r"C:\Users\demo\AppData\Local",
        }
        settings = config.load_settings(env=env)
        self.assertEqual(settings.data_dir, Path(r"D:\tmp\pp-data"))

    def test_graph_mode_is_case_insensitive(self) -> None:
        settings = config.load_settings(env={"GRAPH_MODE": "ON"})
        self.assertEqual(settings.graph_mode, "on")

    def test_invalid_graph_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            config.load_settings(env={"GRAPH_MODE": "bogus"})

    def test_invalid_budget_raises(self) -> None:
        with self.assertRaises(ValueError):
            config.load_settings(env={"AGENT_MAX_STEPS": "abc"})
        with self.assertRaises(ValueError):
            config.load_settings(env={"AGENT_MAX_TOOL_CALLS": "0"})

    def test_retrieval_term_boost(self) -> None:
        self.assertEqual(
            config.load_settings(env={}).retrieval_term_boost, 0.0
        )
        self.assertEqual(
            config.load_settings(
                env={"RETRIEVAL_TERM_BOOST": "0.02"}
            ).retrieval_term_boost,
            0.02,
        )
        with self.assertRaises(ValueError):
            config.load_settings(env={"RETRIEVAL_TERM_BOOST": "1.0"})
        with self.assertRaises(ValueError):
            config.load_settings(env={"RETRIEVAL_TERM_BOOST": "abc"})

    def test_embedding_dimension(self) -> None:
        settings = config.load_settings(env={"EMBEDDING_DIMENSION": "1024"})
        self.assertEqual(settings.embedding_dimension, 1024)
        self.assertIsNone(
            config.load_settings(env={"EMBEDDING_DIMENSION": ""}).embedding_dimension
        )

    def test_capabilities_follow_config(self) -> None:
        empty = config.load_settings(env={})
        self.assertEqual(
            empty.capabilities(),
            {"model": False, "embedding": False, "graph": False},
        )

        configured = config.load_settings(
            env={
                "CHAT_MODEL": "demo-chat",
                "CHAT_API_KEY": "sk-demo",
                "EMBEDDING_MODEL": "demo-embed",
                "EMBEDDING_API_KEY": "sk-demo",
                "NEO4J_PASSWORD": "pw",
                "GRAPH_MODE": "auto",
            }
        )
        self.assertTrue(configured.chat_configured)
        self.assertTrue(configured.embedding_configured)
        self.assertTrue(configured.capabilities()["graph"])

        graph_off = config.load_settings(
            env={"NEO4J_PASSWORD": "pw", "GRAPH_MODE": "off"}
        )
        self.assertFalse(graph_off.capabilities()["graph"])

    def test_summary_hides_secrets(self) -> None:
        settings = config.load_settings(
            env={
                "CHAT_MODEL": "demo-chat",
                "CHAT_API_KEY": "sk-super-secret",
                "NEO4J_PASSWORD": "neo4j-secret",
                "GRAPH_MODE": "on",
            }
        )
        dumped = json.dumps(settings.summary(), ensure_ascii=False)
        self.assertNotIn("sk-super-secret", dumped)
        self.assertNotIn("neo4j-secret", dumped)
        self.assertIn("已配置", dumped)

    def test_provider_specific_key_fallback(self) -> None:
        settings = config.load_settings(
            env={
                "EMBEDDING_PROVIDER": "siliconflow",
                "EMBEDDING_MODEL": "BAAI/bge-m3",
                "SILICONFLOW_API_KEY": "sk-from-provider",
            }
        )
        self.assertEqual(settings.embedding_api_key, "sk-from-provider")
        self.assertTrue(settings.embedding_configured)

    def test_explicit_embedding_key_wins(self) -> None:
        settings = config.load_settings(
            env={
                "EMBEDDING_PROVIDER": "siliconflow",
                "EMBEDDING_MODEL": "BAAI/bge-m3",
                "EMBEDDING_API_KEY": "sk-explicit",
                "SILICONFLOW_API_KEY": "sk-from-provider",
            }
        )
        self.assertEqual(settings.embedding_api_key, "sk-explicit")

    def test_chat_key_fallback(self) -> None:
        settings = config.load_settings(
            env={"CHAT_MODEL": "deepseek-chat", "DEEPSEEK_API_KEY": "sk-deepseek"}
        )
        self.assertEqual(settings.chat_api_key, "sk-deepseek")
        self.assertTrue(settings.chat_configured)


class EnsureDirsTest(unittest.TestCase):
    def test_ensure_dirs_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {"PAPERPILOT_DATA_DIR": tmp}
            settings = config.load_settings(env=env)

            settings.ensure_dirs()
            settings.ensure_dirs()

            for path in (
                settings.data_dir,
                settings.tmp_dir,
                settings.chroma_dir,
                settings.index_dir,
                settings.logs_dir,
                settings.exports_dir,
            ):
                self.assertTrue(path.is_dir(), msg=str(path))
            self.assertEqual(settings.db_path.parent, settings.data_dir)

    def test_paper_dir(self) -> None:
        env = {"PAPERPILOT_DATA_DIR": r"D:\pp-data"}
        settings = config.load_settings(env=env)

        self.assertEqual(settings.paper_dir("paper_001"), Path(r"D:\pp-data") / "paper_001")
        self.assertEqual(settings.paper_dir("paper_1000"), Path(r"D:\pp-data") / "paper_1000")
        for bad in ("paper01", "paper_1", "../evil", "paper_001/../x", ""):
            with self.assertRaises(ValueError):
                settings.paper_dir(bad)

    def test_assert_writable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = config.load_settings(env={"PAPERPILOT_DATA_DIR": tmp})
            settings.assert_writable()
            self.assertFalse((settings.data_dir / ".write_probe").exists())

    def test_assert_writable_rejects_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "not_a_dir"
            file_path.write_text("x", encoding="utf-8")
            settings = config.load_settings(
                env={"PAPERPILOT_DATA_DIR": str(file_path)}
            )
            with self.assertRaises(PermissionError):
                settings.assert_writable()


class LoadDotenvTest(unittest.TestCase):
    def test_dotenv_parsing_and_no_override(self) -> None:
        keys = ["PAPERPILOT_DATA_DIR", "GRAPH_MODE", "CHAT_API_KEY"]
        backup = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                os.environ.pop(key, None)

            with tempfile.TemporaryDirectory() as tmp:
                env_file = Path(tmp) / ".env"
                env_file.write_text(
                    "# 注释\n"
                    "GRAPH_MODE=auto\n"
                    'CHAT_API_KEY="sk-from-dotenv"\n'
                    "PAPERPILOT_DATA_DIR=\n",
                    encoding="utf-8",
                )

                os.environ["GRAPH_MODE"] = "off"
                config.load_dotenv(env_file)

            self.assertEqual(os.environ["GRAPH_MODE"], "off")
            self.assertEqual(os.environ["CHAT_API_KEY"], "sk-from-dotenv")
            self.assertEqual(os.environ["PAPERPILOT_DATA_DIR"], "")
        finally:
            for key in keys:
                if backup[key] is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = backup[key]

    def test_dotenv_with_bom(self) -> None:
        keys = ["PAPERPILOT_DATA_DIR"]
        backup = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                os.environ.pop(key, None)
            with tempfile.TemporaryDirectory() as tmp:
                env_file = Path(tmp) / ".env"
                env_file.write_bytes(
                    "\ufeffPAPERPILOT_DATA_DIR=D:\\bom-data\n".encode("utf-8")
                )
                config.load_dotenv(env_file)
            self.assertEqual(os.environ["PAPERPILOT_DATA_DIR"], r"D:\bom-data")
        finally:
            for key in keys:
                if backup[key] is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = backup[key]


if __name__ == "__main__":
    unittest.main()
