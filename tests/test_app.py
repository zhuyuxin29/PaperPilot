"""模块 7（最小界面）验收测试：辅助函数 + Streamlit 无头冒烟测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from paperpilot import storage
from paperpilot.config import Settings
from paperpilot.ingestion import pipeline
from paperpilot.schemas import PaperState
from pdf_factory import build_pdf

import app as app_module

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
CLEAN_PDF = build_pdf(["First page content.", "Second page content."])
BLANK_PDF = build_pdf(["First page content.", None, "Third page content."])


class HelperTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.settings = Settings(
            data_dir=Path(self._tmp.name) / "data", graph_mode="off"
        )
        self.settings.ensure_dirs()
        self.conn = storage.connect(self.settings.db_path)
        storage.ensure_schema(self.conn)

    def tearDown(self) -> None:
        self.conn.close()
        self._tmp.cleanup()

    def test_format_state(self) -> None:
        self.assertEqual(app_module.format_state(PaperState.READY), "可检索")
        self.assertEqual(app_module.format_state(PaperState.PARTIAL), "部分可检索")
        self.assertEqual(app_module.format_state(PaperState.FAILED), "失败")

    def test_paper_rows(self) -> None:
        pipeline.ingest_pdf(CLEAN_PDF, "a.pdf", self.settings, self.conn)
        pipeline.ingest_pdf(BLANK_PDF, "b.pdf", self.settings, self.conn)

        rows = app_module.paper_rows(self.conn)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["编号"], "paper_002")
        self.assertEqual(rows[0]["状态"], "部分可检索")
        self.assertEqual(rows[1]["编号"], "paper_001")
        self.assertEqual(rows[1]["状态"], "可检索")
        self.assertEqual(rows[1]["页数"], 2)
        self.assertGreater(rows[1]["片段数"], 0)

    def test_finalize_pending_uses_thread_local_connection(self) -> None:
        import threading
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from paperpilot.agent.runner import AgentResult
        from paperpilot.schemas import Paper
        from paperpilot.ui import chat

        storage.save_paper(
            self.conn,
            Paper(
                id="paper_001",
                title="demo",
                file_name="demo.pdf",
                sha256="a" * 64,
            ),
        )
        result = AgentResult(
            answer="答案",
            citations=(),
            status="COMPLETED",
            stop_reason="DONE",
            trace=(),
            steps=1,
            tool_calls=0,
            run_id="run_001",
        )
        fake_st = SimpleNamespace(
            session_state={
                "pending": {
                    "paper_id": "paper_001",
                    "question": "问题",
                    "holder": {"result": result},
                },
                "chats": {},
            }
        )
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                chat._finalize_pending(self.settings)
            except BaseException as exc:  # noqa: BLE001 - 测试中收集后统一断言
                errors.append(exc)

        with patch.object(chat, "st", fake_st), patch.object(
            chat, "_save_chat_history", Mock()
        ):
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        entry = fake_st.session_state["chats"]["paper_001"][0]
        self.assertEqual(entry["revision"], 1)


class AppSmokeTest(unittest.TestCase):
    def test_app_runs_without_exception(self) -> None:
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = str(Path(tmp) / "data")
            previous = os.environ.get("PAPERPILOT_DATA_DIR")
            os.environ["PAPERPILOT_DATA_DIR"] = data_dir
            try:
                at = AppTest.from_file(str(APP_PATH), default_timeout=60)
                at.run()
            finally:
                if previous is None:
                    os.environ.pop("PAPERPILOT_DATA_DIR", None)
                else:
                    os.environ["PAPERPILOT_DATA_DIR"] = previous

        self.assertFalse(at.exception, msg=str(at.exception))
        self.assertTrue(any("PaperPilot" in title.value for title in at.title))

    def _prepare_data(self, tmp: Path) -> Path:
        data_dir = tmp / "data"
        settings = Settings(data_dir=data_dir, graph_mode="off")
        settings.ensure_dirs()
        conn = storage.connect(settings.db_path)
        storage.ensure_schema(conn)
        pipeline.ingest_pdf(CLEAN_PDF, "a.pdf", settings, conn)
        conn.close()
        return data_dir

    def test_chat_page_renders_with_paper(self) -> None:
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = self._prepare_data(Path(tmp))
            previous = os.environ.get("PAPERPILOT_DATA_DIR")
            os.environ["PAPERPILOT_DATA_DIR"] = str(data_dir)
            try:
                at = AppTest.from_file(str(APP_PATH), default_timeout=60)
                at.run()
            finally:
                if previous is None:
                    os.environ.pop("PAPERPILOT_DATA_DIR", None)
                else:
                    os.environ["PAPERPILOT_DATA_DIR"] = previous

        self.assertFalse(at.exception, msg=str(at.exception))
        self.assertGreaterEqual(len(at.chat_input), 1)

    def test_points_and_library_pages_render(self) -> None:
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = self._prepare_data(Path(tmp))
            previous = os.environ.get("PAPERPILOT_DATA_DIR")
            os.environ["PAPERPILOT_DATA_DIR"] = str(data_dir)
            try:
                points = AppTest.from_string(
                    "import streamlit as st\n"
                    "from app import _page_points\n"
                    "st.session_state['current_paper'] = 'paper_001'\n"
                    "st.session_state['show_source'] = True\n"
                    "_page_points()\n",
                    default_timeout=60,
                )
                points.run()
                self.assertFalse(points.exception, msg=str(points.exception))

                chat_source = AppTest.from_string(
                    "import streamlit as st\n"
                    "from app import _page_chat\n"
                    "st.session_state['current_paper'] = 'paper_001'\n"
                    "st.session_state['show_source'] = True\n"
                    "_page_chat()\n",
                    default_timeout=60,
                )
                chat_source.run()
                self.assertFalse(
                    chat_source.exception, msg=str(chat_source.exception)
                )

                library = AppTest.from_string(
                    "from app import _page_library\n_page_library()\n",
                    default_timeout=60,
                )
                library.run()
                self.assertFalse(library.exception, msg=str(library.exception))
            finally:
                if previous is None:
                    os.environ.pop("PAPERPILOT_DATA_DIR", None)
                else:
                    os.environ["PAPERPILOT_DATA_DIR"] = previous


if __name__ == "__main__":
    unittest.main()
