"""PaperPilot HTTP 客户端：Streamlit 在 API 模式下调用后端。

只依赖 ``requests``，与 FastAPI 后端解耦；后端地址由
``PAPERPILOT_API_URL``（默认 http://127.0.0.1:8000）提供。
"""

from __future__ import annotations

import time
from typing import Any

import requests


class ApiError(RuntimeError):
    def __init__(self, message: str, code: str = "API_ERROR", status: int = 0):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


class PaperPilotClient:
    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

    # ------------------------------------------------------------ 内部

    def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(
                method, url, timeout=self.timeout, **kwargs
            )
        except requests.RequestException as exc:
            raise ApiError(
                f"无法连接后端 {self.base_url}：{exc}", "API_UNREACHABLE"
            ) from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = None
            if isinstance(detail, dict):
                raise ApiError(
                    str(detail.get("message") or detail),
                    str(detail.get("code") or "API_ERROR"),
                    response.status_code,
                )
            raise ApiError(
                f"后端返回 {response.status_code}",
                "API_ERROR",
                response.status_code,
            )
        if response.status_code == 204:
            return None
        return response.json()

    # ------------------------------------------------------------ 接口

    def health(self) -> dict:
        return self._request("GET", "/health")

    def list_papers(self) -> list[dict]:
        return self._request("GET", "/papers")

    def get_paper(self, paper_id: str) -> dict:
        return self._request("GET", f"/papers/{paper_id}")

    def get_chunk(self, paper_id: str, chunk_id: str) -> dict:
        return self._request(
            "GET", f"/papers/{paper_id}/chunks/{chunk_id}"
        )

    def build_graph(self, paper_id: str) -> dict:
        return self._request("POST", f"/papers/{paper_id}/graph")

    def reprocess_paper(self, paper_id: str) -> dict:
        return self._request("POST", f"/papers/{paper_id}/reprocess")

    def upload_paper(
        self, data: bytes, file_name: str, *, background: bool = False
    ) -> dict:
        params = {"background": "true"} if background else None
        return self._request(
            "POST",
            "/papers",
            files={"file": (file_name, data, "application/pdf")},
            params=params,
        )

    def wait_for_paper(
        self, paper_id: str, *, timeout: float = 300.0, interval: float = 0.8
    ) -> dict:
        """轮询后台入库状态，直到不再是 parsing/uploaded；失败状态也返回。"""
        deadline = time.monotonic() + timeout
        while True:
            paper = self.get_paper(paper_id)
            if paper["state"] not in ("parsing", "uploaded"):
                return paper
            if time.monotonic() >= deadline:
                raise ApiError(
                    "后台入库超时，请稍后在文献列表中查看结果",
                    "INGEST_TIMEOUT",
                    0,
                )
            time.sleep(interval)

    def delete_paper(self, paper_id: str) -> None:
        self._request("DELETE", f"/papers/{paper_id}")

    def pdf_bytes(self, paper_id: str) -> bytes:
        url = f"{self.base_url}/papers/{paper_id}/pdf"
        try:
            response = self.session.get(url, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ApiError(
                f"无法连接后端 {self.base_url}：{exc}", "API_UNREACHABLE"
            ) from exc
        if response.status_code >= 400:
            raise ApiError("下载 PDF 失败", "PDF_NOT_FOUND", response.status_code)
        return response.content

    def get_summary(self, paper_id: str) -> dict | None:
        try:
            return self._request("GET", f"/papers/{paper_id}/summary")
        except ApiError as exc:
            if exc.code == "SUMMARY_NOT_GENERATED":
                return None
            raise

    def generate_summary(self, paper_id: str, force: bool = True) -> dict:
        return self._request(
            "POST",
            f"/papers/{paper_id}/summary",
            params={"force": str(force).lower()},
        )

    def chat(self, paper_id: str, question: str) -> dict:
        return self._request(
            "POST",
            f"/papers/{paper_id}/chat",
            json={"question": question},
        )
