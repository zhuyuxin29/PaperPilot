"""PaperPilot 配置与路径管理。

职责：
- 读取 .env 与进程环境变量；
- 集中解析数据目录与子目录，业务代码不得硬编码绝对路径；
- 校验 GRAPH_MODE 与预算参数；
- 输出脱敏配置摘要与能力检查结果。

边界：
- 本模块不连接任何外部服务；Neo4j 连通性由图谱模块在 auto/on 下另测。
- 密码与 API Key 只判断“是否配置”，不写日志、不打印。
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

GRAPH_MODES = ("off", "auto", "on")
API_MODES = ("direct", "http")
DEFAULT_NEO4J_URI = "neo4j://127.0.0.1:7687"
_PAPER_DIR_RE = re.compile(r"^paper_\d{3,}$")


def load_dotenv(path: Path | str | None = None) -> None:
    """读取 .env 文件（KEY=VALUE），不覆盖已存在的环境变量。"""
    env_path = Path(path) if path is not None else Path.cwd() / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _clean(env: Mapping[str, str], key: str, default: str = "") -> str:
    return (env.get(key) or default).strip()


def _provider_key(env: Mapping[str, str], provider: str) -> str:
    """按 provider 名称回退读取 ``<PROVIDER>_API_KEY``。

    例如 ``EMBEDDING_PROVIDER=siliconflow`` 时会读取 ``SILICONFLOW_API_KEY``，
    便于复用已有的服务商 Key，而不是强制复制一份到 EMBEDDING_API_KEY。
    """
    if not provider:
        return ""
    return _clean(env, f"{provider.upper().replace('-', '_')}_API_KEY")


def _default_data_dir(env: Mapping[str, str]) -> Path:
    explicit = _clean(env, "PAPERPILOT_DATA_DIR")
    if explicit:
        return Path(explicit)
    local_appdata = _clean(env, "LOCALAPPDATA")
    if local_appdata:
        return Path(local_appdata) / "PaperPilot"
    return Path.home() / "PaperPilot"


def _positive_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = _clean(env, key)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} 必须是整数，当前值：{raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{key} 必须为正数，当前值：{value}")
    return value


def _optional_int(env: Mapping[str, str], key: str) -> int | None:
    raw = _clean(env, key)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} 必须是整数，当前值：{raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{key} 必须为正数，当前值：{value}")
    return value


def _bounded_float(
    env: Mapping[str, str],
    key: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw = _clean(env, key)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} 必须是数字，当前值：{raw!r}") from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{key} 必须在 {minimum}~{maximum} 之间，当前值：{value}"
        )
    return value


@dataclass
class Settings:
    """一次运行使用的配置快照，构造后视为只读。"""

    data_dir: Path
    graph_mode: str = "off"
    api_mode: str = "direct"
    api_url: str = "http://127.0.0.1:8000"
    vector_backend: str = "numpy"

    chat_provider: str = ""
    chat_base_url: str = ""
    chat_model: str = ""
    chat_api_key: str = ""

    embedding_provider: str = ""
    embedding_base_url: str = ""
    embedding_model: str = ""
    embedding_api_key: str = ""
    embedding_dimension: int | None = None

    neo4j_uri: str = DEFAULT_NEO4J_URI
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    max_steps: int = 8
    max_tool_calls: int = 12
    timeout_seconds: int = 240

    # 可选的检索规则词面加权；默认 0（纯 RRF），避免评测归因被隐式重排影响。
    retrieval_term_boost: float = 0.0

    # ---- 派生路径（全部集中在这里） ----
    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.sqlite3"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / ".tmp"

    def paper_dir(self, paper_id: str) -> Path:
        """单篇论文的数据目录：``{data_dir}/paper_001``（源文件 + 片段快照）。"""
        if not _PAPER_DIR_RE.fullmatch(paper_id):
            raise ValueError(
                f"非法 paper_id：{paper_id!r}（应形如 paper_001）"
            )
        return self.data_dir / paper_id

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def summaries_dir(self) -> Path:
        return self.data_dir / "summaries"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    # ---- 能力检查（只判断配置，不判断连通性） ----
    @property
    def chat_configured(self) -> bool:
        return bool(self.chat_model and self.chat_api_key)

    @property
    def embedding_configured(self) -> bool:
        return bool(self.embedding_model and self.embedding_api_key)

    def capabilities(self) -> dict[str, bool]:
        return {
            "model": self.chat_configured,
            "embedding": self.embedding_configured,
            "graph": self.graph_mode != "off" and bool(self.neo4j_password),
        }

    def ensure_dirs(self) -> None:
        """创建数据目录与子目录；可重复调用。"""
        for path in (
            self.data_dir,
            self.tmp_dir,
            self.chroma_dir,
            self.index_dir,
            self.summaries_dir,
            self.logs_dir,
            self.exports_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def assert_writable(self) -> None:
        """验证数据目录确实可写（写一个探针文件再删除）。

        仅当目录存在但没有写权限时才会失败；失败时给出可操作的提示。
        """
        try:
            self.ensure_dirs()
            probe = self.data_dir / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except OSError as exc:
            raise PermissionError(
                f"数据目录不可写：{self.data_dir}（{exc}）。"
                "请修改 .env 中的 PAPERPILOT_DATA_DIR，或检查该目录权限。"
            ) from exc

    def summary(self) -> dict[str, object]:
        """脱敏配置摘要：可用于日志与诊断，不含密钥。"""
        return {
            "data_dir": str(self.data_dir),
            "db_path": str(self.db_path),
            "paper_dir": str(self.data_dir / "paper_XXX"),
            "graph_mode": self.graph_mode,
            "chat": {
                "provider": self.chat_provider or "(未配置)",
                "model": self.chat_model or "(未配置)",
                "api_key": "已配置" if self.chat_api_key else "未配置",
            },
            "embedding": {
                "model": self.embedding_model or "(未配置)",
                "dimension": self.embedding_dimension,
                "api_key": "已配置" if self.embedding_api_key else "未配置",
            },
            "neo4j": {
                "uri": self.neo4j_uri,
                "database": self.neo4j_database,
                "password": "已配置" if self.neo4j_password else "未配置",
            },
            "capabilities": self.capabilities(),
            "budget": {
                "max_steps": self.max_steps,
                "max_tool_calls": self.max_tool_calls,
                "timeout_seconds": self.timeout_seconds,
            },
            "retrieval": {"term_boost": self.retrieval_term_boost},
        }


def load_settings(
    env: Mapping[str, str] | None = None,
    dotenv_path: Path | str | None = None,
) -> Settings:
    """从环境变量（或显式传入的映射）构造 Settings。

    env 为 None 时先加载 .env，再读取进程环境变量；传入映射时完全按映射解析，
    便于测试，不读取磁盘上的 .env。
    """
    if env is None:
        load_dotenv(dotenv_path)
        env = os.environ

    graph_mode = _clean(env, "GRAPH_MODE", "off").lower()
    if graph_mode not in GRAPH_MODES:
        raise ValueError(
            f"GRAPH_MODE 只能是 {'/'.join(GRAPH_MODES)}，当前值：{graph_mode!r}"
        )

    api_mode = _clean(env, "PAPERPILOT_API_MODE", "direct").lower()
    if api_mode not in API_MODES:
        raise ValueError(
            f"PAPERPILOT_API_MODE 只能是 {'/'.join(API_MODES)}，当前值：{api_mode!r}"
        )

    vector_backend = _clean(env, "VECTOR_BACKEND", "numpy").lower()
    if vector_backend not in ("numpy", "chroma"):
        raise ValueError(
            f"VECTOR_BACKEND 只能是 numpy/chroma，当前值：{vector_backend!r}"
        )

    chat_provider = _clean(env, "CHAT_PROVIDER")
    chat_api_key = (
        _clean(env, "CHAT_API_KEY")
        or _provider_key(env, chat_provider)
        or _clean(env, "DEEPSEEK_API_KEY")
    )
    embedding_provider = _clean(env, "EMBEDDING_PROVIDER")
    embedding_api_key = _clean(env, "EMBEDDING_API_KEY") or _provider_key(
        env, embedding_provider
    )

    return Settings(
        data_dir=_default_data_dir(env),
        graph_mode=graph_mode,
        api_mode=api_mode,
        api_url=_clean(env, "PAPERPILOT_API_URL", "http://127.0.0.1:8000"),
        vector_backend=vector_backend,
        chat_provider=chat_provider,
        chat_base_url=_clean(env, "CHAT_BASE_URL"),
        chat_model=_clean(env, "CHAT_MODEL"),
        chat_api_key=chat_api_key,
        embedding_provider=embedding_provider,
        embedding_base_url=_clean(env, "EMBEDDING_BASE_URL"),
        embedding_model=_clean(env, "EMBEDDING_MODEL"),
        embedding_api_key=embedding_api_key,
        embedding_dimension=_optional_int(env, "EMBEDDING_DIMENSION"),
        neo4j_uri=_clean(env, "NEO4J_URI", DEFAULT_NEO4J_URI),
        neo4j_username=_clean(env, "NEO4J_USERNAME", "neo4j"),
        neo4j_password=_clean(env, "NEO4J_PASSWORD"),
        neo4j_database=_clean(env, "NEO4J_DATABASE", "neo4j"),
        max_steps=_positive_int(env, "AGENT_MAX_STEPS", 8),
        max_tool_calls=_positive_int(env, "AGENT_MAX_TOOL_CALLS", 12),
        timeout_seconds=_positive_int(env, "AGENT_TIMEOUT_SECONDS", 240),
        retrieval_term_boost=_bounded_float(
            env, "RETRIEVAL_TERM_BOOST", 0.0, 0.0, 0.05
        ),
    )


def main() -> None:
    """python -m paperpilot.config：创建数据目录并打印脱敏配置摘要。"""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
    settings = load_settings()
    try:
        settings.assert_writable()
    except PermissionError as exc:
        print(f"配置检查失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
    summary = settings.summary()
    summary["dirs_ready"] = True
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
