"""Neo4j 客户端与固定查询模板（模块 11）。

只存“关系视图 + 片段指针”，不存全文、不做向量检索；
图谱是辅助召回，不产生可引用证据——引用仍要 ``read_passages``。
"""

from __future__ import annotations

from typing import Any

from ..config import Settings

WORKSPACE_ID = "local"
EXTRACTOR_VERSION = "graph-v1"

ENTITY_TYPES = ("Paper", "Method", "Experiment", "Result", "Conclusion", "Dataset")
RELATION_TYPES = (
    "USES_METHOD",
    "APPLIED_IN",
    "PRODUCES",
    "SUPPORTS_CONCLUSION",
)

QUERY_TEMPLATES: dict[str, str] = {
    "method_experiments": """
        MATCH (cl:Claim {workspace_id: $workspace_id, status: 'validated',
                         relation_type: 'APPLIED_IN'})
              -[:SUBJECT]->(m:Entity {entity_type: 'Method'})
        MATCH (cl)-[:OBJECT]->(e:Entity {entity_type: 'Experiment'})
        MATCH (cl)-[:SUPPORTED_BY]->(c:Chunk)
        MATCH (cl)-[:IN_PAPER]->(p:Paper)
        WHERE c.workspace_id = $workspace_id
          AND c.paper_id IN $paper_ids
          AND (size($query_terms) = 0 OR any(term IN $query_terms WHERE
               toLower(cl.evidence_text) CONTAINS term))
          AND ($entity_name IS NULL
               OR toLower(m.canonical_name) CONTAINS toLower($entity_name))
        RETURN cl.claim_id AS claim_id, cl.relation_type AS relation_type,
               m.canonical_name AS subject, e.canonical_name AS object,
               c.chunk_id AS chunk_id, c.page_number AS page_number,
               c.paper_id AS paper_id
        ORDER BY c.page_number, cl.claim_id
        LIMIT $limit
    """,
    "experiment_results": """
        MATCH (cl:Claim {workspace_id: $workspace_id, status: 'validated',
                         relation_type: 'PRODUCES'})
              -[:SUBJECT]->(e:Entity {entity_type: 'Experiment'})
        MATCH (cl)-[:OBJECT]->(r:Entity {entity_type: 'Result'})
        MATCH (cl)-[:SUPPORTED_BY]->(c:Chunk)
        MATCH (cl)-[:IN_PAPER]->(p:Paper)
        WHERE c.workspace_id = $workspace_id
          AND c.paper_id IN $paper_ids
          AND (size($query_terms) = 0 OR any(term IN $query_terms WHERE
               toLower(cl.evidence_text) CONTAINS term))
          AND ($entity_name IS NULL
               OR toLower(e.canonical_name) CONTAINS toLower($entity_name))
        RETURN cl.claim_id AS claim_id, cl.relation_type AS relation_type,
               e.canonical_name AS subject, r.canonical_name AS object,
               c.chunk_id AS chunk_id, c.page_number AS page_number,
               c.paper_id AS paper_id
        ORDER BY c.page_number, cl.claim_id
        LIMIT $limit
    """,
    "conclusion_support": """
        MATCH (cl:Claim {workspace_id: $workspace_id, status: 'validated',
                         relation_type: 'SUPPORTS_CONCLUSION'})
              -[:SUBJECT]->(s:Entity)
        MATCH (cl)-[:OBJECT]->(o:Entity {entity_type: 'Conclusion'})
        MATCH (cl)-[:SUPPORTED_BY]->(c:Chunk)
        MATCH (cl)-[:IN_PAPER]->(p:Paper)
        WHERE c.workspace_id = $workspace_id
          AND c.paper_id IN $paper_ids
          AND (size($query_terms) = 0 OR any(term IN $query_terms WHERE
               toLower(cl.evidence_text) CONTAINS term))
          AND s.entity_type IN ['Result', 'Experiment']
          AND ($entity_name IS NULL
               OR toLower(o.canonical_name) CONTAINS toLower($entity_name))
        RETURN cl.claim_id AS claim_id, cl.relation_type AS relation_type,
               s.canonical_name AS subject, o.canonical_name AS object,
               c.chunk_id AS chunk_id, c.page_number AS page_number,
               c.paper_id AS paper_id
        ORDER BY c.page_number, cl.claim_id
        LIMIT $limit
    """,
}


class GraphError(RuntimeError):
    """图谱不可用或查询失败；``code`` 供上层映射。"""

    def __init__(self, message: str, code: str = "GRAPH_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class Neo4jClient:
    """薄封装：连接、约束、写入一条 Claim、执行模板查询。"""

    def __init__(self, settings: Settings, driver=None) -> None:
        self._settings = settings
        self._driver = driver

    @classmethod
    def connect(cls, settings: Settings) -> "Neo4jClient":
        if settings.graph_mode == "off":
            raise GraphError("图谱已关闭（GRAPH_MODE=off）", "GRAPH_DISABLED")
        if not settings.neo4j_password:
            raise GraphError(
                "未配置 NEO4J_PASSWORD", "GRAPH_NOT_CONFIGURED"
            )
        try:
            try:
                from neo4j import GraphDatabase
            except ImportError as exc:
                raise GraphError(
                    "未安装 neo4j 驱动；图谱功能需要 pip install -e \".[graph]\"",
                    "GRAPH_NOT_INSTALLED",
                ) from exc
            try:
                driver = GraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_username, settings.neo4j_password),
                    notifications_min_severity="OFF",
                )
            except TypeError:  # 旧驱动没有该参数
                driver = GraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_username, settings.neo4j_password),
                )
            driver.verify_connectivity()
        except GraphError:
            raise
        except Exception as exc:  # 驱动异常类型随版本变化，统一包装
            raise GraphError(
                f"无法连接 Neo4j：{exc}", "GRAPH_UNAVAILABLE"
            ) from exc
        return cls(settings, driver)

    def _session(self):
        if self._driver is None:
            raise GraphError("Neo4j 驱动未初始化", "GRAPH_UNAVAILABLE")
        return self._driver.session(database=self._settings.neo4j_database)

    def ensure_schema(self) -> None:
        statements = (
            "CREATE CONSTRAINT paper_paper_id IF NOT EXISTS "
            "FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_chunk_id IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_entity_id IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT claim_claim_id IF NOT EXISTS "
            "FOR (c:Claim) REQUIRE c.claim_id IS UNIQUE",
        )
        try:
            with self._session() as session:
                for statement in statements:
                    session.run(statement).consume()
        except GraphError:
            raise
        except Exception as exc:
            raise GraphError(
                f"创建图约束失败：{exc}", "GRAPH_UNAVAILABLE"
            ) from exc

    def write_claim(self, claim: dict[str, Any]) -> None:
        self.write_claims([claim])

    def write_claims(self, claims: list[dict[str, Any]]) -> None:
        """批量写入关系声明：一条 UNWIND 语句替代逐条往返，显著缩短构图时间。"""
        if not claims:
            return
        query = """
        UNWIND $claims AS claim
        MERGE (p:Paper {paper_id: claim.paper_id})
          SET p.workspace_id = claim.workspace_id, p.revision = claim.revision,
              p.title = claim.title
        MERGE (c:Chunk {chunk_id: claim.chunk_id})
          SET c.workspace_id = claim.workspace_id, c.paper_id = claim.paper_id,
              c.revision = claim.revision, c.page_number = claim.page_number,
              c.text_hash = claim.text_hash
        MERGE (s:Entity {entity_id: claim.subject_id})
          ON CREATE SET s.workspace_id = claim.workspace_id,
                        s.entity_type = claim.subject_type,
                        s.canonical_name = claim.subject_name,
                        s.normalization_version = claim.normalization_version
        MERGE (o:Entity {entity_id: claim.object_id})
          ON CREATE SET o.workspace_id = claim.workspace_id,
                        o.entity_type = claim.object_type,
                        o.canonical_name = claim.object_name,
                        o.normalization_version = claim.normalization_version
        MERGE (cl:Claim {claim_id: claim.claim_id})
          ON CREATE SET cl.workspace_id = claim.workspace_id,
                        cl.paper_id = claim.paper_id,
                        cl.revision = claim.revision,
                        cl.relation_type = claim.relation_type,
                        cl.extractor_version = claim.extractor_version,
                        cl.qualifiers_json = claim.qualifiers_json
          SET cl.status = claim.status,
              cl.evidence_text = claim.evidence_text
        MERGE (cl)-[:SUBJECT]->(s)
        MERGE (cl)-[:OBJECT]->(o)
        MERGE (cl)-[:SUPPORTED_BY]->(c)
        MERGE (cl)-[:IN_PAPER]->(p)
        MERGE (p)-[:HAS_CHUNK]->(c)
        """
        try:
            with self._session() as session:
                session.execute_write(
                    lambda tx: tx.run(query, claims=claims).consume()
                )
        except Exception as exc:
            raise GraphError("图谱写入失败，请检查连接后重试", "GRAPH_WRITE_FAILED") from exc

    def query_template(
        self,
        query_type: str,
        *,
        workspace_id: str,
        paper_ids: list[str],
        entity_name: str | None = None,
        limit: int = 10,
        question: str | None = None,
    ) -> list[dict[str, Any]]:
        template = QUERY_TEMPLATES.get(query_type)
        if template is None:
            raise GraphError(
                f"未知查询模板：{query_type}", "GRAPH_BAD_QUERY_TYPE"
            )
        terms = []
        if question:
            from ..rag.keyword import tokenize
            stopwords = {"这篇", "论文", "什么", "哪些", "如何", "是否", "作者", "研究", "的", "了", "是", "what", "which", "the", "paper", "does", "how"}
            terms = [term for term in tokenize(question) if term not in stopwords][:20]
            if not terms:
                return []
        with self._session() as session:
            try:
                from neo4j import Query
            except ImportError as exc:
                raise GraphError(
                    "未安装 neo4j 驱动；图谱功能需要 pip install -e \".[graph]\"",
                    "GRAPH_NOT_INSTALLED",
                ) from exc
            result = session.run(
                Query(template, timeout=2.0),
                workspace_id=workspace_id,
                paper_ids=paper_ids,
                entity_name=entity_name,
                query_terms=terms,
                limit=limit,
            )
            return [dict(record) for record in result]

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def clear_paper_graph(self, workspace_id: str, paper_id: str) -> None:
        """重建前清空该论文的 Claim/Chunk/Paper（实体可能被其他论文共享，保留）。"""
        with self._session() as session:
            for statement in (
                "MATCH (cl:Claim {workspace_id: $ws, paper_id: $pid}) "
                "DETACH DELETE cl",
                "MATCH (c:Chunk {workspace_id: $ws, paper_id: $pid}) "
                "DETACH DELETE c",
                "MATCH (p:Paper {paper_id: $pid}) DETACH DELETE p",
            ):
                session.run(statement, ws=workspace_id, pid=paper_id).consume()

    def delete_workspace(self, workspace_id: str) -> int:
        """删除某工作区的全部图数据（测试/重置用）。"""
        with self._session() as session:
            result = session.run(
                "MATCH (n {workspace_id: $workspace_id}) DETACH DELETE n "
                "RETURN count(n) AS deleted",
                workspace_id=workspace_id,
            ).single()
            return int(result["deleted"]) if result else 0
