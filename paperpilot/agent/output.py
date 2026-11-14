from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnswerClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=4000)
    evidence_ids: list[str] = Field(min_length=1, max_length=20)


class FinalAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "insufficient_evidence", "needs_clarification"]
    claims: list[AnswerClaim] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def check_claims(self):
        if (self.status == "answered") != bool(self.claims):
            raise ValueError("answered 必须包含带引用的 claims；拒答或澄清必须为空 claims")
        return self


OUTPUT_INSTRUCTION = '''
最终回答必须只输出 JSON，不能输出自由文本：
{"status":"answered","claims":[{"text":"一条有依据的结论","evidence_ids":["ev_001"]}]}
每条事实结论必须单独放入 claims，并提供本次回读登记的 evidence_ids。
证据不足时输出 {"status":"insufficient_evidence","claims":[]}。
问题需要澄清时输出 {"status":"needs_clarification","claims":[]}。
如果问题询问的数字、比例、实验、调查结果或结论没有在证据中原样出现，
必须返回 insufficient_evidence；不得用相近主题或其他数字顶替。
例如问“满意度非常满意的百分比”，证据只提到评分权重时，应返回 insufficient_evidence。
不能通过删除引用来保留事实陈述；不要在 text 中手写引用编号。
'''
