"""多轮对话：把追问改写成独立问题。

历史只用于理解指代，不作为证据；改写失败时回退原问题。
"""

from __future__ import annotations

from typing import Sequence

from .llm import ChatMessage, ChatProvider

SYSTEM_PROMPT = """你是检索查询改写器。根据最近几轮对话，把用户的新问题改写成
一个不依赖上下文、可独立检索的完整问题。只输出改写后的问题，不要解释。
如果新问题本身已经完整，原样返回。"""


def rewrite_followup(
    chat: ChatProvider,
    question: str,
    history: Sequence[dict],
    *,
    max_turns: int = 3,
) -> str:
    """history 元素形如 {"question": ..., "answer": ...}。失败返回原问题。"""
    recent = [turn for turn in history if turn.get("question")][-max_turns:]
    if not recent:
        return question

    lines = []
    for turn in recent:
        answer = str(turn.get("answer") or "").strip()[:400]
        lines.append(f"用户：{turn['question']}")
        if answer:
            lines.append(f"回答摘要：{answer}")
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content="\n".join(lines)
            + f"\n\n新问题：{question}\n\n改写后的问题：",
        ),
    ]
    try:
        response = chat.chat(messages)
    except Exception:
        return question
    rewritten = (response.message.content or "").strip().splitlines()
    candidate = rewritten[0].strip().strip('"“”') if rewritten else ""
    return candidate or question
