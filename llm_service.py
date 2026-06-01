from __future__ import annotations

import os


SYSTEM_PROMPT = """
당신은 상명대학교 공식 안내를 돕는 SMU Talk 챗봇입니다.

답변 원칙:
- 반드시 제공된 [검색 결과] 안의 정보만 근거로 답하세요.
- 검색 결과에 없는 날짜, 신청 기간, 담당 부서, 링크, 규정은 만들지 마세요.
- 검색 결과가 질문과 맞지 않으면 "제공된 자료만으로는 정확히 확인하기 어렵습니다"라고 말하세요.
- 사용자가 메뉴 항목을 물으면 해당 항목의 의미, 확인 경로, 주의할 점을 간결하게 정리하세요.
- 답변은 한국어로 자연스럽고 짧게 작성하세요.
- 내부 검색 방식, RAG, FTS, Chroma, Qwen, OpenAI 같은 구현 명칭은 사용자에게 말하지 마세요.
""".strip()


def _enabled() -> bool:
    value = os.environ.get("SMU_USE_OPENAI", "auto").strip().lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    return bool(os.environ.get("OPENAI_API_KEY"))


def _trim_context(context: str) -> str:
    limit = int(os.environ.get("SMU_OPENAI_MAX_CONTEXT_CHARS", "9000"))
    text = context.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n[검색 결과 일부 생략]"


def generate_answer(question: str, rag_context: str) -> str | None:
    if not _enabled() or not os.environ.get("OPENAI_API_KEY"):
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    model = os.environ.get("SMU_OPENAI_MODEL", "gpt-5.4-mini")
    timeout = float(os.environ.get("SMU_OPENAI_TIMEOUT_SECONDS", "20"))
    client = OpenAI(timeout=timeout)

    prompt = f"""
[사용자 질문]
{question}

[검색 결과]
{_trim_context(rag_context)}

[작성 요청]
검색 결과를 바탕으로 사용자가 바로 이해할 수 있게 답변하세요.
관련 링크가 검색 결과에 있으면 함께 안내하세요.
검색 결과가 질문과 어긋나면 억지로 답하지 말고, 공식 홈페이지의 해당 메뉴를 확인하라고 안내하세요.
""".strip()

    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception:
        return None

    text = getattr(response, "output_text", None)
    if not text:
        return None
    return str(text).strip() or None
