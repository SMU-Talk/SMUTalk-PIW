from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_service import get_rag_service


def main() -> None:
    rag = get_rag_service()
    print("RAG status:")
    for key, value in rag.status().items():
        print(f"- {key}: {value}")

    question = "장학금 신청은 어디서 확인하나요?"
    print("\nQuestion:", question)
    print("\nAnswer:\n")
    print(rag.answer(question) or "검색 결과가 없습니다.")


if __name__ == "__main__":
    main()
