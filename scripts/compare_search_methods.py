import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가하여 app 패키지를 찾을 수 있도록 한다.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.hybrid_search_service import (
    search_bm25,
    search_vector,
    search_hybrid,
    create_question_vector,
    select_search_indices,
    get_doc_types_by_keywords,
    get_user_permission_level,
)


def format_hit(hit):
    source = hit.get("_source", {})
    return (
        f"index={hit['_index']} id={hit['_id']} score={hit.get('_score')} "
        f"employee_id={source.get('employee_id')} department={source.get('department')} "
        f"position={source.get('position')} name={source.get('employee_name')} "
        f"text={source.get('embedding_text', '')[:120]}"
    )


def print_hits(title, hits):
    print(f"\n=== {title} ({len(hits)}) ===")
    for idx, hit in enumerate(hits, start=1):
        print(f"{idx}. {format_hit(hit)}")


def compare_search(question, permission_level, employee_id=None, size=5):
    print("Question:", question)
    print("Permission level:", permission_level)
    print("Employee ID:", employee_id)
    print("Size:", size)

    print("Doc types from question:", get_doc_types_by_keywords(question))
    print("Selected indices:", select_search_indices(question, permission_level))

    if employee_id:
        print("Employee permission check:", get_user_permission_level(employee_id))

    bm25_hits = search_bm25(
        question=question,
        permission_level=permission_level,
        employee_id=employee_id,
        size=size,
    )

    vector_hits = search_vector(
        question=question,
        question_vector=create_question_vector(question),
        permission_level=permission_level,
        employee_id=employee_id,
        size=size,
    )

    hybrid_hits = search_hybrid(
        question=question,
        permission_level=permission_level,
        employee_id=employee_id,
        size=size,
    )

    print_hits("BM25 결과", bm25_hits)
    print_hits("벡터 검색 결과", vector_hits)
    print_hits("하이브리드 결과", hybrid_hits)

    bm25_keys = {f"{hit['_index']}::{hit['_id']}" for hit in bm25_hits}
    vector_keys = {f"{hit['_index']}::{hit['_id']}" for hit in vector_hits}
    hybrid_keys = {f"{hit['_index']}::{hit['_id']}" for hit in hybrid_hits}

    print("\n공통 문서(BM25 ∩ Vector):", len(bm25_keys & vector_keys))
    print("하이브리드에 포함된 BM25 문서:", len(hybrid_keys & bm25_keys))
    print("하이브리드에 포함된 Vector 문서:", len(hybrid_keys & vector_keys))

    print("\nBM25 only:", bm25_keys - vector_keys)
    print("Vector only:", vector_keys - bm25_keys)
    print("Hybrid only:", hybrid_keys - (bm25_keys | vector_keys))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare BM25, vector, and hybrid search results.")
    parser.add_argument("--question", required=True, help="질문 텍스트")
    parser.add_argument("--permission_level", type=int, required=True, help="검색 권한 레벨(1,2,3)")
    parser.add_argument("--employee_id", default=None, help="본인 조회 시 사용할 사번(선택)")
    parser.add_argument("--size", type=int, default=5, help="결과 개수")
    args = parser.parse_args()

    compare_search(
        question=args.question,
        permission_level=args.permission_level,
        employee_id=args.employee_id,
        size=args.size,
    )
