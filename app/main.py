import re
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.services.llm_service import generate_answer
from app.services.question_service import (
    extract_employee_id,
    extract_employee_name,
)
from app.services.hybrid_search_service import (
    ACCESSIBLE_INDICES,
    build_context,
    filter_hits_with_answer_values,
    get_allowed_field_value,
    get_category_values,
    get_user_permission_level,
    make_sources,
    search_employees_by_conditions,
    search_employees_by_department_or_team,
    search_hybrid,
)
from app.services.query_policy_service import (
    FIELD_RULES,
    get_max_required_level,
    select_indices_by_fields,
    split_fields_by_permission,
)
from app.services.question_analyzer_service import (
    analyze_question_to_tasks,
    normalize_tasks,
)
from app.services.org_policy_service import (
    DEPARTMENTS,
    TEAMS,
    POSITIONS,
)


app = FastAPI(title="Durian HR RAG Chatbot")


class ChatRequest(BaseModel):
    question: str


class RagChatRequest(BaseModel):
    question: str
    employee_id: str


@app.get("/")
def root():
    return {"message": "Durian RAG API Running"}


@app.post("/chat")
def chat(request: ChatRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="질문을 입력해주세요.",
        )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:4b",
            "prompt": request.question,
            "stream": False,
        },
    )
    result = response.json()

    return {
        "question": request.question,
        "answer": result["response"],
        "model": "gemma3:4b",
    }


def unique_keep_order(items: list[str]) -> list[str]:
    """
    리스트 순서를 유지하면서 중복을 제거한다.
    """

    result = []

    for item in items:
        if item not in result:
            result.append(item)

    return result


def build_denied_message(denied_fields: list[str]) -> str:
    if not denied_fields:
        return ""

    labels = [
        FIELD_RULES.get(field, {}).get("label", field)
        for field in denied_fields
    ]

    return "접근 권한이 없어 제공할 수 없는 정보: " + ", ".join(labels)


def build_category_answer(field_key: str, permission_level: int) -> str:
    """
    부서/팀/직급/직책 목록 질문에 대한 답변.

    예:
    - 부서 종류 알려줘
    - 팀 목록 알려줘
    """

    values = get_category_values(field_key, permission_level)
    label = FIELD_RULES.get(field_key, {}).get("label", field_key)

    if not values:
        return f"조회 가능한 {label} 목록이 없습니다."

    return (
        f"{label} 목록은 다음과 같습니다.\n"
        + "\n".join([f"- {value}" for value in values])
    )


def normalize_task_org_values(task: dict) -> dict:
    """
    task 안의 department/team/position 값이 실제 목록에 있는지 검증한다.

    예:
    - department="채용팀"은 잘못된 값이므로 제거
    - team="채용팀"은 정상 값이므로 유지
    """

    normalized_task = task.copy()

    raw_department = normalized_task.get("department")
    raw_team = normalized_task.get("team")
    raw_position = normalized_task.get("position")

    normalized_task["department"] = (
        raw_department
        if raw_department in DEPARTMENTS
        else None
    )

    normalized_task["team"] = (
        raw_team
        if raw_team in TEAMS
        else None
    )

    normalized_task["position"] = (
        raw_position
        if raw_position in POSITIONS
        else None
    )

    return normalized_task


def sanitize_filters(filters: list[dict]) -> list[dict]:
    """
    LLM이 만든 filters를 main.py에서도 한 번 더 안전하게 검증한다.

    중요:
    - department 값은 DEPARTMENTS에 있어야 한다.
    - team 값은 TEAMS에 있어야 한다.
    - position 값은 POSITIONS에 있어야 한다.
    - FIELD_RULES에 없는 field는 버린다.
    """

    if not isinstance(filters, list):
        return []

    allowed_ops = {"eq", "contains", "gt", "gte", "lt", "lte", "between"}
    safe_filters = []

    for item in filters:
        if not isinstance(item, dict):
            continue

        field = item.get("field")
        op = item.get("op", "eq")
        value = item.get("value")

        if field not in FIELD_RULES:
            continue

        if value is None or value == "":
            continue

        if op not in allowed_ops:
            op = "eq"

        if field == "department" and value not in DEPARTMENTS:
            continue

        if field == "team" and value not in TEAMS:
            continue

        if field == "position" and value not in POSITIONS:
            continue

        safe_filters.append(
            {
                "field": field,
                "op": op,
                "value": value,
            }
        )

    return safe_filters


def add_filter_if_missing(
    filters: list[dict],
    field: str,
    value,
    op: str = "eq",
) -> list[dict]:
    """
    department/team/position 값이 task에는 있는데 filters에 없으면 추가한다.
    """

    if not value:
        return filters

    for item in filters:
        if item.get("field") == field and item.get("value") == value:
            return filters

    filters.append(
        {
            "field": field,
            "op": op,
            "value": value,
        }
    )

    return filters


def build_task_question(original_question: str, task: dict) -> str:
    """
    hybrid 검색에 사용할 질문을 만든다.

    기존 기능은 유지하되, 원문 질문과 filters 값을 함께 넣는다.

    예:
    - 원문: 채용팀의 계약직 알려줘
    - task_question: 채용팀의 계약직 알려줘 채용팀 계약직 직원
    """

    parts = [original_question]

    for key in ["employee_name", "employee_id", "department", "team", "position"]:
        value = task.get(key)

        if value and str(value) not in " ".join(parts):
            parts.append(str(value))

    filters = task.get("filters", [])

    if isinstance(filters, list):
        for item in filters:
            if not isinstance(item, dict):
                continue

            value = item.get("value")

            if value and str(value) not in " ".join(parts):
                parts.append(str(value))

    for field in task.get("target_fields", []):
        label = FIELD_RULES.get(field, {}).get("label")

        if label and label not in " ".join(parts):
            parts.append(label)

    return " ".join(parts)


def format_allowed_hits_answer(hits: list[dict], allowed_fields: list[str]) -> str:
    source_items = make_sources(hits, allowed_fields=allowed_fields)

    if not source_items:
        return "조회 가능한 정보가 없습니다."

    lines = []

    for item in source_items:
        values = []

        if "employee" in allowed_fields:
            employee_name = item.get("employee_name")
            employee_id = item.get("employee_id")
            employee_label = " / ".join(
                value
                for value in [employee_name, employee_id]
                if value
            )

            if employee_label:
                values.append(employee_label)

        for field in allowed_fields:
            if field == "employee":
                continue

            if item.get(field):
                label = FIELD_RULES.get(field, {}).get("label", field)
                values.append(f"{label}: {item[field]}")

        if values:
            lines.append("- " + " / ".join(values))

    return "\n".join(lines) if lines else "조회 가능한 정보가 없습니다."


def get_filter_fields(filters: list[dict]) -> list[str]:
    """
    filters에서 사용된 field 목록을 꺼낸다.
    """

    if not isinstance(filters, list):
        return []

    fields = []

    for item in filters:
        if not isinstance(item, dict):
            continue

        field = item.get("field")

        if field in FIELD_RULES:
            fields.append(field)

    return unique_keep_order(fields)


def to_number(value):
    """
    숫자 비교용 변환 함수.

    예:
    - "80점" -> 80
    - "45,000,000원" -> 45000000
    """

    if value is None:
        return None

    text = str(value).replace(",", "")
    match = re.search(r"-?\d+(\.\d+)?", text)

    if not match:
        return None

    number_text = match.group()

    if "." in number_text:
        return float(number_text)

    return int(number_text)


def filter_match(actual_value, op: str, expected_value) -> bool:
    """
    filter 조건 하나를 비교한다.
    """

    if actual_value is None or actual_value == "":
        return False

    actual_text = str(actual_value).strip()
    expected_text = str(expected_value).strip()

    if op == "eq":
        return actual_text == expected_text

    if op == "contains":
        return expected_text in actual_text

    if op in {"gt", "gte", "lt", "lte"}:
        actual_number = to_number(actual_value)
        expected_number = to_number(expected_value)

        if actual_number is None or expected_number is None:
            return False

        if op == "gt":
            return actual_number > expected_number

        if op == "gte":
            return actual_number >= expected_number

        if op == "lt":
            return actual_number < expected_number

        if op == "lte":
            return actual_number <= expected_number

    if op == "between":
        if not isinstance(expected_value, list) or len(expected_value) != 2:
            return False

        actual_number = to_number(actual_value)
        start_number = to_number(expected_value[0])
        end_number = to_number(expected_value[1])

        if actual_number is None or start_number is None or end_number is None:
            return False

        return start_number <= actual_number <= end_number

    return False


def filter_hits_by_filters(hits: list[dict], filters: list[dict]) -> list[dict]:
    """
    hybrid 검색 결과를 filters 조건으로 후처리한다.

    중요:
    - 직접조회가 아니다.
    - search_hybrid() 이후 결과를 employee_id 기준으로 묶어서 조건을 확인한다.
    - 같은 직원의 정보가 여러 chunk에 나뉘어 있어도 함께 판단한다.
    """

    if not filters:
        return hits

    grouped_hits = {}

    for hit in hits:
        source = hit.get("_source", {})
        employee_id = source.get("employee_id")

        if not employee_id:
            continue

        grouped_hits.setdefault(employee_id, []).append(hit)

    matched_employee_ids = set()

    for employee_id, employee_hits in grouped_hits.items():
        matched_all_filters = True

        for filter_item in filters:
            field = filter_item.get("field")
            op = filter_item.get("op", "eq")
            expected_value = filter_item.get("value")

            if field not in FIELD_RULES:
                continue

            matched_this_filter = False

            for hit in employee_hits:
                source = hit.get("_source", {})
                embedding_text = source.get("embedding_text", "")

                actual_value = get_allowed_field_value(
                    source=source,
                    embedding_text=embedding_text,
                    field_key=field,
                )

                if filter_match(
                    actual_value=actual_value,
                    op=op,
                    expected_value=expected_value,
                ):
                    matched_this_filter = True
                    break

            if not matched_this_filter:
                matched_all_filters = False
                break

        if matched_all_filters:
            matched_employee_ids.add(employee_id)

    return [
        hit
        for hit in hits
        if hit.get("_source", {}).get("employee_id") in matched_employee_ids
    ]


def has_non_org_filters(filters: list[dict]) -> bool:
    """
    department/team/position 외의 조건이 있는지 확인한다.

    예:
    - team=채용팀만 있으면 False
    - team=채용팀 + contract_type=계약직이면 True
    """

    org_fields = {"department", "team", "position"}

    for item in filters:
        if item.get("field") not in org_fields:
            return True

    return False


def process_task(
    task: dict,
    original_question: str,
    requester_employee_id: str,
    permission_level: int,
) -> dict:
    intent = task.get("intent", "unknown")

    # department/team/position 값 검증
    task = normalize_task_org_values(task)

    # filters 정리
    filters = sanitize_filters(task.get("filters", []))

    if task.get("department"):
        filters = add_filter_if_missing(
            filters=filters,
            field="department",
            op="eq",
            value=task.get("department"),
        )

    if task.get("team"):
        filters = add_filter_if_missing(
            filters=filters,
            field="team",
            op="eq",
            value=task.get("team"),
        )

    if task.get("position") and not task.get("employee_name"):
        filters = add_filter_if_missing(
            filters=filters,
            field="position",
            op="eq",
            value=task.get("position"),
        )

    task["filters"] = filters

    # LLM이 뽑은 target_fields 중 FIELD_RULES에 등록된 것만 사용한다.
    target_fields = [
        field
        for field in task.get("target_fields", [])
        if field in FIELD_RULES
    ]

    filter_fields = get_filter_fields(filters)

    if intent == "unknown" or not target_fields:
        return {
            "answer": "",
            "sources": [],
            "permission": {
                "allowed": False,
                "ignored": True,
                "reason": "분석 가능한 유효 필드가 없습니다.",
                "permission_level": permission_level,
                "required_level": 1,
                "allowed_fields": [],
                "denied_fields": [],
                "is_self": bool(task.get("is_self", False)),
            },
        }

    is_self = bool(task.get("is_self", False))

    # LLM이 직원 이름을 못 뽑은 경우를 대비한 보정
    # 예: "오민호 사원번호 알려줘" → employee_name이 null로 오는 경우
    if intent == "single_lookup" and not is_self:
        if not task.get("employee_name") and not task.get("employee_id"):
            guessed_name = extract_employee_name(original_question)

            if guessed_name:
                task["employee_name"] = guessed_name

    # target_fields + filters에 쓰인 필드까지 포함해서 required_level 계산
    search_related_fields = unique_keep_order(target_fields + filter_fields)

    required_level = (
        get_max_required_level(search_related_fields)
        if search_related_fields
        else 1
    )

    # 답변 필드 권한 판단
    answer_fields, denied_answer_fields = split_fields_by_permission(
        target_fields=target_fields,
        permission_level=permission_level,
        is_self=is_self,
    )

    # 조건 필드 권한 판단
    allowed_filter_fields, denied_filter_fields = split_fields_by_permission(
        target_fields=filter_fields,
        permission_level=permission_level,
        is_self=is_self,
    )

    denied_fields = unique_keep_order(denied_answer_fields + denied_filter_fields)
    denied_message = build_denied_message(denied_fields)

    # 조건 필드 권한이 없으면 검색 자체를 하지 않는다.
    if denied_filter_fields:
        return {
            "answer": denied_message,
            "sources": [],
            "permission": {
                "allowed": False,
                "permission_level": permission_level,
                "required_level": required_level,
                "allowed_fields": answer_fields,
                "denied_fields": denied_fields,
                "is_self": is_self,
            },
        }

    # 사용자가 요청한 답변 필드 중 허용된 것이 하나도 없으면 검색하지 않는다.
    if not answer_fields:
        return {
            "answer": denied_message or "조회 가능한 필드가 없습니다.",
            "sources": [],
            "permission": {
                "allowed": False,
                "permission_level": permission_level,
                "required_level": required_level,
                "allowed_fields": [],
                "denied_fields": denied_fields,
                "is_self": is_self,
            },
        }

    # allowed_fields = 최종 답변 표시용 필드
    # context_fields = 조건 검증/LLM context/sources에 넣을 필드
    allowed_fields = list(answer_fields)
    context_fields = unique_keep_order(answer_fields + allowed_filter_fields)

    if intent in {"single_lookup", "employee_list", "condition_search"}:
        if "employee" not in allowed_fields:
            allowed_fields.insert(0, "employee")

        if "employee" not in context_fields:
            context_fields.insert(0, "employee")

    # 기본 검색 권한은 요청자의 실제 권한
    search_permission_level = permission_level

    # 본인 질문이면 본인 정보 조회를 허용하기 위해 필요한 레벨까지 검색 범위를 올린다.
    # 단, 아래 search_employee_id에서 반드시 requester_employee_id로 필터를 건다.
    if is_self:
        search_permission_level = max(permission_level, required_level)

    accessible_indices = ACCESSIBLE_INDICES.get(search_permission_level, [])

    # 중요:
    # 인덱스 선택은 answer_fields + allowed_filter_fields 기준으로 한다.
    # 예: target_fields=["employee"], filters=[contract_type]이면 contract_type 인덱스도 포함해야 한다.
    search_fields = unique_keep_order(answer_fields + allowed_filter_fields)

    indices = select_indices_by_fields(
        accessible_indices=accessible_indices,
        allowed_fields=search_fields,
    )

    search_employee_id = None

    # 본인 질문이면 무조건 요청자 employee_id로 검색한다.
    if is_self:
        search_employee_id = requester_employee_id

    # 타인 질문에서 employee_id가 직접 들어온 경우
    elif task.get("employee_id"):
        search_employee_id = str(task["employee_id"]).strip().upper()

    # 기존 정규식 추출 보조
    else:
        search_employee_id = extract_employee_id(original_question)

    task_question = build_task_question(original_question, task)

    # =========================
    # 1. 카테고리 목록 질문
    # 예: 부서 종류 알려줘, 팀 목록 알려줘
    # → hybrid 검색보다 직접 집계가 정확하다.
    # =========================
    if intent == "category_list":
        answers = [
            build_category_answer(field, search_permission_level)
            for field in answer_fields
            if field in {"department", "team", "job_grade", "position"}
        ]

        answer = (
            "\n\n".join(answers)
            if answers
            else "카테고리 필드를 찾을 수 없습니다."
        )

        if denied_message:
            answer = f"{answer}\n\n{denied_message}"

        return {
            "answer": answer,
            "sources": [],
            "permission": {
                "allowed": True,
                "permission_level": permission_level,
                "required_level": required_level,
                "allowed_fields": answer_fields,
                "denied_fields": denied_fields,
                "is_self": is_self,
            },
        }

    search_hits = []

    # =========================
    # 2. 직원 목록 질문
    # 예: 개발부 직원 모두 알려줘
    # → 부서/팀/직책만 있는 단순 직원 목록은 기존 직접 조회 유지
    # → 단, 계약형태/성과점수/연봉 등 추가 조건이 있으면 hybrid로 보낸다.
    # =========================
    if (
        intent == "employee_list"
        and (task.get("department") or task.get("team") or task.get("position"))
        and set(answer_fields) == {"employee"}
        and not has_non_org_filters(filters)
    ):
        search_hits = search_employees_by_conditions(
            permission_level=search_permission_level,
            department=task.get("department"),
            team=task.get("team"),
            position=task.get("position"),
            size=50,
        )

        answer = format_allowed_hits_answer(
            hits=search_hits,
            allowed_fields=allowed_fields,
        )

    # =========================
    # 3. 단일 조회 / 조건 검색
    # 예:
    # - 김민수 부서 알려줘
    # - 내 연봉 알려줘
    # - 채용팀의 계약직 알려줘
    # - 성과점수 80점 이상 직원 알려줘
    # → 기존 hybrid 검색 사용
    # =========================
    else:
        search_hits = search_hybrid(
            question=task_question,
            permission_level=search_permission_level,
            employee_id=search_employee_id,
            size=50,
            indices=indices,
        )

        search_hits = filter_hits_by_filters(
            hits=search_hits,
            filters=filters,
        )

        search_hits = filter_hits_with_answer_values(
            hits=search_hits,
            answer_fields=answer_fields,
        )

        if not search_hits:
            answer = "조회 가능한 정보가 없습니다."

        # 조건 검색/직원 목록형 결과는 LLM에게 긴 목록 생성을 맡기지 않고 직접 포맷한다.
        elif intent in {"condition_search", "employee_list"} and filters:
            answer = format_allowed_hits_answer(
                hits=search_hits,
                allowed_fields=allowed_fields,
            )

        # 부서/팀 조건 + 특정 필드 조회도 직접 포맷한다.
        # 예: "마케팅부 직원들 연봉 알려줘"
        elif intent == "condition_search" and (task.get("department") or task.get("team")):
            answer = format_allowed_hits_answer(
                hits=search_hits,
                allowed_fields=allowed_fields,
            )

        else:
            context = build_context(
                search_hits,
                task_question,
                allowed_fields=context_fields,
            )

            answer = generate_answer(
                question=task_question,
                context=context,
            )

    if denied_message:
        answer = f"{answer}\n\n{denied_message}"

    return {
        "answer": answer,
        "sources": make_sources(
            search_hits,
            allowed_fields=context_fields,
        ),
        "permission": {
            "allowed": True,
            "permission_level": permission_level,
            "required_level": required_level,
            "allowed_fields": answer_fields,
            "denied_fields": denied_fields,
            "is_self": is_self,
        },
    }


@app.post("/rag-chat")
def rag_chat(request: RagChatRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="질문을 입력해주세요.",
        )

    if not request.employee_id or not request.employee_id.strip():
        raise HTTPException(
            status_code=400,
            detail="employee_id를 입력해주세요.",
        )

    employee_id = request.employee_id.strip().upper()
    permission_level = get_user_permission_level(employee_id)

    if permission_level is None:
        raise HTTPException(
            status_code=404,
            detail="요청자 사번을 찾을 수 없습니다.",
        )

    if permission_level not in [1, 2, 3]:
        raise HTTPException(
            status_code=400,
            detail="계산된 permission_level이 유효하지 않습니다.",
        )

    analysis = analyze_question_to_tasks(request.question)
    tasks = normalize_tasks(analysis, request.question)

    task_results = [
        process_task(
            task=task,
            original_question=request.question,
            requester_employee_id=employee_id,
            permission_level=permission_level,
        )
        for task in tasks
    ]

    answers = [
        result["answer"]
        for result in task_results
        if result.get("answer")
    ]

    sources = []
    for result in task_results:
        sources.extend(result.get("sources", []))

    return {
        "success": any(
            result.get("permission", {}).get("allowed")
            for result in task_results
        ),
        "answer": "\n\n".join(answers),
        "permission": {
            "employee_id": employee_id,
            "permission_level": permission_level,
            "tasks": [
                result.get("permission", {})
                for result in task_results
            ],
        },
        "tasks": tasks,
        "sources": sources,
        "model_type": "gemma3:4b",
    }