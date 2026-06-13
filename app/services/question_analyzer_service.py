import json
import re
import requests

from app.services.query_policy_service import (
    ALLOWED_FIELDS,
    ALLOWED_INTENTS,
    FIELD_RULES,
)

from app.services.org_policy_service import (
    DEPARTMENTS,
    TEAMS,
    POSITIONS,
)


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"


DEFAULT_TASK = {
    "intent": "unknown",
    "target_fields": ["unknown"],
    "employee_name": None,
    "employee_id": None,
    "department": None,
    "team": None,
    "position": None,
    "filters": [],
    "is_self": False,
}


ALLOWED_FILTER_OPS = {
    "eq",
    "contains",
    "gt",
    "gte",
    "lt",
    "lte",
    "between",
}


def build_field_schema_text() -> str:
    """
    FIELD_RULES 기준으로 LLM에게 보여줄 HR 필드 목록을 만든다.

    이제 question_analyzer_service.py에 필드 목록을 직접 길게 쓰지 않는다.
    app/services/query_policy_service.py의 FIELD_RULES만 관리하면 된다.
    """

    lines = []

    for field_key, rule in FIELD_RULES.items():
        label = rule.get("label", field_key)
        lines.append(f"- {field_key}: {label}")

    lines.append("- unknown: 알 수 없는 필드")

    return "\n".join(lines)


def clean_json_text(text: str) -> str:
    """
    LLM 응답에서 JSON 부분만 추출한다.
    """

    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and start < end:
        text = text[start : end + 1]

    return text


def compact_text(text: str) -> str:
    """
    질문 비교용으로 공백을 제거한다.
    예: "채용팀의 계약직" -> "채용팀의계약직"
    """

    if not text:
        return ""

    return re.sub(r"\s+", "", text)


def find_known_value(question: str, values: list[str]) -> str | None:
    """
    질문 안에서 미리 정의된 부서/팀/직책 값을 찾는다.
    이건 조건 하드코딩이 아니라 조직 기준값 보정이다.
    """

    compact_question = compact_text(question)

    for value in values:
        if value in compact_question:
            return value

    return None


ORG_ALIAS_RULES = [
    {
        "keywords": ["영업", "영업관련", "영업쪽", "영업담당"],
        "field": "department",
        "value": "영업부",
    },
    {
        "keywords": ["개발", "개발관련", "개발쪽", "개발담당"],
        "field": "department",
        "value": "개발부",
    },
    {
        "keywords": ["인사", "인사관련", "인사쪽", "인사담당"],
        "field": "department",
        "value": "인사부",
    },
    {
        "keywords": ["기획", "기획관련", "기획쪽", "기획담당"],
        "field": "department",
        "value": "기획부",
    },
    {
        "keywords": ["마케팅", "마케팅관련", "마케팅쪽", "마케팅담당"],
        "field": "department",
        "value": "마케팅부",
    },
    {
        "keywords": ["채용", "채용관련", "채용쪽", "채용담당"],
        "field": "team",
        "value": "채용팀",
    },
]


def find_org_alias(question: str) -> tuple[str, str] | None:
    """
    질문의 조직 별칭을 실제 부서/팀 값으로 보정한다.

    예:
    - "영업 관련 직원" -> ("department", "영업부")
    - "채용 계약직" -> ("team", "채용팀")

    조건 if문을 늘리는 용도가 아니라, 조직 기준값 사전 보정만 담당한다.
    """

    compact_question = compact_text(question)

    for rule in ORG_ALIAS_RULES:
        if any(keyword in compact_question for keyword in rule["keywords"]):
            return rule["field"], rule["value"]

    return None


def is_org_alias_text(text: str | None) -> bool:
    """
    employee_name 자리에 들어오면 안 되는 조직/별칭 표현인지 확인한다.
    """

    if not text:
        return False

    compact_value = compact_text(str(text))

    if compact_value in DEPARTMENTS or compact_value in TEAMS or compact_value in POSITIONS:
        return True

    if compact_value.endswith("관련"):
        return True

    for rule in ORG_ALIAS_RULES:
        if compact_value in rule["keywords"]:
            return True

    return False


EVALUATION_GRADE_WORDS = {
    "우수": "A",
    "탁월": "A",
    "최상": "A",
    "좋음": "A",
    "양호": "B",
    "보통": "C",
    "미흡": "D",
    "부진": "D",
}


def get_evaluation_field_from_question(question: str) -> str:
    """
    평가 연도가 질문에 있으면 해당 연도 필드를, 없으면 최신 평가 필드를 사용한다.
    """

    for year in ["2020", "2021", "2022", "2023", "2024"]:
        if year in question:
            return f"evaluation_{year}"

    return "evaluation_2024"


def normalize_semantic_filters(filters: list[dict], question: str) -> list[dict]:
    """
    LLM이 자주 헷갈리는 의미 필터를 보정한다.

    예:
    - "평가가 우수한"은 최신 인사고과_2024 eq "A"다.
    - "2023년 평가가 우수한"은 evaluation_2023 eq "A"다.
    - "성과점수 80점 이상"처럼 숫자와 점수가 명시된 경우는 기존 숫자 조건을 유지한다.
    """

    if not filters:
        return filters

    compact_question = compact_text(question)
    evaluation_grade = None

    for word, grade in EVALUATION_GRADE_WORDS.items():
        if word in compact_question:
            evaluation_grade = grade
            break

    if not evaluation_grade:
        return filters

    normalized_filters = []
    has_evaluation_filter = False
    evaluation_field = get_evaluation_field_from_question(compact_question)
    evaluation_fields = {
        "evaluation",
        "evaluation_2020",
        "evaluation_2021",
        "evaluation_2022",
        "evaluation_2023",
        "evaluation_2024",
    }

    for item in filters:
        if not isinstance(item, dict):
            continue

        field = item.get("field")

        if field in evaluation_fields:
            has_evaluation_filter = True
            normalized_filters.append(
                {
                    "field": field if field != "evaluation" else evaluation_field,
                    "op": "eq",
                    "value": evaluation_grade,
                }
            )
            continue

        normalized_filters.append(item)

    if not has_evaluation_filter and ("평가" in compact_question or "고과" in compact_question):
        normalized_filters.append(
            {
                "field": evaluation_field,
                "op": "eq",
                "value": evaluation_grade,
            }
        )

    return normalized_filters


def normalize_target_fields_by_question(
    target_fields: list[str],
    question: str,
) -> list[str]:
    """
    LLM이 비슷한 식별자 필드를 헷갈린 경우 질문 원문 기준으로 보정한다.

    예:
    - "주민번호 알려줘"를 employee_id로 잘못 분석하면 rrn으로 바꾼다.
    - "사번 알려줘"는 employee_id 그대로 둔다.
    """

    compact_question = compact_text(question)

    if "주민등록번호" in compact_question or "주민번호" in compact_question:
        return [
            "rrn" if field == "employee_id" else field
            for field in target_fields
        ]

    return target_fields


def normalize_filters(raw_filters) -> list[dict]:
    """
    LLM이 반환한 filters를 안전한 형태로 정리한다.

    허용 예:
    [
        {"field": "team", "op": "eq", "value": "채용팀"},
        {"field": "contract_type", "op": "eq", "value": "계약직"}
    ]

    예전 호환용으로 dict도 지원한다.
    예:
    {"team": "채용팀", "contract_type": "계약직"}
    """

    normalized_filters = []

    if isinstance(raw_filters, dict):
        raw_filters = [
            {
                "field": field,
                "op": "eq",
                "value": value,
            }
            for field, value in raw_filters.items()
        ]

    if not isinstance(raw_filters, list):
        return []

    for item in raw_filters:
        if not isinstance(item, dict):
            continue

        field = item.get("field")
        op = item.get("op", "eq")
        value = item.get("value")

        if field not in FIELD_RULES:
            continue

        if op not in ALLOWED_FILTER_OPS:
            op = "eq"

        if value is None or value == "":
            continue

         # 부서/팀/직책 값 검증
        if field == "department" and value not in DEPARTMENTS:
            continue

        if field == "team" and value not in TEAMS:
            continue

        if field == "position" and value not in POSITIONS:
            continue

        normalized_filters.append(
            {
                "field": field,
                "op": op,
                "value": value,
            }
        )

    return normalized_filters


def add_filter_if_missing(
    filters: list[dict],
    field: str,
    value,
    op: str = "eq",
) -> list[dict]:
    """
    같은 field/value 조건이 없으면 filters에 추가한다.
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


def analyze_question_to_tasks(question: str) -> dict:
    field_schema_text = build_field_schema_text()

    department_list_text = ", ".join(DEPARTMENTS)
    team_list_text = ", ".join(TEAMS)
    position_list_text = ", ".join(POSITIONS)

    prompt = f"""
너는 HR RAG 챗봇의 질문 분석기입니다.

너의 역할은 사용자의 질문을 하나 이상의 task 객체로 나누는 것입니다.
권한 판단은 하지 마세요.
사용자에게 답변하지 마세요.
반드시 JSON만 반환하세요.

사용 가능한 intent 값:
- single_lookup: 특정 직원 또는 본인의 특정 정보를 묻는 질문
- employee_list: 특정 부서, 팀, 직책에 해당하는 직원 목록을 묻는 질문
- category_list: 부서, 팀, 직급, 직책 같은 종류/목록을 묻는 질문
- condition_search: 조건에 맞는 직원을 찾거나, 조건에 맞는 직원의 특정 정보를 묻는 질문
- unknown: 분류하기 어려운 질문

사용 가능한 HR 필드:
{field_schema_text}

부서 리스트:
{department_list_text}

팀 리스트:
{team_list_text}

직책 리스트:
{position_list_text}

[가장 중요한 개념]

target_fields와 filters를 반드시 구분하세요.

target_fields:
- 사용자가 답변으로 보고 싶은 정보입니다.
- 예: "오민호 이메일 알려줘" -> target_fields는 ["email"]
- 예: "채용팀의 계약직 알려줘" -> 답변으로 보고 싶은 것은 직원이므로 target_fields는 ["employee"]

filters:
- 검색 조건입니다.
- 예: "채용팀의 계약직 알려줘"
  - team = 채용팀
  - contract_type = 계약직
- 예: "마케팅부 정규직 이메일 알려줘"
  - department = 마케팅부
  - contract_type = 정규직
  - 답변으로 보고 싶은 것은 이메일

[반환 형식]

반드시 아래 JSON 형식 그대로 반환하세요.

{{
  "tasks": [
    {{
      "intent": "condition_search",
      "target_fields": ["employee"],
      "employee_name": null,
      "employee_id": null,
      "department": null,
      "team": null,
      "position": null,
      "filters": [
        {{
          "field": "team",
          "op": "eq",
          "value": "채용팀"
        }},
        {{
          "field": "contract_type",
          "op": "eq",
          "value": "계약직"
        }}
      ],
      "is_self": false
    }}
  ]
}}

[공통 규칙]

- 권한 판단은 절대 하지 마세요.
- 사용자가 직접 요청한 답변 필드만 target_fields에 넣으세요.
- 검색 조건으로 쓰인 필드는 filters에 넣으세요.
- filters의 field는 반드시 사용 가능한 HR 필드 중 하나를 사용하세요.
- filters의 op는 eq, contains, gt, gte, lt, lte, between 중 하나를 사용하세요.
- 정확히 일치하는 조건은 op를 eq로 설정하세요.
- 포함 조건은 op를 contains로 설정하세요.
- 이상 조건은 gte, 초과 조건은 gt를 사용하세요.
- 이하 조건은 lte, 미만 조건은 lt를 사용하세요.
- "성과점수 80점 이상"처럼 숫자 점수를 말한 경우에만 performance_score와 gte/lte/gt/lt를 사용하세요.
- "평가가 우수한", "고과가 우수한"처럼 평가 등급을 말한 경우에는 최신 평가인 evaluation_2024를 사용하세요.
- 평가 등급 표현은 실제 데이터 등급으로 바꾸세요. 우수/탁월/좋음은 A, 양호는 B, 보통은 C, 미흡/부진은 D입니다.
- "2023년 평가가 우수한"처럼 연도가 있으면 evaluation_2023 eq "A"처럼 해당 연도 필드를 사용하세요.
- 숫자가 없는 "우수한 평가"를 임의로 80점 이상으로 바꾸지 마세요.
- 본인 정보를 묻는 경우 is_self를 true로 설정하세요.
- 직원 이름이 있으면 employee_name에 넣으세요.
- EMP0001 같은 실제 사번이 있으면 employee_id에 넣으세요.
- 사원번호/사번을 물으면 target_fields는 employee_id입니다.
- 주민등록번호/주민번호를 물으면 target_fields는 rrn입니다.
- 주민등록번호/주민번호를 employee_id로 분석하지 마세요.
- 실제 부서 목록 중 하나가 포함되면 department에도 넣고 filters에도 넣으세요.
- 실제 팀 목록 중 하나가 포함되면 team에도 넣고 filters에도 넣으세요.
- 실제 직책 목록 중 하나가 조건으로 포함되면 position에도 넣고 filters에도 넣으세요.
- "영업 관련", "영업쪽", "영업 담당"처럼 조직 별칭 표현은 영업부 조건으로 분석하세요.
- "채용 관련", "채용쪽", "채용 담당"처럼 팀 별칭 표현은 채용팀 조건으로 분석하세요.
- "팀"과 "팀장/팀원"은 다릅니다. "팀장", "팀원"은 team이 아니라 position입니다.
- 사용자가 여러 개의 독립된 질문을 하면 task를 여러 개로 나누세요.

[분류 규칙]

1. 특정 직원 또는 본인의 정보를 묻는 질문
- intent는 single_lookup입니다.
- 예: "오민호 이메일 알려줘"
- 예: "내 입사일 알려줘"

2. 직원 목록을 묻는 질문
- intent는 employee_list 또는 condition_search입니다.
- 단순 부서/팀/직책 직원 목록이면 employee_list입니다.
- 조건이 붙으면 condition_search입니다.

3. 조건 검색
- "계약직", "정규직", "성과점수 80점 이상", "TOEIC 900점 이상"처럼 조건이 있으면 filters에 넣으세요.
- "평가가 우수한", "고과가 우수한"처럼 평가 등급 조건이 있으면 evaluation_2024 eq "A"를 filters에 넣으세요.
- 조건에 맞는 직원을 묻는 질문이면 target_fields는 ["employee"]입니다.
- 조건에 맞는 직원의 이메일/연봉/주소 등을 묻는 질문이면 target_fields는 해당 필드입니다.

4. 목록/종류 질문
- "종류", "목록", "리스트", "전체 부서", "어떤 부서", "부서 뭐 있어"처럼 실제 목록을 묻는 경우만 category_list입니다.
- "인사부 알려줘"처럼 실제 부서명만 말하고 알려달라는 경우는 category_list가 아닙니다.
- 이 경우는 employee_list 또는 condition_search입니다.

[필드 매핑 예시]

- 사원번호, 사번 -> employee_id
- 이름 -> employee_name
- 성별 -> gender
- 나이 -> age
- 생년월일 -> birth_date
- 주민등록번호, 주민번호 -> rrn
- 병역 -> military
- 입사일 -> hire_date
- 근속기간 -> tenure
- 학력 -> education
- 출신대학 -> university
- 학점 -> gpa
- 채용경로 -> hire_path
- 계약형태, 계약직, 정규직 -> contract_type
- 이전직장명 -> previous_company
- 이전최종직급 -> previous_job_grade
- 이전담당업무 -> previous_task
- 회사명 -> company
- 사업장위치 -> work_location
- 부서 -> department
- 팀 -> team
- 부서레벨 -> department_level
- 직급 -> job_grade
- 직책 -> position
- 직급레벨 -> job_grade_level
- 퇴직구분 -> retirement_type
- 퇴직일자 -> retirement_date
- 이메일, 메일 -> email
- 전화번호, 연락처 -> phone
- 주소 -> address
- 연봉 -> salary
- 잔업시간 -> overtime
- 미사용휴가일수 -> unused_vacation
- 급여은행 -> salary_bank
- 계좌번호 -> account
- 4대보험가입여부 -> insurance
- 성과점수 -> performance_score
- 평가, 인사평가, 고과, 인사고과 -> evaluation_2024
- 2020년 평가 -> evaluation_2020
- 2021년 평가 -> evaluation_2021
- 2022년 평가 -> evaluation_2022
- 2023년 평가 -> evaluation_2023
- 2024년 평가 -> evaluation_2024
- 자격증 -> certificate
- TOEIC, 토익 -> toeic
- 자격증수당여부 -> certificate_allowance
- 포상이력 -> award_history
- 징계이력 -> disciplinary_history
- 징계사유 -> disciplinary_reason

[예시 1]
질문: "오민호 이메일 알려줘"
답변:
{{
  "tasks": [
    {{
      "intent": "single_lookup",
      "target_fields": ["email"],
      "employee_name": "오민호",
      "employee_id": null,
      "department": null,
      "team": null,
      "position": null,
      "filters": [],
      "is_self": false
    }}
  ]
}}

[예시 2]
질문: "내 주소랑 전화번호 알려줘"
답변:
{{
  "tasks": [
    {{
      "intent": "single_lookup",
      "target_fields": ["address", "phone"],
      "employee_name": null,
      "employee_id": null,
      "department": null,
      "team": null,
      "position": null,
      "filters": [],
      "is_self": true
    }}
  ]
}}

[예시 3]
질문: "채용팀의 계약직 알려줘"
답변:
{{
  "tasks": [
    {{
      "intent": "condition_search",
      "target_fields": ["employee"],
      "employee_name": null,
      "employee_id": null,
      "department": null,
      "team": "채용팀",
      "position": null,
      "filters": [
        {{
          "field": "team",
          "op": "eq",
          "value": "채용팀"
        }},
        {{
          "field": "contract_type",
          "op": "eq",
          "value": "계약직"
        }}
      ],
      "is_self": false
    }}
  ]
}}

[예시 4]
질문: "마케팅부 정규직 이메일 알려줘"
답변:
{{
  "tasks": [
    {{
      "intent": "condition_search",
      "target_fields": ["email"],
      "employee_name": null,
      "employee_id": null,
      "department": "마케팅부",
      "team": null,
      "position": null,
      "filters": [
        {{
          "field": "department",
          "op": "eq",
          "value": "마케팅부"
        }},
        {{
          "field": "contract_type",
          "op": "eq",
          "value": "정규직"
        }}
      ],
      "is_self": false
    }}
  ]
}}

[예시 5]
질문: "성과점수 80점 이상 직원 알려줘"
답변:
{{
  "tasks": [
    {{
      "intent": "condition_search",
      "target_fields": ["employee"],
      "employee_name": null,
      "employee_id": null,
      "department": null,
      "team": null,
      "position": null,
      "filters": [
        {{
          "field": "performance_score",
          "op": "gte",
          "value": 80
        }}
      ],
      "is_self": false
    }}
  ]
}}

[예시 6]
질문: "평가가 우수한 계약직 알려줘"
답변:
{{
  "tasks": [
    {{
      "intent": "condition_search",
      "target_fields": ["employee"],
      "employee_name": null,
      "employee_id": null,
      "department": null,
      "team": null,
      "position": null,
      "filters": [
        {{
          "field": "evaluation_2024",
          "op": "eq",
          "value": "A"
        }},
        {{
          "field": "contract_type",
          "op": "eq",
          "value": "계약직"
        }}
      ],
      "is_self": false
    }}
  ]
}}

[예시 7]
질문: "부서 종류 알려줘"
답변:
{{
  "tasks": [
    {{
      "intent": "category_list",
      "target_fields": ["department"],
      "employee_name": null,
      "employee_id": null,
      "department": null,
      "team": null,
      "position": null,
      "filters": [],
      "is_self": false
    }}
  ]
}}

[예시 8]
질문: "인사부 알려줘"
답변:
{{
  "tasks": [
    {{
      "intent": "employee_list",
      "target_fields": ["employee"],
      "employee_name": null,
      "employee_id": null,
      "department": "인사부",
      "team": null,
      "position": null,
      "filters": [
        {{
          "field": "department",
          "op": "eq",
          "value": "인사부"
        }}
      ],
      "is_self": false
    }}
  ]
}}

[예시 9]
질문: "영업 관련 직원 찾아줘"
답변:
{{
  "tasks": [
    {{
      "intent": "employee_list",
      "target_fields": ["employee"],
      "employee_name": null,
      "employee_id": null,
      "department": "영업부",
      "team": null,
      "position": null,
      "filters": [
        {{
          "field": "department",
          "op": "eq",
          "value": "영업부"
        }}
      ],
      "is_self": false
    }}
  ]
}}

사용자 질문:
{question}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 700,
                },
            },
            timeout=180,
        )

        response.raise_for_status()

        raw_text = response.json().get("response", "").strip()
        json_text = clean_json_text(raw_text)

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            print("[ERROR] question analyzer JSON decode failed")
            print("[DEBUG] raw_text:", raw_text)
            return {"tasks": [DEFAULT_TASK.copy()]}

    except requests.exceptions.Timeout:
        print("[ERROR] question analyzer timeout")
        return {"tasks": [DEFAULT_TASK.copy()]}

    except requests.exceptions.RequestException as e:
        print("[ERROR] question analyzer request failed:", e)
        return {"tasks": [DEFAULT_TASK.copy()]}


def normalize_tasks(analysis: dict, question: str = "") -> list[dict]:
    """
    LLM 분석 결과를 코드에서 안전하게 보정한다.

    중요:
    - 권한 판단은 여기서 하지 않는다.
    - 필드명과 filters 형식만 안전하게 정리한다.
    """

    tasks = analysis.get("tasks", [])

    if not isinstance(tasks, list) or not tasks:
        tasks = []

    found_department = find_known_value(question, DEPARTMENTS)
    found_team = find_known_value(question, TEAMS)
    found_position = find_known_value(question, POSITIONS)
    found_org_alias = find_org_alias(question)

    if found_org_alias and not found_department and not found_team:
        alias_field, alias_value = found_org_alias

        if alias_field == "department":
            found_department = alias_value

        if alias_field == "team":
            found_team = alias_value

    compact_question = compact_text(question)

    list_keywords = [
        "종류",
        "목록",
        "리스트",
        "전체부서",
        "전체팀",
        "전체직급",
        "전체직책",
        "어떤부서",
        "어떤팀",
        "부서뭐",
        "팀뭐",
        "직급뭐",
        "직책뭐",
    ]

    requested_category_list = any(
        keyword in compact_question
        for keyword in list_keywords
    )

    normalized_tasks = []

    for task in tasks:
        if not isinstance(task, dict):
            continue

        intent = task.get("intent", "unknown")

        if intent not in ALLOWED_INTENTS:
            intent = "unknown"

        target_fields = task.get("target_fields", ["unknown"])

        if not isinstance(target_fields, list):
            target_fields = ["unknown"]

        safe_fields = [
            field
            for field in target_fields
            if field in ALLOWED_FIELDS
        ]

        if not safe_fields:
            safe_fields = ["unknown"]

        safe_fields = normalize_target_fields_by_question(
            target_fields=safe_fields,
            question=question,
        )

        raw_department = task.get("department")
        raw_team = task.get("team")
        raw_position = task.get("position")

        department = raw_department if raw_department in DEPARTMENTS else found_department
        team = raw_team if raw_team in TEAMS else found_team
        position = raw_position if raw_position in POSITIONS else found_position

        filters = normalize_filters(task.get("filters", []))
        filters = normalize_semantic_filters(filters, question)

        if department:
            filters = add_filter_if_missing(
                filters=filters,
                field="department",
                op="eq",
                value=department,
            )

        if team:
            filters = add_filter_if_missing(
                filters=filters,
                field="team",
                op="eq",
                value=team,
            )

        if position and not task.get("employee_name"):
            filters = add_filter_if_missing(
                filters=filters,
                field="position",
                op="eq",
                value=position,
            )

        # "인사부 알려줘"처럼 실제 부서명이 있는데 목록 질문이 아니면
        # category_list가 아니라 employee_list로 보정한다.
        if (
            intent == "category_list"
            and not requested_category_list
            and (department or team or position)
        ):
            intent = "employee_list"
            safe_fields = ["employee"]

        # filters가 있는데 unknown이면 조건 검색으로 보정한다.
        if filters and intent == "unknown":
            intent = "condition_search"

        # filters가 있는데 target_fields를 못 잡았으면 직원 목록 조회로 본다.
        if filters and safe_fields == ["unknown"]:
            safe_fields = ["employee"]

        employee_name = task.get("employee_name")

        if is_org_alias_text(employee_name):
            employee_name = None

        normalized_tasks.append(
            {
                "intent": intent,
                "target_fields": safe_fields,
                "employee_name": employee_name,
                "employee_id": task.get("employee_id"),
                "department": department,
                "team": team,
                "position": position,
                "filters": filters,
                "is_self": bool(task.get("is_self", False)),
            }
        )

    if not normalized_tasks:
        normalized_tasks.append(DEFAULT_TASK.copy())

    return normalized_tasks
