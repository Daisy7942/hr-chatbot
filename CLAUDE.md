# 인사 데이터 기반 RAG 챗봇 프로젝트

## 코딩 규칙

- 모든 코드는 기초적인 Python 문법으로 작성한다.
- 타입 힌트 사용 금지.
- 고급 문법(list comprehension 중첩, dict.fromkeys 등 비직관적 패턴) 사용 금지.
- 반복문, 조건문, 기본 자료구조(list, dict)를 활용한 단순한 코드로 작성한다.

## 프로젝트 개요

- **프로젝트명**: 인사 데이터 기반 RAG 챗봇 (두리안정보기술)
- **구성**: FastAPI + OpenSearch + Ollama (gemma3:4b)
- **임베딩 모델**: paraphrase-multilingual-MiniLM-L12-v2 (384차원)
- **데이터**: 더미 인사 데이터 3개 CSV, 각 2,000건 (총 6,000 레코드)

---

## 데이터 파일 구성

3개 파일은 `사원번호`를 공통 키(PK/FK)로 연결됩니다.

| 파일명 | 컬럼 수 | 주요 내용 |
|---|---|---|
| 기본인사정보.csv | 30개 | 이름, 부서, 직급, 연락처 등 인적사항 |
| 역량성과.csv | 13개 | 성과점수, 인사고과, 자격증, 포상/징계 |
| 급여정보.csv | 7개 | 연봉, 잔업시간, 미사용휴가, 계좌 및 보험 |

---

## OpenSearch 인덱스 구조 (데이터구조정의서 v1.4)

### 인덱스 목록

| 인덱스명 | security_level | 포함 필드 |
|---|---|---|
| hr_basic_1 | 1 | 이름, 성별, 나이, 입사일, 부서, 직급, 이메일, 부서레벨, 직급레벨 등 일반 인적사항 |
| hr_basic_2 | 2 | 생년월일, 병역, 학력, 출신대학, 학점, 전화번호, 이전직장 정보 |
| hr_basic_3 | 3 | 주민등록번호, 주소, 퇴직구분, 퇴직일자 |
| hr_performance_2 | 2 | 성과점수, 인사고과(2020~2024), 자격증, TOEIC, 포상이력 |
| hr_performance_3 | 3 | 징계이력, 징계사유, 자격증수당여부 |
| hr_salary_2 | 2 | 잔업시간, 미사용휴가일수 |
| hr_salary_3 | 3 | 연봉, 급여은행, 계좌번호, 4대보험가입여부 |

### 인덱스별 필드 매핑

```
기본인사정보.csv → hr_basic_1, hr_basic_2, hr_basic_3
역량성과.csv     → hr_performance_2, hr_performance_3
급여정보.csv     → hr_salary_2, hr_salary_3
```

### 공통 메타데이터 필드 (전체 인덱스 공통)

| 필드명 | 타입 | 설명 |
|---|---|---|
| _id | OpenSearch metadata | 문서 고유 ID (예: BAS_00001) |
| employee_id | keyword | 사원번호 — 본인 여부 확인에 사용 |
| department | keyword | 부서명 — 표시용 |
| position | keyword | 직급명 — 표시용 |
| embedding_text | text | Hybrid Search용 자연어 텍스트 (Nori 형태소 분석기) |
| embedding_vector | knn_vector | 384차원 임베딩 벡터 |

- `security_level`은 JSONL 문서에 포함되지 않으며 인덱스 자체 설정으로 관리
- `department`, `position`은 표시용이므로 keyword 타입 (Nori 분석 미적용)

### _id 유형코드

| 인덱스 | 유형코드 | 예시 |
|---|---|---|
| hr_basic_* | BAS | BAS_00001 |
| hr_performance_* | PERF | PERF_00001 |
| hr_salary_* | SAL | SAL_00001 |

---

## 권한 구조

### permission_level 산출
```
permission_level = MAX(부서레벨, 직급레벨)
```

| 구분 | 값 |
|---|---|
| 부서레벨 | 일반부서=1, 인사부=3 |
| 직급레벨 | 사원·대리·과장=1, 차장·부장=2, 이사·사장=3 |

### 접근 조건

| 구분 | 접근 가능 인덱스 |
|---|---|
| 본인 데이터 (전 레벨) | 전체 인덱스 7개 |
| Level 1 (사원/대리/과장) | hr_basic_1 |
| Level 2 (차장/부장) | hr_basic_1/2, hr_performance_2, hr_salary_2 |
| Level 3 (이사/사장, 인사부) | 전체 인덱스 7개 |

---

## API 구조

### Request (POST /predict)
```json
{ "employee_id": "EMP0001", "question": "내 기본정보 알려줘", "model_type": "ollama_gemma3_4b" }
```

### Response
```json
{
  "success": true,
  "answer": "홍길동님은 인사부 소속 대리입니다.",
  "permission": { "allowed": true, "permission_level": 1 },
  "sources": [{ "index": "hr_basic_1", "_id": "BAS_00001" }]
}
```

---

## 폴더 구조

```
dma/
├── 데이터 전처리 정제/
│   ├── dataset/            # 원본 더미 CSV (기본인사정보, 역량성과, 급여정보)
│   ├── output/             # 정제된 CSV 출력 (gitignore)
│   ├── preprocessing.ipynb
│   ├── requirements.txt
│   └── .env
├── JSONL 변환/
│   ├── output/             # JSONL 출력 (gitignore)
│   ├── jsonl_conversion.ipynb
│   ├── requirements.txt
│   └── .env
└── CLAUDE.md
```
