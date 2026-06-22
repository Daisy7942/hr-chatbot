# Durian HR RAG Chatbot — 시스템 흐름도

---

## 1. 데이터 적재 파이프라인 (Batch)

> CSV 원본 파일을 OpenSearch에 임베딩 벡터와 함께 적재하는 배치 작업입니다.
> `python pipeline.py` 로 실행합니다.

```
[ pipeline.py 실행 ]
         │
         ▼
[ OpenSearch 연결 확인 ]
         │
         ▼
[ 사용자 사전 복사 ]
  user_dictionary.txt → OpenSearch config/
         │
         ▼
[ nori 플러그인 확인 ]
  미설치 시 자동 설치 후 재시작 안내
         │
         ▼
[ 임베딩 모델 로딩 ]
  paraphrase-multilingual-MiniLM-L12-v2 (384차원)
         │
         ▼
< 사전 파일(MD5) 변경됨? >
   변경됨 │   변경 없음
         ▼           ▼
[ 인덱스 전체   [ 없는 인덱스만
  삭제 후 재생성 ]   생성 확인 ]
         │           │
         └─────┬─────┘
               │
               ▼
[ ① CSV 읽기 ]
  기본인사정보 / 역량성과 / 급여정보
         │
         ▼
[ ② 결측값 처리 ]
  빈 값 → "미입력"
         │
         ▼
[ ③ 유효성 검사 · 오류 행 제거 ]
  나이·연봉·직급 등 범위·목록 체크
         │
         ▼
[ ④ 레코드 변환 ]
  CSV 행 → 직원별 dict
         │
         ▼
[ ⑤ 변경 감지 ]
  기존 OpenSearch 문서와 메타·텍스트 해시 비교
         │
         ▼
[ ⑥ 청킹 ]
  필드 추출 → "필드명: 값" 120토큰 단위로 분할
         │
         ▼
[ ⑦ 임베딩 ]
  청크 텍스트 → 384차원 벡터
         │
         ▼
[ ⑧ Bulk 적재 ]
  OpenSearch 인덱스에 텍스트 + 벡터 저장
               │
               ▼
         [ error.log 저장 ]
               │
               ▼
            [ 완료 ]
```

**인덱스 구성 (7개)**

| 인덱스 | 접근 등급 | 주요 필드 |
|---|---|---|
| hr_basic_1 | Level 1+ | 이름·부서·직급·입사일 등 |
| hr_basic_2 | Level 2+ | 학력·이전직장·연락처 등 |
| hr_basic_3 | Level 3+ | 주민번호·주소·퇴직정보 |
| hr_performance_2 | Level 2+ | 성과점수·인사고과·자격증 |
| hr_performance_3 | Level 3+ | 징계이력·자격증수당 |
| hr_salary_2 | Level 2+ | 잔업시간·미사용휴가 |
| hr_salary_3 | Level 3+ | 연봉·계좌번호·4대보험 |

> 권한 등급 계산: `max(부서레벨, 직급레벨)` → Level 1 / 2 / 3

---

## 2. 실시간 RAG 챗봇 (Runtime)

> 사용자 질문이 들어왔을 때 권한 검사 → 검색 → 답변 생성까지의 흐름입니다.

```
[ 브라우저: 사번 + 질문 입력 ]
         │  POST /api/rag-chat
         ▼
[ Express 프록시 :3000 ]
  /api → FastAPI :8000 중계
         │
         ▼
[ 입력값 검증 ]
  사번·질문 비어 있으면 HTTP 400
         │
         ▼
< 사번 변경됨? >
   달라짐 │   동일함
         ▼           │
[ 대화 기억 초기화 ]  │
         │           │
         └─────┬─────┘
               │
               ▼
[ 사번 조회 ]
  hr_basic_1 에서 검색
  없으면 HTTP 404
               │
               ▼
[ 권한 등급 확인 ]
  max(부서레벨, 직급레벨) → Level 1 / 2 / 3
               │
               ▼
[ 질문 보강 ]
  이전 대화 컨텍스트 기반으로 생략된 표현 복원
               │
               ▼
[ LLM 질문 분해 ]
  "홍길동 부서와 연봉" → Task1(부서) / Task2(연봉)
               │
               ▼  (Task별 순환)
┌─────────────────────────────────────┐
│                                     │
│  < 요청자 권한 ≥ Task 요구 등급? >  │
│   권한 있음 │   권한 없음           │
│             │         ▼             │
│             │  [ACCESS_DENIED]      │
│             │   "Level N 이상 필요" │
│             ▼                       │
│  [ OpenSearch 하이브리드 검색 ]     │
│    BM25 (nori 형태소 분석)          │
│    + KNN (384차원 벡터 유사도)      │
│    → RRF 점수 병합                  │
│             │                       │
│             ▼                       │
│  [ LLM 답변 생성 ]                  │
│    Ollama (gemma3:4b)               │
│    또는 OpenAI (gpt-4o-mini)        │
│             │                       │
└─────────────┴── 다음 Task 반복 ─────┘
               │ (모든 Task 완료)
               ▼
[ 종합 답변 조합 + 세션 업데이트 ]
               │
               ▼
[ JSON 응답 반환 → 브라우저 표시 ]
```

---

## 3. 컴포넌트 구성

```
  [ 브라우저 ]  index.html / Alpine.js
        │
        │  POST /api/rag-chat
        ▼
  [ 프론트 서버 :3000 ]  server.js / Express
        │
        │  POST /rag-chat
        ▼
  [ 백엔드 :8000 ]  FastAPI
    ├─ session_service        대화 기억 · 사번 관리
    ├─ task_processor_service 질문 분해 · 권한 필터
    ├─ hybrid_search_service  BM25 + KNN + RRF
    └─ llm_service            Ollama / OpenAI
        │
        │  검색 쿼리
        ▼
  [ OpenSearch :9200 ]
    인덱스 7개 (hr_basic / hr_performance / hr_salary)
    nori 형태소 분석 + 384차원 벡터 검색
        ▲
        │  Bulk 적재 (배치)
  [ 파이프라인 ]  pipeline.py
    CSV → 전처리 → 변환 → 청킹 → 임베딩 → 적재
```
