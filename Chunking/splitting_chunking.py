import json
from pathlib import Path
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

print('라이브러리 로딩 완료!')

BASE_DIR = Path(__file__).resolve().parent

load_dotenv()

INPUT_DIR  = Path(os.getenv('INPUT_DIR',  str(BASE_DIR.parent / 'JSONL' / 'output')))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', str(BASE_DIR / 'output')))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 임베딩 모델 (embedding_indexer 와 동일)
EMBEDDING_MODEL = os.getenv('EMBED_MODEL_NAME', 'paraphrase-multilingual-MiniLM-L12-v2')

# 청크당 최대 토큰 수 (.env에서 관리)
MAX_TOKENS = int(os.environ['MAX_TOKENS'])

print(f'입력 디렉토리: {INPUT_DIR}')
print(f'출력 디렉토리: {OUTPUT_DIR}')
print(f'최대 토큰 수: {MAX_TOKENS}')

# ── 토크나이저 로딩 ────────────────────────────────────────────────────────────

print(f'\n임베딩 모델 로딩: {EMBEDDING_MODEL}')
try:
    model = SentenceTransformer(EMBEDDING_MODEL)
    tokenizer = model.tokenizer
except Exception as e:
    print(f'임베딩 모델 로딩 실패: {e}')
    raise SystemExit(1)


def count_tokens(text):
    # 텍스트의 토큰 수 계산 (special token 포함)
    return len(tokenizer.encode(text))


# ── 1. 데이터 로딩 ─────────────────────────────────────────────────────────────

jsonl_files = sorted(
    f for f in INPUT_DIR.glob('*.jsonl')
    if f.name != 'changes_history.jsonl'
)

if not jsonl_files:
    print(f'JSONL 파일 없음: {INPUT_DIR}')
    print('JSONL 변환 스크립트를 먼저 실행해 주세요.')
    raise SystemExit(1)

record_sets = {}

for path in jsonl_files:
    records = []
    try:
        f_handle = open(path, 'r', encoding='utf-8')
    except Exception as e:
        print(f'JSONL 파일 열기 실패: {path.name} → {e}')
        raise SystemExit(1)
    with f_handle as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as e:
                print(f'JSONL 파싱 실패: {path.name} → {e}')
                raise SystemExit(1)
            records.append(record)

    record_sets[path.stem] = records
    print(f'  로딩: {path.name}  ({len(records):,}건)')

print(f'\n로딩 완료! 총 {len(record_sets)}개 파일')

# ── 2. 텍스트 정규화 ───────────────────────────────────────────────────────────

def normalize_field(line):
    # 필드 줄 단위로 공백 정규화 (줄바꿈 구조 유지)
    words = line.strip().split()
    return ' '.join(words)


normalize_count = 0

for file_name, records in record_sets.items():
    for rec in records:
        original = rec.get('embedding_text', '')
        fields = [normalize_field(f) for f in original.split('\n') if f.strip()]
        normalized = '\n'.join(fields)
        if original != normalized:
            normalize_count += 1
        rec['embedding_text'] = normalized

print(f'\n정규화 완료! 적용된 레코드 수: {normalize_count:,}건')

# ── 3. 동적 청킹 ──────────────────────────────────────────────────────────────

def chunk_by_tokens(embedding_text, max_tokens):
    # 필드를 하나씩 추가하면서 토큰 한계를 체크
    fields = [f for f in embedding_text.split('\n') if f.strip()]
    chunks = []
    current_fields = []

    for field in fields:
        # 현재 필드를 추가했을 때 토큰 수 계산
        candidate_text = '\n'.join(current_fields + [field])
        if count_tokens(candidate_text) > max_tokens and current_fields:
            # 한계 초과 → 이전 필드까지로 청크 마감하고 새 청크 시작
            chunks.append('\n'.join(current_fields))
            current_fields = [field]
        else:
            current_fields.append(field)

    # 마지막 청크 추가
    if current_fields:
        chunks.append('\n'.join(current_fields))

    return chunks


print('\n청크 분할 시작...\n')

chunked_sets = {}

empty_text_count = 0

for file_name, records in record_sets.items():
    chunks = []
    for rec in records:
        embedding_text = rec.get('embedding_text', '')
        # 빈 embedding_text 방어 로직
        if not embedding_text.strip():
            emp_id = rec.get('employee_id', '?')
            print(f'  경고: embedding_text 비어있음 → 사원 {emp_id} 스킵')
            empty_text_count += 1
            continue
        chunk_texts = chunk_by_tokens(embedding_text, MAX_TOKENS)
        for chunk_text in chunk_texts:
            chunk_record = {
                'employee_id':      rec.get('employee_id', ''),
                'employee_name':    rec.get('employee_name', ''),
                'department':       rec.get('department', ''),
                'department_level': rec.get('department_level', ''),
                'job_grade':        rec.get('job_grade', ''),
                'job_grade_level':  rec.get('job_grade_level', ''),
                'embedding_text':   chunk_text,
                'source':           rec.get('source', ''),
                'timestamp':        rec.get('timestamp', ''),
                'changed':          rec.get('changed', []),
            }
            chunks.append(chunk_record)

    chunked_sets[file_name] = chunks
    print(f'청크 분할 결과: [{file_name}]')
    print(f'  원본 레코드 수: {len(records):,}건')
    print(f'  생성된 청크 수: {len(chunks):,}건\n')

print(f'청킹 완료! (빈 embedding_text 스킵: {empty_text_count}건)')

# ── 4. 검증 ────────────────────────────────────────────────────────────────────

print('\n데이터 소실 여부 확인 중...')
print('-' * 50)

for file_name in record_sets:
    original_count = len(record_sets[file_name])
    chunk_count    = len(chunked_sets[file_name])
    status = '정상' if chunk_count >= original_count else '경고: 데이터 소실 발생'
    print(f'  {file_name}')
    print(f'  원본 {original_count:,}건 → 청크 {chunk_count:,}건  [{status}]\n')

print('-' * 50)

required_fields = ['employee_id', 'department', 'job_grade', 'embedding_text']

print('\n필수 필드 유지 확인 중...')
print('-' * 50)

for file_name, chunks in chunked_sets.items():
    missing_count = 0
    for chunk in chunks:
        for field in required_fields:
            if not chunk.get(field):
                missing_count += 1
                break
    status = '정상' if missing_count == 0 else f'누락 {missing_count}건'
    print(f'  {file_name}: [{status}]')

print('-' * 50)

print('\n토큰 한계 초과 여부 확인 중...')
print('-' * 50)

for file_name, chunks in chunked_sets.items():
    over_count = 0
    for chunk in chunks:
        if count_tokens(chunk['embedding_text']) > MAX_TOKENS:
            over_count += 1
    status = '정상' if over_count == 0 else f'초과 {over_count}건'
    print(f'  {file_name}: [{status}]')

print('-' * 50)

# ── 5. 결과 저장 ───────────────────────────────────────────────────────────────

print('\n결과 저장 중...\n')

for file_name, chunks in chunked_sets.items():
    out_path = OUTPUT_DIR / f'{file_name}.jsonl'

    with open(out_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    print(f'  저장: {out_path.name}  ({len(chunks):,}건)')

print('\n모든 파일 저장 완료!')
