import json

files = [
    ('기본인사정보_정제.jsonl', 'Chunking/output/기본인사정보_정제.jsonl'),
    ('역량성과_정제.jsonl', 'Chunking/output/역량성과_정제.jsonl'),
    ('급여정보_정제.jsonl', 'Chunking/output/급여정보_정제.jsonl'),
    ('통합인사정보_정제.jsonl', 'Chunking/output/통합인사정보_정제.jsonl'),
]

for name, path in files:
    with open(path, encoding='utf-8') as fp:
        lines = [json.loads(l) for l in fp if l.strip()]
    sources = set(r.get('source', '') for r in lines)
    emp_ids = set(r.get('employee_id', '') for r in lines)
    print(f'{name}: {len(lines)}청크, {len(emp_ids)}명, source={sources}')

    # embedding_text 샘플 확인
    sample = lines[0] if lines else None
    if sample:
        print(f'  embedding_text 샘플: {sample.get("embedding_text","")[:80]}')
        print(f'  changed 샘플: {sample.get("changed", [])}')
        print()
