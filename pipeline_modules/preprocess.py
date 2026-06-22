# ══════════════════════════════════════════════════════════════════════════════
# 1단계 처리: 원본 CSV 검증·교정 (functions.py의 run_validations 호출)
# ══════════════════════════════════════════════════════════════════════════════
# 검증에 쓰는 상태(_errors 등)는 functions.py 안에 캡슐화돼 있고,
# run_validations(df)가 그 df의 (에러목록, 제거할 행번호)를 반환한다.
#
# 예외 처리 원칙:
#   - 한 CSV 파일 로딩/검증 실패해도 다음 파일은 계속 처리한다.
#   - 실패 사유는 콘솔 print + uncaught_exceptions 리스트에 기록한다.
#   - 호출하는 pipeline.py 가 이 리스트를 최종 error.log 에 함께 남긴다.
import traceback
import pandas as pd
from pipeline_modules.functions import run_validations
from pipeline_modules.config import DATASET_DIR
from pipeline_modules.errors import PipelineError


def run_preprocessing():
    # 원본 CSV들을 읽어 컬럼별 검증·교정 후 정제된 DataFrame들을 메모리로 반환한다.
    print('\n========== 1단계: 전처리 ==========')
    print(f'입력 폴더: {DATASET_DIR}')

    csv_files = sorted(DATASET_DIR.glob('*.csv'))
    if not csv_files:
        # 입력 자체가 없으면 더 진행할 수 없으므로 파이프라인을 중단한다.
        raise PipelineError(f'CSV 파일 없음: {DATASET_DIR}')

    dfs = {}
    source_filenames = {}   # path.stem -> 원본 CSV 파일명 (source 필드에 사용)
    uncaught_exceptions = []  # 파일/검증 단계에서 격리된 예외 모음

    for path in csv_files:
        # CSV 한 개 로딩이 깨져도 다른 파일은 계속 진행한다.
        try:
            df = pd.read_csv(path, encoding='utf-8-sig', dtype=object)
        except Exception as error:
            print(f'  [경고] {path.name} 로딩 실패 → 이 파일은 건너뜀: {error}')
            uncaught_exceptions.append({
                '단계': '1단계 전처리(CSV 로딩)',
                '대상': path.name,
                '오류': str(error),
                '상세': traceback.format_exc(),
            })
            continue

        dfs[path.stem] = df
        source_filenames[path.stem] = path.name
        print(f'  로딩: {path.name}  ({len(df):,}행 / {len(df.columns)}열)')

    cleaned = {}
    all_errors = []

    for source_name, df in dfs.items():
        print(f'\n처리 중: {source_name}  ({len(df):,}행)')

        # 한 파일의 검증 함수가 예외를 던지면 그 파일은 건너뛰고 나머지는 계속한다.
        try:
            errors, drop_rows = run_validations(df)
        except Exception as error:
            print(f'  [경고] {source_name} 검증 실패 → 이 파일은 건너뜀: {error}')
            uncaught_exceptions.append({
                '단계': '1단계 전처리(검증)',
                '대상': source_name,
                '오류': str(error),
                '상세': traceback.format_exc(),
            })
            continue

        for err in errors:
            err['파일명'] = source_name
        all_errors.extend(errors)
        print(f'  에러: {len(errors):,}건')

        df_clean = df.drop(index=list(drop_rows)).reset_index(drop=True)
        cleaned[source_name] = df_clean
        print(f'  정제 결과: {len(df_clean):,}행 (제거 {len(drop_rows)}행)')

    print(f'\n전처리 에러 누적: {len(all_errors):,}건')
    if uncaught_exceptions:
        print(f'전처리 예외 격리: {len(uncaught_exceptions):,}건 (자세한 내용은 error.log)')

    return cleaned, all_errors, source_filenames, uncaught_exceptions
