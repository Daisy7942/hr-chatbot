# ══════════════════════════════════════════════════════════════════════════════
# 데이터 파이프라인 진입점
# ══════════════════════════════════════════════════════════════════════════════
# 이 파일은 '호출 흐름'과 '최상단 예외 처리'만 담당한다.
#   함수 정의(def)        -> pipeline_modules/functions.py
#   단계별 처리 코드        -> pipeline_modules/preprocess, convert, indexing
#   설정·상수 / 예외        -> pipeline_modules/config, errors
#
# 단계 사이의 데이터는 파일이 아니라 메모리(변수)로 직접 넘긴다.
#
# 최상단 예외 처리 원칙:
#   - PipelineStop  : 정상 종료 안내 (예: nori 플러그인 설치 후 재시작 안내).
#   - PipelineError : 의도된 중단 사유 (필수 자원 없음, 인덱스 생성 실패 등). 종료 코드 1.
#   - 그 외 Exception : 단계 안에서 격리 못한 예상 못한 예외.
#                       전체 트레이스를 콘솔에 찍고 격리된 예외 1건으로 만들어 error.log 에 기록한 뒤
#                       종료 코드 1로 깔끔히 종료한다. (서버 프로세스가 그대로 죽는 걸 방지)
import traceback
from datetime import datetime

from pipeline_modules.errors import PipelineError, PipelineStop
from pipeline_modules.functions import (
    connect_opensearch, load_embedding_model,
    ensure_user_dictionary, ensure_nori_plugin, write_error_log,
)
from pipeline_modules.preprocess import run_preprocessing
from pipeline_modules.convert import run_jsonl_conversion
from pipeline_modules.indexing import create_indices, run_indexing


# 모듈 전역 플래그: main() 안에서 write_error_log 가 한 번이라도 정상 호출되어
# 풍부한 error.log 가 디스크에 기록된 적이 있는지 추적한다.
# 최상단 except 가 main() 종료 후 예외에서도 호출될 때, 이미 기록된 로그를
# 빈 리스트로 덮어쓰는 사고를 막기 위한 가드다.
_error_log_written = False


def main():
    # 실행 순서:
    #   1) OpenSearch 연결 확인
    #   2) 사용자 사전 복사 -> nori 플러그인 확인(없으면 설치 후 재시작 안내하며 정상 종료)
    #   3) 임베딩 모델 로딩 -> 인덱스 생성
    #   4) 1단계(전처리) -> 2단계(변환) -> 3단계(인덱싱)
    #   5) 에러 로그 저장
    global _error_log_written

    client = connect_opensearch()

    ensure_user_dictionary()
    ensure_nori_plugin(client)

    model = load_embedding_model()
    create_indices(client)

    # 각 단계는 자체 try/except 로 한 행/한 직원/한 인덱스 단위 실패를 격리하고,
    # 격리된 예외 목록(uncaught_exceptions)을 추가로 반환한다.
    dfs_clean, preprocessing_errors, source_filenames, preprocess_exceptions = (
        run_preprocessing()
    )
    records_by_source, convert_exceptions = run_jsonl_conversion(dfs_clean, source_filenames)
    indexing_errors, chunking_errors, indexing_exceptions = run_indexing(
        records_by_source, model, client
    )

    # 단계별 격리 예외를 하나로 모아 error.log 에 남긴다.
    all_uncaught_exceptions = (
        preprocess_exceptions + convert_exceptions + indexing_exceptions
    )

    write_error_log(
        preprocessing_errors,
        chunking_errors,
        indexing_errors,
        all_uncaught_exceptions,
    )
    # 정상 기록 완료 표시. 이후 다른 라인에서 예외가 나도 최상단 except 가
    # 이 파일을 빈 내용으로 덮어쓰지 않도록 한다.
    _error_log_written = True

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'\n========== 전체 완료 ({now}) ==========')


if __name__ == '__main__':
    try:
        main()
    except PipelineStop as stop:
        # 정상 종료(예: nori 설치 후 재시작 안내). 종료 코드 0.
        print(f'\n{stop}')
    except PipelineError as error:
        # 의도된 중단(필수 자원 없음 등). 종료 코드 1.
        print(f'\n파이프라인 중단:\n{error}')
        raise SystemExit(1)
    except Exception as error:
        # 위에서 못 잡힌 예상 못한 예외.
        # 단계 안에서 격리되지 않은 상위 호출 단계(연결, 사전 복사, 모델 로딩 등) 또는
        # 단계별 try/except 망을 빠져나온 케이스를 마지막으로 받는다.
        # 트레이스를 콘솔에 그대로 찍어 디버깅에 도움을 주고,
        # error.log 에도 단일 항목으로 기록한 뒤 종료 코드 1로 종료한다.
        print('\n파이프라인 예상 못한 예외 발생:')
        traceback.print_exc()

        # ── error.log 기록 ──────────────────────────────────────────────────
        # write_error_log 는 파일을 'w' (truncate) 모드로 연다.
        # main() 이 이미 정상 기록을 마친 뒤 발생한 예외(예: 마지막 print 의 인코딩 오류)에서
        # 빈 리스트로 다시 호출하면 풍부한 단계별 에러 이력이 통째로 사라진다.
        # 그래서 _error_log_written 플래그로 가드한다.
        #   - 아직 기록 전 (=대부분의 케이스)  → 단일 격리 항목으로 새로 기록
        #   - 이미 기록 완료                  → 기존 파일 보존, 콘솔 안내만 한다
        if _error_log_written:
            print('이미 기록된 error.log 를 보존하기 위해 재기록은 건너뜁니다.')
        else:
            try:
                write_error_log(
                    preprocessing_errors=[],
                    chunking_errors=[],
                    indexing_errors=[],
                    uncaught_exceptions=[{
                        '단계': '최상단(main)',
                        '대상': '-',
                        '오류': str(error),
                        '상세': traceback.format_exc(),
                    }],
                )
            except Exception as log_error:
                # 로그 기록조차 실패하면 더 할 수 있는 일이 없으므로 콘솔 안내만 한다.
                print(f'에러 로그 기록도 실패: {log_error}')

        raise SystemExit(1)
