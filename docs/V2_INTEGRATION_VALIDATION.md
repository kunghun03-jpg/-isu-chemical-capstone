# 중간발표 V2 실데이터 통합 검증

이 도구는 웹의 `dashboard_v2.analyze` → `candidate_analysis.analyze_with_engine`를 그대로 호출한다. 새 모델·새 특징·새 정책을 만들지 않는다. 파일명 대신 중립 표시명 `uploaded.mat`을 추론 API에 전달한다. CSV의 condition/date/정답은 감사 메타데이터이며 모델 입력이 아니다. 특징 입력은 CH3의 기존 9개 수치만 사용한다.

## WSL 실행

기존 웹 서버를 종료하고 Git 저장소에서 아래를 실행한다. 미커밋 작업은 보존한다. Python은 기존 웹 전용 환경을 그대로 사용한다. 새 RF를 학습하거나 모델을 최신 폴더로 임의 변경하지 않는다.

```bash
git fetch origin
git switch --track origin/feat/v2-integration-validation
# 이미 로컬 브랜치가 있으면 git switch feat/v2-integration-validation
export ISU_CHEMICAL_V2_MODEL=autoencoder
export ISU_CHEMICAL_CANDIDATE_LABELS=0
# 기존 61개 평가에 썼던 실제 RF 번들 절대경로를 기존 설정에 유지한다.
: "${ISU_CHEMICAL_CANDIDATE_BUNDLE:?기존 RF 번들 경로를 설정하세요}"
validation_output="$HOME/v2-integration/$(date -u +%Y%m%dT%H%M%S%NZ)"
"$HOME/capstone-web-venv/bin/python" -B scripts/validate_v2_integration.py \
  --test-predictions "$ISU_CHEMICAL_CANDIDATE_BUNDLE/test_predictions.csv" \
  --rf-bundle "$ISU_CHEMICAL_CANDIDATE_BUNDLE" \
  --ae-bundles "${ISU_CHEMICAL_V2_BUNDLE_ROOT:-$PWD/outputs/inference_bundles/date_prelim_20261106}" \
  --data-root "$PWD" --data-kind actual --output "$validation_output"
```

`--data-root`은 CSV 상대 경로를 해석할 기준이다. `data/...`이면 저장소 루트를 지정한다. 절대경로는 그대로 사용한다. 자동 추측·검색·경로 치환을 하지 않으므로 경로 오류는 명시적으로 보고된다. CSV는 path/rpm/raw_prediction 필수 열이 필요하며 중복 파일/RPM이나 잘못된 클래스는 실행 전 거부한다.

CSV에 다음 파일이 있으면 우선 처리한다. 목록에 없으면 summary의 `priority_file_included=false`를 보고한다. 파일을 임의로 추가하거나 Unbalance 예측을 하드코딩하지 않는다.

```text
data/20260919/20260919_1800RPM/Unbalance_-1/Vib/20260919_132506_Vib.mat
```

이 파일에 사용자가 알려 준 오프라인 결과는 Unbalance지만, 실제 비교 기준은 입력 CSV의 기록이다. AE 정상으로 RF가 생략되면 불일치가 아니다. 임의로 RF를 강제 실행하지 않는다.

## 출력과 해석

출력은 저장소 밖의 **새 폴더**만 허용하며 기존 결과를 덮어쓰지 않는다.

- `integration_results.csv`: 파일/RPM/조건/날짜, 입력 해시, AE 점수·임계값·판정, AE 버전, RF 실행 상태·원시 클래스·버전·해시, 최종 후보·보류·사유, 특징값/순서/설정, DB 검사 및 오류
- `offline_web_comparison.csv`: 기존 CSV 원시 예측과 웹 경로 결과 비교
- `integration_summary.json`: RPM별 비교, 조건별 파일 수, AE/후보/보류 요약, 입력·아티팩트 해시, 원본 변경 감지, 한계
- `PRESENTATION_CHECKLIST.md`: 실행 결과와 함께 사용하는 발표 점검표
- `case_XXXX/integration.sqlite3`, `case_XXXX/ae_only.sqlite3`: 이 실행의 새 검증 DB. 운영 DB가 아니다.

비교 분류:

- `match`: AE 이상이며 실제 RF 원시 예측 일치
- `mismatch`: AE 이상이며 실제 RF 원시 예측 불일치
- `skipped_ae_normal`: AE 정상으로 RF 생략, 불일치 분모에서 제외
- `rf_unavailable`: AE는 완료됐지만 모델/특징 오류 등으로 RF 결과 없음
- `analysis_error`: MAT 읽기/AE 분석 오류

`rf_executed=true`는 예측이 반환됐다는 뜻이다. 호출 실패 여부는 `rf_execution_status=failed_or_unavailable`와 실제 reasons에서 확인한다. AE 성공 뒤 RF 오류가 발생하면 AE 결과와 보류 상태도 저장 검사를 수행한다.

종료 코드 0은 비교/DB 검사의 기술적 통과이며 진단 정확도나 모든 조건 커버리지를 뜻하지 않는다. mismatch/analysis_error/rf_unavailable/DB 실패/입력 아티팩트 변경은 종료 코드 1이다. 정상 생략만 있어도 비교는 기술적으로 통과할 수 있으므로 실제 RF 비교 건수를 확인한다. condition 열이 없으면 조건 커버리지를 확인할 수 없다. Normal1/2, IR/OR/Roller, 두 Unbalance 및 Looseness가 누락됐다면 그 조건은 검증되지 않았다.

RF 불일치 시 원래 평가에서 사용한 모델 버전·해시, RPM, 특징 순서/스키마/계산 설정/측정 계약, 저장된 9개 특징을 원본 EDA CSV와 비교한다. 클래스만 있는 test_predictions.csv로는 원래 모델·특징의 동일성을 증명할 수 없다. 이 도구는 원인 조사 자료를 남기며 근거 없이 원인을 확정하지 않는다. Test로 정책을 조정하지 않는다.

## 저장·이력 안전성

운영 DB는 열지 않는다. 새 파일별 DB에 저장 → 같은 결과 중복 저장 → 재조회 → 새 Python 프로세스에서 재조회까지 비교한다. AE-only 결과도 별도 새 DB에 저장해 후보 없는 기록의 호환을 확인한다. 기존 운영 DB의 내용을 검사하거나 지우는 기능은 없다. 모델 경로는 신뢰하는 본인 번들만 사용한다(joblib 로드).

실제 웹에서 저장을 시연하기 전에는 서버 종료 후 SQLite backup으로 운영 DB를 외부 백업한다. DB 경로는 실제 설정값을 사용한다.

```bash
: "${ISU_CHEMICAL_HISTORY_DB:?실제 운영 DB 경로를 확인하고 설정하세요}"
"$HOME/capstone-web-venv/bin/python" -B - <<'PY'
import os,sqlite3
from pathlib import Path
from datetime import datetime,timezone
source=Path(os.environ['ISU_CHEMICAL_HISTORY_DB']).expanduser().resolve()
if not source.is_file(): raise SystemExit('DB가 존재하지 않습니다. 경로 확인 필요')
out=Path.home()/'v2-db-backups'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out.mkdir(parents=True)
with sqlite3.connect(source.as_uri()+'?mode=ro',uri=True) as src, sqlite3.connect(out/'history.sqlite3') as dst:
    src.backup(dst)
print(out/'history.sqlite3')
PY
"$HOME/capstone-web-venv/bin/python" -B -m streamlit run streamlit_app_v2.py --server.port 8501
```

## 발표 전 수동 확인

1. Chrome 번역 아이콘에서 원문 표시 및 '이 사이트 번역 안 함'을 선택한다. 확장 프로그램의 자동 번역도 끈다. 잘못 바뀐 한국어를 앱 원문으로 오인하지 않는다.
2. 실제 MAT 업로드 및 RPM 선택 후 분석한다. 분석 중 spinner가 끝날 때까지 버튼을 반복 조작하지 않는다. 이 CLI는 브라우저 중복 클릭 동작을 검증하지 않는다.
3. 파일·RPM 변경 후 AE/RF/FFT가 초기화되는지 확인한다.
4. AE 정상은 RF 생략, AE 이상은 원시 예측과 최종 보류를 분리한다. 최종 후보를 원시 예측으로 대체하지 않는다.
5. 실제 Normal1/2, IR/OR/Roller, Unbalance -1/-2 및 Looseness, 다른 날짜·RPM을 확인한다.
6. 저장 후 웹 서버를 재시작하고 이력에서 점수·원시 예측·후보·사유를 확인한다. CLI의 프로세스 재조회 검증은 실제 브라우저 재시작 검증과 구분한다.
7. 모델 누락/손상과 오류는 실제 메시지를 확인한다. 원본 모델을 손상시키지 말고 복사본만 이용한다.

## Codex 검증 범위

실제 사용자 MAT·학습 번들은 접근할 수 없으므로 합성 테스트만 수행한다. 기존 합성 RF 아티팩트의 실제 추론과 웹 경로 비교, 모의 오류, AE 정상 생략, 보류, 새 프로세스 DB 재조회 테스트를 구분한다. RF.fit을 금지하며 재학습하지 않는다. 사용자 Test 61개 성능을 이번 실행 결과로 재사용하지 않는다. 이 작업은 기존 V2 디자인/모델/DB 스키마를 변경하지 않는다.

이번 신규 테스트 9개와 기존 회귀 38개를 검증한다. 별도 CLI 합성 입력 1개로 요청한 네 산출물 생성과 재시작 DB 검사도 확인한다.
