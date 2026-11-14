# Looseness 오류 3개 및 Normal 날짜 대조 후속 검증

## 실행 상태

사용자 보고: Looseness 907개 중 904개 성공/3개 오류, 9/12는 44/451 이상, 9/16은 453/453 이상, 기존 36개 일치, conclusions_ready=false다. 이 값들은 배경 정보이며 새 결과로 하드코딩하지 않는다. 실제 WSL CSV/JSON/MAT가 Codex에 없어 경로·원인·Normal 614개 결과는 이 환경에서 확인하지 못했다.

알려준 `/home/kungh/v2-error-analysis/normal_sample_36_run/`는 Normal 표본 캐시 후보이며 전체 614개 결과와 구분한다. Looseness 전체 결과 폴더는 실행 시 정확히 지정한다.

## WSL 실행

```bash
cd "/mnt/c/Users/kungh/OneDrive/바탕 화면/isu_chemical"
git fetch origin
git switch --track origin/feat/looseness-normal-followup-main
# 로컬 브랜치가 있으면 git switch feat/looseness-normal-followup-main
read -r -p 'Looseness 전체 분석 결과 폴더(analysis_report.json/file_results.csv): ' loose_results
followup_output="$HOME/v2-error-analysis/normal-followup-$(date -u +%Y%m%dT%H%M%S%NZ)"
"$HOME/capstone-web-venv/bin/python" -B scripts/followup_looseness_normal.py \
  --looseness-output "$loose_results" \
  --data-root "$PWD/data" \
  --bundles "$PWD/outputs/inference_bundles/date_prelim_20261106" \
  --cache-root "$HOME/v2-error-analysis/normal_sample_36_run" \
  --data-kind actual --output "$followup_output"
```

이미 완료된 전체 Normal 폴더가 다른 위치라면 --cache-root를 추가한다. Looseness 결과의 상위 폴더 아래에 있는 기존 file_results.csv/integration_results.csv/normal_results.csv도 자동 검색한다. 기존 출력은 읽기만 하며 새 폴더에 쓴다. 입력 루트가 없으면 실데이터 0개로 보고하지 않고 중단한다.

## 재사용 정책

- MAT 경로/날짜/RPM/Normal1·Normal2와 SHA256을 확인한다.
- AE 모델 해시·저장 threshold 및 bundle_id 일치가 필요하다.
- bundle_id가 없는 구형 integration CSV는 해당 integration_summary의 manifest·정규화·AE 파일 SHA256으로 보완한다. 모델 가중치 해시 하나만으로 통계를 검증했다고 보지 않는다.
- 검증된 캐시가 있으면 AE를 호출하지 않는다. 특징 코드/NumPy/SciPy 근거가 있으면 9개 특징도 재사용하며, 없으면 특징만 다시 계산한다.
- 기존 캐시가 있지만 근거가 부족하면 기본적으로 AE를 자동 재실행하지 않고 오류/이유를 남긴다. 검토 후 의도적으로 새 분석이 필요할 때만 `--allow-new-analysis-for-rejected-cache`를 지정한다. 현재 요청의 기본 명령에는 이 옵션이 없다.
- 검증된 캐시끼리 점수가 충돌하면 자동 재실행으로 숨기지 않고 보류한다.
- 기록이 없는 Normal만 기존 SavedModelInference 및 동일 9개 특징 경로로 계산한다. AE 재학습·threshold 재계산·RF 예측은 없다.
- 새 결과도 캐시가 되므로 같은 상위 폴더의 새 출력에서 재실행하면 처리 성공한 Normal을 재사용한다. 오류/미완료 파일은 재검사 대상이다.

## Looseness 오류 확인

looseness_errors.csv는 기존 실패 파일 경로·상태·예외를 그대로 보존한다. 해당 파일의 현재 SHA256, 기존 SHA 일치 여부, MAT 로더·9개 특징·AE 단계를 별도로 검사한다. 기존 성공 904개는 AE를 재실행하지 않는다. 기존 오류 파일만 재점검하며 원본 904/3 집계와 원래 conclusions_ready를 자동 덮어쓰지 않는다. 과거 오류가 현재 재현되지 않아도 원인을 임의 확정하지 않는다.

## 출력

- looseness_errors.csv: 실제 기존 오류 경로/예외와 현재 재현 검사
- normal_results.csv: Normal별 SHA·AE 결과·특징·재사용/신규·오류
- normal_by_date_rpm.csv: 날짜×RPM별 성공 n/AE 이상 수/관측 이상 비율
- group_summary.csv: Normal1/2 및 Looseness 강도별 분포와 오류 수
- date_rpm_comparison.csv: Normal/Looseness 날짜×RPM 점수·RMS 중앙값과 이상 비율
- date_shift_by_condition.csv: 같은 RPM/원래 조건의 날짜별 절대값·차이·안전 비율
- normal_control_comparison.csv: Normal1/2 대조 날짜 차이의 차이(인과효과 아님)
- cache_rejections.csv: 재사용하지 못한 이유
- followup_summary.json: source 상태·캐시/실제 AE 호출 수·근거 해시·처리 수·모델 일치·한계
- FOLLOWUP_REPORT.md: 한국어 검증 보고서
- PNG: 날짜/RPM/조건별 AE·RMS 등 분포 (기존 plotting 재사용)

Normal 614는 사용자 기대 수다. 실제로 다시 세며 다르면 경고 필드에 기록한다. `all_requested_normal_validated`는 실제 614개·전부 유효·두 날짜×세 RPM 모두 표본 존재·모델/원본 일치·탐색 제외 없음일 때만 true다. 아니면 종료 코드 1이며 완료된 관측 결과는 남는다. `comparison_valid`는 모델/파일/처리의 기술적 일치이며 독립 성능 검증이 아니다. 오류는 정상/이상 분모에서 제외하고 미처리 수를 확인한다.

파일별 CSV를 저장하며 중단된 결과도 다음 새 실행의 캐시 후보가 된다. 중단 시 가능한 부분 보고서를 남긴다. 이전 출력 폴더를 덮어쓰지 않고 새 폴더를 사용한다. 여러 프로세스가 같은 출력에 동시에 쓰지 않는다.

## 해석

1. 기존 Looseness의 날짜 차이가 기록상 유지되는지 확인하고 3개 오류를 성공 사례로 분류하지 않는다.
2. Normal에도 같은 날짜 변화가 관측되는지 Normal1/2·RPM별로 비교한다. Normal 혼합 비율이나 표본 수 차이를 고려한다.
3. Looseness AE 정상은 폴더 레이블 기준 누락 후보다. Normal의 날짜 분포 변화와 별개로 기술한다.
4. 정상군도 날짜 차이가 있다고 센서/부하/날짜 학습을 확정하지 않는다. 차이가 작다고 결함 특이성을 입증한 것도 아니다.
5. Normal의 Train/Validation/threshold 및 같은 Run 중복 여부가 미확인이다. 관측 비율을 독립 Test FPR/정확도/일반화 성능으로 표현하지 않는다.
6. 모델·threshold·원본 MAT·운영 DB·Streamlit을 변경하지 않는다.

## 검증

합성 테스트에서 검증된 캐시의 AE 호출 0, 미검증 캐시 자동 반복 금지, 캐시 충돌 보류, 정규화 포함 아티팩트 근거, 새 결과 재사용, 기존 오류 기록 보존 및 RF/DB 미호출을 검사한다. 실제 사용자 오류 3개의 원인과 전체 Normal 결과는 WSL 실행 후 확인해야 한다.

테스트: main 기반 신규 6개와 기존 94개를 함께 실행하여 총 100개 합성/회귀 테스트가 통과했다. 최종 해시/보고서 변경 후 신규 6개를 재실행했다. 실제 사용자 결과 CSV와 Normal MAT는 Codex에 없어 실제 오류 3개 경로/원인, 실제 재사용 개수 및 날짜별 비율을 확인하지 못했다.
