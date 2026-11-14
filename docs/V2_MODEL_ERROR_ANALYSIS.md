# V2 후속 1단계: Looseness 검증 및 AE/RF 오류 조사

기존 모델·정규화·임계값·보류 정책·웹 UI·운영 DB를 유지하는 원인 조사다. 개선 모델을 학습하거나 운영에 적용하지 않는다. 기준 통합 검증 커밋은 main PR #15에 병합되어 추가 미병합 의존성이 없다.

## 1. Looseness 파일 목록

저장소에서 기존 웹 환경을 사용한다. 원본 MAT는 수정하지 않는다. 탐색은 기존 EDA metadata 계약을 재사용한다: 정확한 날짜/RPM 폴더, Looseness0.05/Looseness0.1/LoosenessFull 및 Vib 폴더, 측정 파일 날짜가 일치해야 한다. 조건명은 검증용 정답 메타데이터이지 추론 입력이 아니다. 모호한 경로는 excluded_files.csv에 기록한다. 모든 유효 파일을 선택하며 날짜별/RPM별로 후속 분석한다. MAT 내용을 읽지 않는 목록 생성 단계다.

```bash
git fetch origin
git switch --track origin/feat/v2-condition-error-analysis
# 이미 브랜치가 있으면 git switch feat/v2-condition-error-analysis
stamp="$(date -u +%Y%m%dT%H%M%S%NZ)"
loose_cases="$HOME/v2-error-analysis/$stamp-cases"
loose_run="$HOME/v2-error-analysis/$stamp-integration"
error_report="$HOME/v2-error-analysis/$stamp-report"
"$HOME/capstone-web-venv/bin/python" -B scripts/discover_looseness_cases.py \
  --mat-root "$PWD/data" --output "$loose_cases"
```

다른 데이터 루트면 --mat-root를 실제 위치로 바꾼다. 여러 루트는 옵션을 반복한다. 심볼릭 링크는 실제 경로 기준 중복 제거한다. 발견 파일 0이면 추론을 진행하지 말고 제외 목록과 폴더 구조를 먼저 확인한다.

## 2. 실제 MAT 추론 (WSL에서만)

기존 RF/AE 번들 경로를 유지한다. raw_prediction 공란은 참조값 미확인이지 Normal이나 오분류가 아니다.

```bash
export ISU_CHEMICAL_V2_MODEL=autoencoder
export ISU_CHEMICAL_CANDIDATE_LABELS=0
: "${ISU_CHEMICAL_CANDIDATE_BUNDLE:?기존 RF 번들 경로를 설정하세요}"
"$HOME/capstone-web-venv/bin/python" -B scripts/validate_v2_integration.py \
  --additional-cases "$loose_cases/looseness_cases.csv" \
  --rf-bundle "$ISU_CHEMICAL_CANDIDATE_BUNDLE" \
  --ae-bundles "${ISU_CHEMICAL_V2_BUNDLE_ROOT:-$PWD/outputs/inference_bundles/date_prelim_20261106}" \
  --data-root "$PWD" --data-kind actual --output "$loose_run"
```

기존 61개 Test도 함께 실행하려면 위 명령에 `--test-predictions "$ISU_CHEMICAL_CANDIDATE_BUNDLE/test_predictions.csv"`를 추가한다. 중복 파일/RPM은 거부한다. 별도 Looseness만 실행할 때는 --test-predictions를 생략한다. 실제 RF가 반환되면 비교는 no_offline_reference, AE 정상이면 skipped_ae_normal이다. 최종 후보 보류는 별도 필드다. 원래의 AE/RF 추론 및 임시 DB 안전 검사는 그대로 사용한다. 운영 DB는 열지 않는다.

**독립 Test 주의:** 확장한 Looseness 파일은 Train/Validation과 중복될 수 있다. split_files.csv 및 학습 출처와 관계를 별도로 조사하기 전에는 독립 Test 성능으로 보고하지 않는다. 날짜별 분리 출력은 파일의 독립성을 증명하지 않는다.

## 3. 기존 CSV와 오류 분석

실제 MAT가 없어도 기존 integration_results.csv와 RF test_predictions.csv만 있으면 이 단계는 실행된다. 입력 CSV를 읽기만 하며 MAT를 재실행하지 않는다. 서로 다른 입력의 성능을 합치지 않고 source별로 분리한다.

```bash
read -r -p '기존 61개 integration_results.csv 절대 경로: ' existing_integration
"$HOME/capstone-web-venv/bin/python" -B scripts/analyze_v2_model_errors.py \
  --integration "$existing_integration" \
  --integration "$loose_run/integration_results.csv" \
  --offline-rf "$ISU_CHEMICAL_CANDIDATE_BUNDLE/test_predictions.csv" \
  --output "$error_report"
```

기존 CSV만 조사하려면 Looseness --integration 줄을 제외한다. 옵션은 반복 가능하나 같은 소스 경로를 두 번 입력하면 거부한다. CSV에는 date/condition/path/rpm이 필요하다. 알려진 조건만 범주로 매핑하고 잘못된 날짜, 비유한 점수/특징, 판정 불일치, 잘못된 RF gate는 제외 사유를 남긴다. 날짜/RPM/조건 라벨의 진위를 CSV만으로 입증하지는 못한다.

## 출력

- condition_coverage.csv: source·날짜·900/1200/1800RPM·원래 조건별 파일 수, 표본 0 포함
- ae_error_analysis.csv: AE 점수/저장 임계값/판정, 정상 오탐 또는 결함 미탐, 특징량
- rf_error_analysis.csv: 원시 예측·정답 범주·오분류 여부와 최종 후보/보류/사유 분리
- rpm_condition_summary.csv: source·날짜·RPM·조건별 AE 표본/FPR/미탐률 및 RF 원시 성능
- rf_confusion.csv 및 rf_confusion_*.png: RPM·날짜·source별 원시 RF 혼동행렬
- ae_distribution_*.png: 날짜/조건별 점수 분포와 **저장된** 임계값
- ae_feature_*.png 및 features_*.png: 1X/2X/RMS 관계 및 특징 분포
- excluded_rows.csv, analysis.json: 오류 제외 및 입력 해시, 지정 사례 실제 행
- MODEL_ERROR_ANALYSIS.md: 사실/가설/한계와 개선 미검증 명시

출력은 Git 저장소 밖 새 폴더만 허용하고 기존 결과는 덮어쓰지 않는다. 학습이나 임계값 선택 기능은 없다. None/공란 비율은 표본 없음이며 0%로 해석하지 않는다. RF 혼동행렬에서 표본 없는 셀 0은 검증 성공을 뜻하지 않는다.

## 분모와 해석

- 정상 FPR = 정상 파일 중 AE 이상의 비율. 결함 미탐률 = 해당 결함 파일 중 AE 정상의 비율. 유효 AE 기록만 분모에 포함하고 제외 행은 별도 기록한다.
- 웹 RF는 AE 이상에서만 실행하므로 원시 RF 정확도는 gate된 부분집합이다. 오프라인 전체 RF 정확도와 직접 비교하지 않는다.
- 오프라인 CSV의 accepted=False는 기록된 보류 여부다. 원시 예측을 최종 후보명으로 변환하지 않는다. 최종 후보명이 CSV에 없으면 기록 없음으로 둔다.
- 1800RPM Unbalance_-1, 20260919_132506_Vib.mat의 0.375726 < 0.445470은 사용자 관측값이다. 보고서는 실제 입력 CSV에서 해당 행이 있는지 확인하며 값을 실행 결과로 주입하지 않는다. 이는 기존 판정 규칙상 미탐/ RF 생략과 일관되지만, 주파수 특징이 왜 복원오차로 이어졌는지는 추가 근거가 필요하다.
- 900/1200RPM Unbalance→Looseness, 1800RPM Unbalance→Unbalance와 베어링 하위 조건의 결과는 실제 혼동행렬을 확인한다. 사용자 관측을 모든 데이터에 적용하지 않는다.
- 날짜/부하 차이, 정상·결함 특징 중첩, 약한 결함 강도, 모델 표현력은 의심 가능한 가설이다. 분포·상관만으로 물리적 원인 또는 개선 성능을 확정하지 않는다.

## 자동 검증 범위

합성 CSV와 합성 MAT 테스트만 수행한다. 기존 합성 RF 아티팩트는 읽기 전용으로 사용하고 RF.fit을 금지한다. 실제 사용자 MAT 결과는 이 환경에서 생성하지 않는다. 합성 테스트를 실제 결함 검출 성능으로 표현하지 않는다. Matplotlib 캐시 경고가 있으면 쓰기 가능한 MPLCONFIGDIR을 지정할 수 있으며 모델 계산에는 영향을 주지 않는다.

## 병합 전

- 기존 모델/임계값/보류 기준 해시 불변 확인
- 실제 Looseness 세 조건 × 세 RPM × 날짜별 coverage 및 제외 사유 확인
- 기존 Train/Validation/Test 중복 관계 확인
- 지정 Unbalance 파일의 기존 점수·임계값·RF 생략과 CSV 일치 확인
- raw RF 혼동과 최종 보류를 별도로 해석
- 잘못된 CSV·비유한값 제외, 원본/운영 DB 보존 확인
- 이 결과만으로 모델 개선이나 원인 확정을 주장하지 않음

Codex에서 신규 7개와 기존 47개, 총 54개 합성/회귀 테스트가 통과했다.
