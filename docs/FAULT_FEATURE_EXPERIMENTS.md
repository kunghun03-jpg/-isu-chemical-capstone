# 날짜 변화·느슨함 중첩과 회전 고조파 특징 실험

## 목적과 해석 범위

기존 실데이터 결과에서 Looseness Recall이 거의 0이고 날짜별 예측이 달랐다는 사용자 관측을 후속 조사한다. **원인 확정이나 웹 통합이 아니라 탐색적 오프라인 실험**이다. 날짜별 특징 변화가 Normal/Looseness 차이보다 큰지 기술 통계로 비교하고, 회전 고조파·상대 에너지를 추가했을 때 같은 시험 파일에서 개선·악화가 나타나는지 평가한다.

실제 WSL 입력:

- EDA: `/home/kungh/fault-signatures/20261113T131205447206Z/`
- 기존 분류 평가: `/home/kungh/fault-classifier-evaluation/20261113T133757285078Z/`

이 경로와 실제 MAT는 Codex 환경에 없어 실제 분석·분류 성능은 재검증하지 못했다. 사용자가 제공한 239개/18 folds 및 Macro F1·Recall은 외부 실행 기록이며 이번 코드의 실행 결과가 아니다. **고조파가 Looseness를 개선했다고 아직 주장할 수 없다.**

이전 시험 결과를 보고 특징을 추가한 후속 실험이므로 동일 날짜의 재평가는 새로운 최종 시험이 아니다. 모델 채택 전 새로운 날짜·독립 Run 평가가 필요하다. 날짜 분리가 센서·부하·설비·측정 순서의 독립성이나 인과관계를 보장하지 않는다.

## 변경 범위

- `scripts/analyze_feature_date_shift.py`: CSV만으로 날짜 변화·정상/느슨함 중첩 분석.
- `scripts/extract_harmonic_features.py`: 원본 MAT를 읽고 확장 특징 생성.
- `scripts/compare_fault_feature_sets.py`: 전체 baseline 재현과 공통 파일의 paired 비교.
- `scripts/fault_feature_common.py`: 새 출력 폴더, 해시, 버전, 오류 보고 공통 처리.
- `scripts/evaluate_fault_classifier.py`: 기존 규칙·학습·평가 함수에 선택적 특징 목록 인자를 추가. 기본값은 기존 9개이며 계산·RF 설정·fold 생성·지표 정의는 유지.
- `tests/test_fault_feature_experiments.py`: 합성 검증.

AE·DAE·OCSVM·IF 코드/가중치/정규화/임계값, Streamlit, DB 및 원본 MAT·EDA·기존 평가 산출물은 변경하지 않는다. 학습된 실험 모델 파일을 배포하거나 저장하지 않는다.

## WSL 실행 전 확인

1. 현재 터미널이 **실제 소스 Git 저장소**인지 `git rev-parse --show-toplevel`로 확인한다. 데이터 폴더와 소스 폴더가 같다고 가정하지 않는다.
2. 기존 웹 전용 `~/capstone-web-venv/bin/python`과 NumPy·SciPy·scikit-learn·Matplotlib을 사용한다. 이 도구는 환경을 설치하거나 바꾸지 않는다.
3. EDA 최상위 `file_features.csv`와 `analysis.json`은 같은 실행의 원본이어야 한다. 기존 검증 로더로 경로/RPM/날짜/정확한 조건/측정 설정/기록된 MAT SHA256을 대조한다.
4. 확장 추출은 JSON에 기록된 절대 MAT 경로를 그대로 사용한다. MAT가 이동했거나 사라졌으면 값을 추정하지 않고 제외한다. 같은 진동값을 재저장한 MAT도 바이트 SHA256이 달라지면 제외한다.
5. 기존 평가 폴더의 `evaluation.json`, `predictions.csv`도 필요하다. baseline 재현 검사는 같은 EDA 해시, 9개 특징 및 RF 설정을 요구한다.
6. 출력은 Git과 입력 폴더 밖의 새 경로다. 기존 출력 폴더를 재사용하지 않는다. 아래 명령은 한 실행 전용 부모 아래 3개의 새 하위 폴더를 만든다.

## 정확한 실행 순서

소스 저장소에서 브랜치를 가져온다. 이미 로컬 브랜치가 있으면 `git switch feat/harmonic-date-generalization`을 사용한다. 로컬 변경을 강제 초기화하지 않는다.

```bash
git fetch origin
git switch --track origin/feat/harmonic-date-generalization
```

같은 WSL 터미널에서 아래 블록을 실행한다. 마지막 단계가 끝날 때까지 `experiment_dir` 값을 유지한다. 각 명령이 실패하면 다음 단계로 넘어가지 말고 해당 JSON/제외 사유를 확인한다.

```bash
experiment_dir="$HOME/fault-feature-experiments/$(date -u +%Y%m%dT%H%M%S%NZ)-harmonic"

# 1. 기존 9개 특징의 날짜 변화·중첩 분석: 원본 MAT 불필요
"$HOME/capstone-web-venv/bin/python" -B -m scripts.analyze_feature_date_shift \
  --eda-dir /home/kungh/fault-signatures/20261113T131205447206Z \
  --output "$experiment_dir/01-date-shift"

# 2. 원본 MAT에서 고조파를 재측정: MAT 경로·SHA256 및 기존 9개 특징 재검사
"$HOME/capstone-web-venv/bin/python" -B -m scripts.extract_harmonic_features \
  --eda-dir /home/kungh/fault-signatures/20261113T131205447206Z \
  --output "$experiment_dir/02-harmonics"

# 3. 전체 baseline 재현 + 공통 파일의 9개 vs 23개 특징 비교
"$HOME/capstone-web-venv/bin/python" -B -m scripts.compare_fault_feature_sets \
  --eda-dir /home/kungh/fault-signatures/20261113T131205447206Z \
  --extension-dir "$experiment_dir/02-harmonics" \
  --reference-evaluation /home/kungh/fault-classifier-evaluation/20261113T133757285078Z \
  --output "$experiment_dir/03-comparison"

# 4. 전체 지표와 주요 Recall의 baseline / extended / delta 확인
"$HOME/capstone-web-venv/bin/python" -B - "$experiment_dir/03-comparison" <<'PY'
import csv, json, sys
from pathlib import Path
p = Path(sys.argv[1])
report = json.loads((p / "comparison.json").read_text())
print("status:", report["status"])
print("reference:", report["reference_reproduction"]["status"])
print("common/EDA:", report["common_files"], "/", report["validated_eda_files"])
print("evaluated/skipped folds:", report["evaluated_folds"], "/", report["skipped_folds"])
wanted = {"macro_f1", "balanced_accuracy", "Looseness_recall", "Normal_recall", "Unbalance_recall"}
with (p / "overall_summary.csv").open(encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        if r["metric"] in wanted:
            print(r["protocol"], r["model"], r["metric"],
                  "baseline=" + r["baseline"], "extended=" + r["extended"], "delta=" + r["delta"])
print("RPM/date detail:", p / "metric_deltas.csv")
print("Failures:", p / "failure_cases.csv")
PY
```

종료 코드 0은 해당 단계의 결과 생성, 2는 유효 데이터·평가 부족 또는 baseline 재현 불일치, 1은 설정·실행 오류다. `completed`, `inputs_unchanged`, `status`가 실패면 CSV가 있어도 유효한 결과로 사용하지 않는다. 추출에서 일부 MAT가 제외되어도 나머지가 있으면 0이므로 숫자와 사유를 꼭 확인한다. `--reference-evaluation`은 생략 가능하지만 그러면 과거 baseline 재현은 미확인으로 남는다.

## A. 날짜 변화와 Normal/Looseness 중첩

RPM·날짜·원래 조건을 유지하고 Normal1/Normal2를 구분한다. 같은 RPM·날짜 안에서만 NormalCombined와 LoosenessCombined를 파일 가중치로 만든다. 각 느슨함 심도와 각 정상 조건도 별도 비교하므로 통합 결과만 보고 결론내리지 않는다.

`normal_looseness_overlap.csv`:

- 같은 RPM·날짜에서 Normal1/Normal2/NormalCombined 각각 대 Looseness0.05/0.1/Full/Combined를 비교.
- 실제 관측 수, 소표본 경고, 중앙값, Wasserstein 거리(raw 특징 단위), KS 거리, 10-bin histogram overlap, Cliff's delta를 기록.
- Histogram overlap은 두 표본의 공통 최솟값~최댓값을 10등분한 bin 확률의 최소값 합이다. 1은 같은 bin 비율, 0은 겹치는 bin 없음이다. **표본이 몇 개뿐이면 같은 모집단에서도 0에 가까워질 수 있다.** 분류 정확도나 모집단의 분리 확률이 아니다. 쌍마다 bin 경계가 달라 작은 차이의 순위를 과도하게 비교하지 않는다.
- Cliff's delta는 `P(Looseness>Normal)-P(Looseness<Normal)`이고 ties는 우월 확률에 절반 반영한다. 결함 원인이나 검증된 분류 성능이 아니다.

`within_condition_date_shift.csv`: 같은 RPM·같은 원래 조건에서 가능한 날짜 쌍의 거리와 분포 차이. Normal1/2 및 통합 정상/느슨함도 구분한다.

`date_vs_fault_distance.csv`: 특징별로 Normal1, Normal2 및 선택한 느슨함 조건의 날짜 간 Wasserstein 거리 중앙값과, 같은 날짜의 NormalCombined 대 해당 느슨함 조건 거리 중앙값을 비교한다. 비율이 1보다 크면 이 요약에서는 날짜 변화가 더 크다는 뜻이다. 인과적 "날짜 민감도"나 유의성 검정은 아니다. 결함 거리가 0이면 비율은 결측으로 남기며 두 원래 거리를 직접 확인한다. 서로 다른 단위의 raw 거리를 특징 간 직접 비교하지 않는다.

분포 요약·날짜별 PNG 및 overlap heatmap을 함께 저장한다. 전체 EDA를 사용한 기술 통계는 분류기의 학습·전처리·임계값에 전달하지 않는다. 정상 측정 구성이 날짜마다 달라지는 영향도 있을 수 있다.

## B. 원본 MAT 확장 특징

기존 `preview_mat` 계약인 signals=(6,65536), fs=25600Hz, 전체 6채널 품질 검증, CH3 선택을 유지한다. 기존 `analyze_signal`의 평균 제거·전체 기록 대칭 Hann FFT·단측 coherent gain 보정·PSD·엔벨로프를 재사용한다. AE 전처리/점수 경로는 호출하지 않는다.

1X~5X는 폴더 RPM/60의 배수이며 **실제 회전수 계측이나 order tracking이 아니다**. Δf=0.390625Hz, 탐색 허용폭은 원본 EDA 설정(기본 ±1Hz)으로 고정한다. 그 범위 안에서 최대 진폭 bin을 측정하고 실제 bin 주파수도 JSON에 기록한다. `peaks.csv`는 사용하지 않는다.

기존 9개 + 새 14개 = **23개 특징**:

| 추가 특징 | 정의 |
| --- | --- |
| three/four/five_x_amplitude | A3, A4, A5 |
| two/three/four/five_to_one_ratio | A2/A1, A3/A1, A4/A1, A5/A1 |
| harmonic_2to5_amplitude_sum | S=A2+A3+A4+A5 |
| harmonic_2to5_amplitude_squared_sum | A2²+A3²+A4²+A5²; PSD 적분 에너지와 구분 |
| harmonic_sum_over_1x | S/A1 |
| harmonic_2to5_amplitude_fraction | S/(A1+S) |
| energy_low/mid/high_fraction | 각 기존 대역 평균제곱 / 세 대역 합 |

에너지 비율 분모는 기존 [0,200), [200,2000), [2000,10000]Hz 세 대역 합이다. 10kHz 초과~Nyquist 에너지는 포함하지 않는다.

1X 비율은 `A1 > max(1e-12 raw units, max(A1..A5)*1e-6)`일 때만 정의한다. 작은 분모를 epsilon으로 대체해 거대한 값을 만들지 않는다. 미정의 값은 결측으로 저장하고 이후 훈련 날짜의 median으로만 처리한다. 세 대역 합이 1e-24 이하이면 에너지 비율도 결측이다. 이는 사전에 고정한 수치 정책이며 테스트 분포로 최적화하지 않는다. 센서 단위가 미확인이라는 한계도 남는다.

원본 MAT SHA256이 EDA 기록과 다르거나 재계산한 기존 9개 특징이 기존 값과 rtol=1e-6/atol=1e-9 범위에서 일치하지 않으면 제외한다. baseline 값은 재계산 값으로 바꾸지 않고 기존 CSV 값을 그대로 보존한다. 결측이 있는 비율 때문에 파일 자체를 탈락시키지는 않는다. 추출된 CSV SHA256, 파일별 재계산/분모 정보, 제외 사유를 JSON에 저장한다. CSV/JSON 출처·값이 일치하지 않으면 비교 도구는 중단한다.

## C. 동일 파일·fold 비교

1. `baseline_all`: 검증된 원래 EDA 파일 전체의 기존 9개 특징. 원래 평가 재현용.
2. `baseline_common`: 원본 MAT 확인과 확장 추출에 성공한 공통 파일에서 기존 9개 특징.
3. `extended_common`: **정확히 같은 공통 파일·정답·학습/시험 fold**에서 23개 특징.

개선/악화 비교는 2 대 3만 사용한다. 1 대 2 차이는 파일 집합이 달라진 영향이며 특징 개선이 아니다. MAT 누락·오류·기존 특징 불일치로 빠진 파일을 숨기지 않는다. 공통 시험 수와 skipped fold를 반드시 확인한다.

기존 `evaluate_rows`, `fold_specs`, 학습 규칙, median imputer, RF, 지표를 재사용한다. per_rpm_lodo와 unseen_rpm_lodo(시험 RPM·시험 날짜 모두 훈련에서 제외)를 유지한다. 두 paired 변형의 fold 내용과 시험 정답을 코드에서 대조한다.

규칙 특징·방향·임계값, 결측 처리·전체 결측 열 fallback 0은 훈련 날짜에서만 결정한다. 두 변형 모두 별도 표준화나 공통 특징 선택을 하지 않는다. RF는 기존 200 trees/max_depth=5/min_samples_leaf=2/class_weight=balanced/seed=42/n_jobs=1 설정을 유지한다. 기본 max_features=sqrt도 같으며 특징 수가 늘면 실제 후보 특징 수가 달라진다는 점을 기록한다. 시험 결과를 보고 설정을 조정하거나 유리한 특징 부분집합을 고르지 않는다.

원래 평가와 재현 비교는 파일별 `(protocol, model, fold, path)`의 정답·예측을 비교한다. 불일치면 `reference_baseline_mismatch_review_required`로 표시하고 차이 CSV를 저장한다. 버전·파일 집합·환경을 조사하기 전 재현 성공으로 주장하지 않는다. RF의 투표 값은 보정된 확률이 아니고 중요도는 인과관계가 아니다. 이 평가는 AE로 걸러내지 않은 후보 분류이며 최종 시스템 전체 성능이 아니다.

## 출력과 결과 확인 순서

```text
01-date-shift/
  date_shift.json
  feature_distributions.csv
  normal_looseness_overlap.csv
  within_condition_date_shift.csv
  date_vs_fault_distance.csv
  feature_date_shift_<RPM>rpm.png
  normal_looseness_overlap_<RPM>rpm.png
02-harmonics/
  harmonic_extraction.json
  extended_file_features.csv
  extraction_exclusions.csv
  eda_validation_exclusions.csv
03-comparison/
  comparison.json
  common_cohort.csv
  not_in_common_cohort.csv
  overall_summary.csv
  metric_deltas.csv
  paired_predictions.csv
  failure_cases.csv
  condition_transitions.csv
  reference_prediction_differences.csv
  paired_metrics_<protocol>.png
  baseline_all/                      # 아래 baseline_common/, extended_common/도 동일
    predictions.csv
    folds.json
    metrics.csv
    class_metrics.csv
    confusion_matrices.csv
    confusion_<protocol>_<model>.png
    training_imputation.csv
    rule_thresholds.csv
    rf_feature_importance.csv
```

먼저 해시·baseline 재현과 평가 범위를 확인한다. 그다음 `overall_summary.csv`에서 두 프로토콜을 **분리하여** Macro F1, Balanced Accuracy, Looseness/Normal/Unbalance Recall을 확인한다. `metric_deltas.csv`에는 overall/RPM/date/RPM·date별 모든 클래스 precision/recall/F1 차이가 있다. delta는 extended−baseline이며 음수도 그대로 저장한다. 기존 도구와 동일하게 없는 클래스는 fixed-four macro F1에 0으로 포함하고 balanced accuracy는 시험에 실제 존재하는 클래스 평균을 사용하므로 클래스 support도 확인한다.

`failure_cases.csv`는 개선·악화·계속 오답을 파일별 기록하고 `condition_transitions.csv`는 원래 조건별 전이를 센다. Normal1/2, 각 느슨함 조건, Unbalance_-1/-2에서 실패가 특정 날짜·RPM에 몰리는지 확인한다. 전체 평균 상승만으로 채택하지 않는다. Looseness 개선과 함께 Normal Recall이 무너지는 등 trade-off도 보고한다. 성능이 악화되거나 차이가 작으면 그대로 보고하며 통계적 검증 없이 유의한 개선이라고 쓰지 않는다.

## 자동 테스트와 병합 전 확인

```bash
MPLCONFIGDIR=/tmp/fault-experiment-mpl \
  "$HOME/capstone-web-venv/bin/python" -B -m unittest \
  tests.test_fault_feature_experiments tests.test_fault_classifier \
  tests.test_fault_signatures tests.test_dashboard_v2 tests.test_ae_v2_parity -v
```

합성 데이터로 중첩/날짜 변화, 정확한 고조파/비율, 작은 분모·비유한값, 원본 MAT 해시와 기존 특징 일치, 누락 MAT에서 미생성, baseline 기본 인자 유지, 날짜/RPM 분리, 시험 값을 바꿔도 훈련 산출물 유지, paired 파일 일치, 음수 delta 보존, 기존 평가 재현, 출력·입력 보호를 검사한다.

병합 전 변경 파일만 포함되는지와 기존 미커밋 작업이 보존됐는지 확인한다. 실제 사용자 EDA·MAT로 위 순서를 실행하고, baseline 재현·공통 코호트·Recall trade-off·날짜/RPM별 실패를 확인한 후 최종 후보 모델 채택을 결정한다. 웹 통합은 이번 범위가 아니다.

2026-11-13 클라우드 검증: Python 3.14.4 / NumPy 2.4.6 / SciPy 1.18.1 / scikit-learn 1.9.0에서 신규 12개 + 기존 분류 12개 + EDA 11개 + V2 6개 + AE 경로 비교 6개, 총 **47개 테스트 통과**. 기존 V2 테스트의 Streamlit 실행 문맥·테스트 DB 연결 종료 경고가 있었으나 실패는 없었다. 테스트 MAT·CSV·PNG는 임시 합성 자료이며 실제 사용자 성능 검증 결과가 아니다.
