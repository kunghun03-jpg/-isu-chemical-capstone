# 날짜·RPM 간 결함 후보 분류 오프라인 평가

## 목적과 현재 확인 범위

EDA의 최상위 `file_features.csv`와 `analysis.json`만 읽어 특징량의 날짜별 변화를 요약하고, 짧은 학습 규칙과 고정 Random Forest(RF)를 동일 시험 파일에서 비교한다. 원본 MAT가 없어도 실행된다. 기존 AE·DAE·OCSVM·IF, 모델 번들, 정규화, 임계값, Streamlit, DB는 읽거나 변경하지 않는다. 새 후보 분류기의 학습은 오프라인 실험이며, 정상만 학습한 AE 이상 탐지와 별개다.

유효한 EDA 파일 전체의 네 범주 분류를 평가하며 AE 판정으로 파일을 걸러내지 않는다. 따라서 이 결과는 AE 이상 탐지와 결함 후보 분석을 연결한 최종 시스템 전체의 성능이 아니다.

사용자 EDA 경로는 `/home/kungh/fault-signatures/20261113T131205447206Z/`이다. 이 경로와 실제 CSV는 Codex 환경에서 접근할 수 없어 **실제 분류 성능이나 날짜·RPM 간 재현성은 검증하지 못했다.** 900RPM·20260912의 고주파/1X 관측을 다른 조건의 진단 규칙으로 미리 고정하지 않는다.

## WSL 실행

실제 소스 Git 저장소에서 새 브랜치를 가져온다. 데이터 폴더와 소스 저장소가 같은 경로라고 가정하지 않는다.

```bash
cd /실제/Git/저장소
git fetch origin
git switch --track origin/feat/fault-classifier-evaluation
```

로컬 브랜치가 이미 있으면 `git switch feat/fault-classifier-evaluation`을 사용한다. 기존 로컬 변경은 강제 초기화하지 않는다. main에 병합된 뒤에는 업데이트한 main에서도 실행할 수 있다.

사용자가 지정한 실제 EDA 결과로 실행:

```bash
"$HOME/capstone-web-venv/bin/python" -B -m scripts.evaluate_fault_classifier \
  --eda-dir /home/kungh/fault-signatures/20261113T131205447206Z
```

기본 결과 위치는 `~/fault-classifier-evaluation/<UTC 실행 시각>/`이며 기존 폴더를 덮어쓰지 않는다. 명시하려면 `--output "$HOME/fault-classifier-evaluation/새폴더명"`을 추가한다. 출력은 Git 저장소 및 입력 EDA 폴더와 분리된 새 폴더여야 한다. 원본 MAT 경로가 현재 머신에 없어도 괜찮다.

실행 전:

1. 입력 폴더가 완성된 **한 번의 EDA 실행 결과**인지 확인한다. 다른 실행의 CSV·JSON 또는 RPM별 하위 CSV를 섞지 않는다.
2. 최상위 `file_features.csv`와 `analysis.json`이 모두 있는지 확인한다. `peaks.csv`와 `condition_summary.csv`는 이번 분류 입력으로 사용하지 않는다.
3. 웹 전용 환경에 NumPy·SciPy·scikit-learn·Matplotlib이 있는지 확인한다. 자동 설치·환경 변경은 하지 않는다. 클라우드에서는 Python 3.14.4, NumPy 2.4.6, scikit-learn 1.9.0으로 검사했다.
4. 날짜별 클래스 수가 매우 작거나 누락될 수 있음을 감안한다. 파일 수가 많아도 독립 Run 수와 같지 않다.
5. 이 단계에서 출력 성능을 보고 입력 특징, RF 설정, 임계값 선택법을 바꿔 같은 날짜를 최종 시험으로 다시 보고하지 않는다. 이후 변경 시 새 독립 검증이 필요하다.

## 입력 검증과 제외

완료 상태의 EDA JSON을 요구하고 CH3, 대역 경계, 엔벨로프 carrier·trim, order 탐색 범위를 검사한다. 각 CSV 행은 절대 MAT 경로를 키로 JSON의 `analyzed` 파일과 일대일 대응해야 한다.

- 경로에서 추출한 날짜·RPM·정확한 조건명이 CSV 및 JSON과 같아야 한다.
- 기록된 fs=25600Hz, 샘플 수=65536, 측정 길이=2.56초, 전체 6채널/선택 CH3를 확인한다.
- FFT 해상도, 명목 1X·2X, order 탐색 허용폭, trim 이후 엔벨로프 해상도가 분석 설정과 같아야 한다.
- 원본 MAT의 기록된 SHA256 before/after가 유효하고 일치해야 한다.
- 중복 CSV 경로는 모든 중복 행을 제외한다. 서로 다른 경로라도 같은 MAT SHA256이면 모든 해당 행을 제외해 내용 중복에 의한 누출 위험을 줄인다.
- 특징은 비음수가 되어야 하며 RF 수치 범위인 float32 최대값을 넘거나 Inf/잘못된 문자열이면 제외한다. 빈 값·NaN은 결측으로 기록하고 훈련 전용 처리에 넘긴다. 모든 특징이 결측인 행은 제외한다.
- JSON에는 있지만 CSV에 없는 분석 파일도 별도로 기록한다. 제외 사유는 `excluded_files.csv`에 남는다. 전체 입력 형식/설정이 잘못되면 평가를 중단한다.

이는 **기록된 출처의 일관성 검사**다. 원본 MAT를 다시 읽거나 특징량을 재계산하지 않으므로, CSV 값이 실제 MAT에서 정확히 계산됐음을 독립적으로 증명하지 못한다. 동일 행의 모든 특징을 묶어 사용하며 다른 파일의 특징을 join하거나 보충하지 않는다. 입력 CSV·JSON은 실행 전후 SHA256을 비교한다. 변경 감지 시 결과를 실패/무효로 표시한다.

## 클래스 및 입력 특징

| 분류 클래스 | 원래 조건명 |
| --- | --- |
| Normal | Normal1, Normal2 |
| Bearing | IR, OR, Roller |
| Looseness | Looseness0.05, Looseness0.1, LoosenessFull |
| Unbalance | Unbalance_-1, Unbalance_-2 |

원래 조건명을 모든 예측·오분류 결과에 유지한다. `Normal3`나 모호한 조건명은 포함하지 않는다.

두 방법 모두 다음 **9개 특징**만 사용한다: `rms_ac`, `peak_ac`, `crest_factor_ac`, `one_x_amplitude`, `two_x_amplitude`, `energy_low`, `energy_mid`, `energy_high`, `envelope_rms_ac`.

경로, 날짜, RPM, 원래 조건명, 정답 클래스, 해시, 피크 주파수나 진단 의미를 부여한 식별 정보는 입력 특징에 포함하지 않는다. RPM은 평가 집단을 나누는 용도로만 사용한다. 원본 EDA의 모든 수치 열을 자동 선택하지 않는다.

## 특징 분포와 정상 대비 차이

`feature_distributions.csv`는 RPM·날짜를 고정한 뒤 원래 조건별 및 네 범주별 특징의 관측/결측 수, 평균, 중앙값, P25/P75를 저장한다. 같은 RPM·날짜의 Normal1/Normal2를 합친 정상 중앙값과의 차이 및 비율을 기록한다. 정상 기준이 없거나 0이면 비율을 임의로 만들지 않는다. Normal1·Normal2의 개별 행도 제공한다. 정상 범주 통합은 파일 가중치이며 독립 표본을 추가하는 것이 아니다.

`feature_date_shift_<RPM>rpm.png`에는 각 원래 조건의 날짜별 중앙값을 보여준다. 이 요약은 **사후 기술 통계**이며 분류기의 특징 선택·임계값 결정·전처리로 전달되지 않는다. 한 날짜의 차이가 다른 날짜에서 재현되지 않으면 안정적인 근거라고 주장할 수 없다. 관측 차이가 없다는 사실도 그대로 보고해야 한다.

## 두 평가 프로토콜

| 프로토콜 | 훈련 | 평가 | 해석 |
| --- | --- | --- | --- |
| per_rpm_lodo | 같은 RPM의 나머지 날짜 | 해당 RPM의 제외 날짜 | 기존 RPM에서 새로운 날짜로 일반화 |
| unseen_rpm_lodo | 시험 RPM을 제외한 다른 RPM이면서 시험 날짜를 제외한 날짜 | 해당 RPM·날짜 | RPM·날짜 모두 훈련에서 보지 못한 조건으로 이전 |

모든 프로토콜에서 시험 날짜는 어떤 RPM에서도 훈련에 들어가지 않는다. 날짜가 하나뿐이면 훈련/평가를 억지로 나누지 않는다. 두 방법은 각 fold에서 정확히 같은 훈련 파일·시험 파일·입력 열·결측 처리값을 사용한다. 프로토콜은 서로 다른 질문이므로 결과를 합쳐 하나의 전체 성능으로 만들지 않는다.

Run 정보가 검증되지 않아 **독립성 미확인 예비 평가**다. 날짜 분리와 중복 해시 검사는 Run·설비·센서·부하의 독립성을 보장하지 않는다. 원래 EDA의 층화 샘플에 대한 결과이며 전체 모집단의 대표성도 보장하지 않는다.

## 훈련 날짜에서만 결정하는 항목

결측은 각 훈련 fold의 중앙값으로 처리한다. 훈련에서 전체가 결측인 열은 사전 고정한 0을 사용하고 해당 사실을 기록한다. 시험 결측 또는 시험 분포로 통계를 갱신하지 않는다. 축 기준 임계값 규칙과 RF 모두 표준화를 요구하지 않아 **표준화는 수행하지 않는다**. 공통 전처리에서는 고정 9개 열을 모두 유지한다. 규칙 내부의 클래스별 특징 선택은 아래와 같이 훈련 데이터에서만 수행한다.

### 해석 가능한 학습 규칙

물리적으로 확정된 결함 규칙이 아닌, 단순한 **훈련 기반 규칙 baseline**이다. Bearing·Looseness·Unbalance 각각에 대해:

1. 훈련 파일에서 해당 클래스 대 나머지 클래스를 구분한다.
2. 9개 특징, `>` 또는 `<=`, 훈련 값 사이의 경계값을 비교한다.
3. 훈련 binary balanced accuracy가 최대인 특징 1개와 임계값 1개를 선택한다. 0.5를 넘지 못하면 해당 클래스 규칙을 만들지 않는다.
4. 여러 규칙이 활성화되면 훈련 binary balanced accuracy 순으로 선택한다. 동률이면 고정 클래스 순서 Bearing → Looseness → Unbalance를 따른다. 활성 규칙이 없으면 Normal이다.

임계값은 median으로 대체한 **원래 특징 단위**다. 특성 이름·방향·임계값·훈련 품질·우선순위를 `rule_thresholds.csv`에 기록하며 파일별 활성 근거를 `predictions.csv`에 남긴다. 조건이 `energy_high > ...`로 선택돼도 고주파가 그 결함의 원인이라는 뜻은 아니다. train 품질은 시험 성능이 아니며 규칙 학습 자체도 작은 표본에 과적합할 수 있다. 규칙은 후보 확률을 출력하지 않는다.

### Random Forest

초기 설정은 사전 고정한다: 200 trees, max_depth=5, min_samples_leaf=2, class_weight=balanced, random_state=42, n_jobs=1. Grid search, 시험 날짜 기반 조정, 확률 보정은 하지 않는다. `uncalibrated_vote_fraction_*`는 RF 내부 점수이며 **신뢰할 수 있게 보정된 확률이 아니다**. 훈련에 없는 클래스의 점수 열은 비어 있을 수 있다.

특징 중요도는 각 훈련 RF의 impurity importance를 기록한다. 상관된 특징과 연속 변수의 편향이 있으므로 인과관계, 물리적 결함 원인, 독립적인 성능 증거로 해석하지 않는다.

## 지표와 부족한 데이터 처리

- Confusion matrix는 항상 Normal/Bearing/Looseness/Unbalance 순서의 4×4다.
- `macro_f1_fixed_four_classes`는 네 클래스 F1 평균이며 없는 클래스도 0으로 포함한다.
- `balanced_accuracy_present_classes`는 실제 시험 표본이 있는 클래스 recall의 평균이다. 없는 시험 클래스 목록도 기록한다. 비교 시 지원 클래스 구성이 같은지 확인한다.
- 고정 네 클래스 macro recall과 클래스별 precision/recall/F1/support도 기록한다. Accuracy만으로 결론내리지 않는다.
- 훈련에 Normal과 최소 한 결함 클래스가 없으면 두 방법 모두 해당 fold를 건너뛴다. 일부 결함 클래스가 훈련에 없더라도 시험 파일은 제거하지 않으며 누락 사실과 결과를 그대로 기록한다.
- 날짜 하나만으로 훈련하거나 클래스당 5개 미만이면 불안정성 경고를 기록한다. 5개가 통계적 충분성을 보장하는 기준은 아니다.
- `overall`은 **해당 프로토콜에서 실제 평가된 파일의 out-of-fold 예측을 합친 지표**다. fold F1의 단순평균이 아니다. `coverage_by_protocol` 및 건너뛴 fold를 함께 확인해야 한다. 낮은 성능을 숨기거나 누락 클래스만 제외해 높게 표시하지 않는다.

`condition_errors.csv`에서 Normal1/Normal2, Unbalance_-1/Unbalance_-2, 각 Looseness 조건별 오류 수·비율·예측 클래스 분포를 볼 수 있다. `Unbalance_-1`을 별도로 살피되 이름만으로 실제 불평형 심도가 약하다고 확정하지 않는다. Normal이 결함으로 분류된 사례와 결함 종류끼리 혼동한 사례는 `misclassified_files.csv`에서 구분해 볼 수 있다.

## 출력 파일

```text
<새 출력 폴더>/
  evaluation.json                   # 출처·설정·fold 학습/시험 경로·누락·한계·전체 지표
  validated_files.csv               # 원래 조건·범주·출처·결측 현황
  excluded_files.csv                # 제외 행과 사유
  feature_distributions.csv         # RPM·날짜·조건별 분포와 동일 조건의 정상 대비 차이
  feature_date_shift_<RPM>rpm.png
  training_imputation.csv           # fold별 훈련 중앙값, 전체 결측 열 표시
  rule_thresholds.csv               # fold별 규칙, 임계값, 방향, 선택 근거
  rf_feature_importance.csv          # 훈련 RF 중요도
  predictions.csv                   # 공통 시험 파일의 두 방법 예측과 근거/비보정 점수
  misclassified_files.csv
  condition_errors.csv              # 원래 조건별 세부 오분류
  metrics.csv                       # overall/rpm/date/rpm_date별 지표, 프로토콜 분리
  class_metrics.csv                 # 같은 구분의 클래스별 precision/recall/F1/support
  confusion_matrices.csv            # 같은 구분의 4×4 행렬을 long format으로 저장
  confusion_<protocol>_<model>.png   # 프로토콜별 전체 평가 행렬
```

종료 코드: 0은 평가 결과 작성, 2는 데이터 부족으로 평가 미수행, 1은 실행 오류다. 일부 fold가 생략돼도 나머지 평가가 있으면 0이므로 `coverage_by_protocol`, `skipped_folds`, 개별 fold 제한을 반드시 확인한다. 출력 실패 또는 입력 변경 시 `evaluation.json`의 `status`를 확인한다. 입력 검증 전에 중단되면 결과 폴더가 생성되지 않을 수 있다.

## 테스트와 병합 전 확인

```bash
MPLCONFIGDIR=/tmp/fault-classifier-mpl \
  "$HOME/capstone-web-venv/bin/python" -B -m unittest tests.test_fault_classifier -v
```

합성 CSV/JSON만으로 입력 대응·중복·오류 제외, 날짜/RPM 분리, 훈련 전용 결측 처리, 규칙 해석, 시험 데이터를 바꿔도 훈련 산출물이 유지되는지, 누락 클래스, 지표, 공통 시험 목록, CSV/JSON/PNG 생성과 읽기 전용 동작을 검사한다. 실제 분류 정확도 검증으로 표현하지 않는다.

병합 전에는 추가 파일 3개만 포함되는지, 기존 EDA·모델·웹·DB 변경이 없는지 확인한다. 코드 병합과 실제 성능 승인/웹 채택은 별개다. WSL에서 실제 CSV 검증·제외 사유·평가 범위를 확인하고, 낮은 성능 및 날짜/RPM 간 실패를 포함해 보고한 다음 웹 연결 여부를 결정해야 한다.

2026-11-13 클라우드 검증: Python 3.14.4, NumPy 2.4.6, scikit-learn 1.9.0에서 신규 12개 + 기존 EDA 11개 + V2 6개 + AE 경로 비교 6개로 총 35개 테스트 통과. 실제 사용자 CSV는 없어 합성 입력만 검사했다. 기존 V2 테스트에서 Streamlit 실행 문맥 및 DB 연결 종료 관련 경고가 출력됐으나 테스트 실패는 없었다.
