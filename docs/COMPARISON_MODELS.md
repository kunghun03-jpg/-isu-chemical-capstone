# OCSVM·Isolation Forest 비교 모델

3개 날짜 데이터 확보와 품질검사를 완료했고, 2026-11-06 승인된 날짜 그룹 예비 실험으로
AE·OCSVM·IF 실제 학습·평가를 완료했습니다. [결과](DATE_PRELIMINARY_RESULTS.md)를 확인하세요.
아래의 Run 실험 설명과 명령은 향후 독립 Run 본 실험 기준이며, 날짜 예비 모드는 별도로 구분합니다.
단계별 선행 조건은 [개발 계획](DEVELOPMENT_PLAN.md)을 따릅니다.

## 입력과 기본 설정

기존 FFT → log1p → 정상 학습 기준 Z-score 결과를 그대로 사용합니다.
`(구간 수, 1, 3201)`에서 채널 축만 제거하여 `(구간 수, 3201)`로 변환합니다.
AE와 같은 주파수 정보로 비교하기 위해 축약·특징 선택·추가 정규화는 하지 않습니다.
전체 스펙트럼이 OCSVM·IF의 최적 특징이라고 검증된 것은 아닙니다.
OCSVM은 학습 구간 수 증가에 따라 시간·메모리 비용이 커질 수 있습니다.

| 모델 | 고정 설정 |
| --- | --- |
| OCSVM | kernel=rbf, nu=0.05, gamma=scale, tol=0.001, cache_size=200MB |
| IF | n_estimators=300, max_samples=auto, contamination=auto, max_features=1.0, bootstrap=False, random_state=42, n_jobs=1 |

정상 train 구간만 fit합니다. OCSVM은 난수를 사용하지 않는 설정이며 IF는 시드·작업 수를 고정했습니다.
gamma=scale은 정상 학습 특징만으로 계산됩니다. nu=0.05는 고정 학습 설정이며 결함 비율이나 최종 Q95 임계값이 아닙니다.
전체 sklearn get_params, 표본 수, Python·NumPy·sklearn·joblib 버전을 config.json에 보존합니다.
정확한 재현에는 환경 버전도 맞춰야 합니다. 결함 데이터로 튜닝하지 않았습니다.

## 이상점수·임계값·평가 단위

- 점수: 두 모델 모두 `-model.score_samples(X)`. 클수록 이상입니다.
- 내장 predict 경계는 최종 판정에 사용하지 않습니다.
- 구간 임계값: 정상 validation 구간 점수의 Q95.
- 파일 점수: 파일별 8개 구간 점수의 산술평균. AE 파일 평가와 같은 집계입니다.
- 파일 임계값: 정상 validation 파일 평균 점수의 Q95.
- 점수가 해당 단위 임계값보다 클 때만 Abnormal, 같거나 작으면 Normal.

결함 시험 데이터는 점수·지표 계산과 데이터 일치 검사에만 사용하고 임계값·파라미터 선택에 사용하지 않습니다.
구간 임계값과 파일 임계값은 서로 다르므로 바꿔 쓰지 않습니다.
모델별 이상점수나 임계값의 숫자 크기를 직접 비교하지 않습니다.

## Run 차단과 공정 비교

세 모델의 진입점은 동일한 split CSV와 네 분할의 normalized NPZ를 사용합니다.
Run 모드의 확정 ID·중복 배정 또는 날짜 예비 모드의 고정 날짜 배정, 파일 누락·중복, 각 파일의 0~7 구간,
NPZ의 날짜·RPM·상태·Run ID 일치를 학습 전에 검사합니다.
이는 입력된 Run 배정의 검사이며 실제 운전 독립성은 실험 기록으로 확인해야 합니다.

분할 목록과 정규화 배열의 식별값을 모델·결과에 저장합니다.
다른 분할·재정규화 데이터·오래된 결과가 섞이면 비교를 중단합니다.
AE는 기존 구조·손실·학습 설정·CSV 이름을 유지하고 Run 검사와 식별값만 추가했습니다.
식별값이 없는 예전 AE 체크포인트는 새 비교에 사용할 수 없습니다.

## 실행 명령

아래는 책임님 피드백과 Run 확정 후 본 실험에 사용할 명령입니다.
날짜 예비 실험에서는 같은 모델 단계를 별도 date_preliminary 설정으로 실행했습니다.
같은 RPM·run-name으로 check → split → spectra → normalize를 먼저 완료해야 합니다.

```powershell
python run_project.py --rpm 1800 --stage ocsvm --run-name ch3_runs
python run_project.py --rpm 1800 --stage iforest --run-name ch3_runs
```

각 명령은 정상 학습 → 정상 검증 임계값 → 점수 저장 → 시험 지표 계산을 한 번에 수행합니다.
AE는 기존 train → evaluate → file_evaluate 순서를 사용합니다.

```powershell
python run_project.py --rpm 1800 --stage compare --run-name ch3_runs
```

compare는 학습 없이 세 모델의 기존 평가 결과만 비교합니다.
모든 모델의 시험 파일·구간·상태·데이터 식별값과 판정 일관성이 같아야 합니다.
정상/결함 시험이 모두 있어야 하며 학습·검증 점수는 최종 시험 지표에 포함하지 않습니다.

## 결과 위치

`outputs/{RPM}rpm/{실험이름}/comparisons/ocsvm/` 및 `comparisons/iforest/`:

| 파일 | 내용 |
| --- | --- |
| model.joblib | 학습 모델·설정·구간/파일 임계값 |
| config.json | 특징·전체 파라미터·환경 버전·데이터 식별값·표본 수 |
| thresholds.json | segment·file Q95 |
| scores_segment.csv | 네 분할 전체의 구간 점수·메타데이터·판정 |
| scores_file.csv | 네 분할 전체의 파일 점수·메타데이터·판정 |
| metrics.json | 구간·파일별 시험 지표 |

AE 결과는 기존 CSV 이름을 사용하고 `evaluation_segment_context_{RPM}rpm.json`,
`evaluation_file_context_{RPM}rpm.json`에 데이터 식별값을 저장합니다.
세 모델 비교는 실험 폴더의 `comparison_summary.json`에 저장합니다.
불러오기는 `one_class.load_model(path, fingerprint=...)`로 현재 실험을 확인합니다.
joblib은 이 프로젝트에서 생성한 신뢰할 수 있는 모델 파일만 불러옵니다.

지표는 정상 오탐률·specificity·결함 탐지율·상태별 이상 판정 비율·AUROC·AUPRC입니다.
AUPRC는 `average_precision_score`의 Average Precision이며 사다리꼴 PR 면적과 구분합니다.
비율은 0~1로 저장합니다. 정상 상태별 판정 비율은 오탐률, 결함 상태별 비율은 탐지율입니다.
주 결과는 파일 단위로 제시하고 구간 단위는 보조로 봅니다. 구간을 독립 Run으로 해석하지 않습니다.

## 테스트와 다음 단계

```powershell
python -m unittest discover -s tests -v
python -m tests.smoke_test_ch3
```

비교 모델 검사는 임시 폴더의 가상 CH3 배열만 사용합니다. AE 비교 연결은 초기 가중치 추론으로
검사하며 AE를 재학습하지 않습니다. 실제 프로젝트는 미확정 Run으로 실행이 차단되는지만 확인합니다.
가상 검사 지표를 실제 설비 성능으로 보고하지 않습니다.

책임님 피드백 확인 후 실제 Run 확인 → 공통 분할·품질검사 반영·전처리 → 세 모델 학습/평가 →
동일 시험 파일 기준 비교 → 그림·표 작성이 남아 있습니다.
