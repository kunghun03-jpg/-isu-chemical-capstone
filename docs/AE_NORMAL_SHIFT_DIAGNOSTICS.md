# 고정 AE 정상 오탐 원인 진단

기존 모델·mean/std·Q95·전처리·학습 산출물을 읽기 전용으로 사용합니다.
새 가중치·임계값·정규화 기준을 만들지 않고 Test 기반 임계값 탐색도 하지 않습니다.
이 도구는 WSL 실제 NPZ에서 실행해야 하며 클라우드에서는 합성 자료로만 테스트했습니다.

## 실행

Git 프로젝트 루트에서 웹 전용 Python으로 실행합니다.

```bash
"$HOME/capstone-web-venv/bin/python" -B -m scripts.analyze_ae_normal_shift
```

기본 입력은 `outputs/{900,1200,1800}rpm/date_prelim_20261106/`와
`outputs/inference_bundles/date_prelim_20261106/{900,1200,1800}rpm/`입니다.
원본 학습 산출물이 다른 폴더에 있다면 다음처럼 지정합니다.

```bash
"$HOME/capstone-web-venv/bin/python" -B -m scripts.analyze_ae_normal_shift --outputs-root "/실제/outputs" --bundle-root "/실제/inference_bundles/date_prelim_20261106"
```

기본 출력은 `~/ae-normal-shift/<UTC timestamp>/`입니다. `--output`으로 새 외부 폴더를
지정할 수 있습니다. 기존 폴더·Git 내부·입력 폴더와 겹치는 출력은 거부합니다.
학습·평가의 기존 CLI stage를 재실행하지 않으므로 기존 threshold/결과를 갱신하지 않습니다.

## 필요한 실제 파일

각 RPM의 원본 실험 폴더:

- split CSV와 split_config JSON
- train/validation/test_normal/test_anomaly_normalized.npz
- train/validation/test_normal_spectra.npz
- autoencoder_{RPM}rpm.pt
- evaluation_file_{RPM}rpm.csv

네 분할 normalized NPZ는 기존 load_experiment의 전체 데이터 지문 검사에 필요합니다.
**Fault 자료는 지문 검사 외에 점수 분포·스펙트럼 진단이나 threshold 계산에 사용하지 않습니다.**
일반 MAT 원본은 이번 분석에 필요하지 않습니다. 번들은 기존 로더가 해시를 검사하는 완전한
파일 구성이 필요합니다. 버전 고정 웹 환경을 사용하세요.

## 검사와 재사용

- load_experiment로 분할·구간·메타데이터·데이터 지문 검증
- 기존 SavedModelInference의 AE와 원본 체크포인트 state_dict 일치 검사
- pre-Z Spectrum NPZ의 주파수·파일·구간·날짜·상태를 normalized NPZ와 대조
- 기존 apply_normalization 결과가 저장 normalized X와 원소 단위로 일치하는지 확인
- 기존 evaluate와 같은 float32 모델 forward, double 차이·제곱·평균으로 구간 MSE 계산
- 기존 score_rows/file_scores로 8구간 평균 파일 점수 생성
- 기존 verify_saved로 원래 Test Normal 파일·점수와 비교
- 원래 평가 CSV의 threshold가 번들 값과 동일한지, 파일별 판정/오탐률이 재현되는지 확인

저장 임계값을 **그대로 판정 기준**으로 사용합니다. 재계산 Validation Q95는 일치 확인용입니다.
부동소수점/실행 환경의 작은 차이를 rtol=1e-7, atol=1e-10으로 검사하고 차이를 기록합니다.
사용자가 보고한 재계산 Q95 소수값을 새 임계값으로 덮어쓰지 않습니다.
점수는 허용 오차 안이어도 판정이 바뀌면 별도 목록과 검토 필요 상태를 남깁니다.

## 출력 해석

각 RPM에 대해 다음을 만듭니다.

| 파일 | 내용 |
| --- | --- |
| {RPM}_scores.png | Train 9/12·Validation 9/16·Test Normal 9/19 파일 MSE 분포와 고정 Q95 한 그래프 |
| {RPM}_spectra.png | Normal1/Normal2 각각의 날짜별 pre-Z·Z-score·복원 제곱오차 중앙값과 10–90% 범위 |
| {RPM}_normalization.png | 고정 Train std, 날짜 간 pre-Z 중앙값 차이를 그 std로 나눈 수치 |
| {RPM}_file_scores.csv | 새로 계산한 정상 파일 점수·날짜·상태·고정 기준·판정 |
| {RPM}_distributions.csv | 날짜/Normal1/Normal2/전체별 파일 수·중앙값·분위수·기준 초과 비율 |
| {RPM}_band_errors.csv | 파일별 0–100/100–1000/1000–10000Hz의 입력·정규화·복원 오차 |
| {RPM}_band_shifts.csv | 위 세 대역 및 고정 500Hz 대역별 Train 대비 입력 분포 차이 |
| {RPM}_profiles.npz | 그래프에 쓰인 정상 파일별 Spectrum/오차 profile과 고정 Train std |
| report.json | 기존 FPR 비교, 수치 차이, 실제 관측값, 미확인 가설, 원본 전후 해시 |

스펙트럼 profile은 **구간별 log1p/Z-score 값을 8구간 평균한 것**입니다. 원래 선형 진폭을
평균한 뒤 로그를 취한 값이나 PSD/물리 단위 진동 속도가 아닙니다.
스펙트럼 그림의 범위는 파일 사이의 분포이며 추정 신뢰구간이 아닙니다.
Normal1·Normal2를 따로 비교하여 상태 구성 비율의 영향을 숨기지 않습니다.

band_mse는 해당 대역 내 평균 오차이며, score_contribution은 해당 대역 오차 합/3201입니다.
서로 겹치지 않는 세 대역의 contribution 합이 파일 MSE와 일치하는지 검사합니다.
z_rms는 정규화된 스펙트럼의 수치 RMS이며 원본 진동의 물리 RMS가 아닙니다.
band_shifts의 절대 차이/Train std는 수학적 확대량으로 **AE 오차 자체가 아닙니다.**

## 관측과 가설의 구분

report.json의 observations는 실제 입력 파일에서 계산한 점수·비율·차이입니다.
hypotheses_not_confirmed에는 날짜/운전 조건의 Domain Shift 가능성, 작은 Train std의
변화 확대 가능성, Normal1/Normal2 혼합 영향이 가설로만 기록됩니다.
센서·장착·단위·운전 상태 중 어떤 것이 원인인지는 이 NPZ 분석만으로 확정하지 않습니다.

Train·Validation의 기준 초과 비율은 학습/검증 자료의 기술 통계이며 독립 시험 FPR로
대체하지 않습니다. Test Normal FPR만 기존 시험 기록과 직접 대조합니다.
모든 날짜의 q95_descriptive_only는 분포 설명용이며 Test 임계값 후보가 아닙니다.
실제 Run 독립성은 미확인이며 기존 Test는 이미 분석한 개발 자료입니다.

종료 0은 진단 및 기존 판정 재현 성공, 종료 2는 판정 차이/원본 변경 검토 필요,
종료 1은 누락·불일치·비유한값 등의 검사 실패입니다. 실패도 report.json에 남기며
이전 출력을 덮어쓰거나 불일치를 숨기지 않습니다. 보고서·profile에도 실제 자료 정보가
있으므로 공개 Git에 추가하지 않습니다.

## 테스트

`python -m unittest tests.test_ae_normal_shift -v`

합성 NPZ·임시 AE 가중치로 FPR 재현·정상 분포 차이 검출·그래프 생성·고정 threshold 보존·
누락/변조/출력 보호를 확인합니다. 실제 WSL 자료의 오탐 원인을 확인한 결과가 아닙니다.
