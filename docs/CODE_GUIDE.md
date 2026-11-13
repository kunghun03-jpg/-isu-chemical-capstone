# 파이썬 코드 안내

처음에는 `run_project.py`의 `STAGES` 목록과 `main()`을 읽으면 전체 처리 순서를 알 수 있습니다.
실제 계산은 `vibration` 폴더의 각 모듈이 맡습니다.
AE 정규화만 바꾸는 개발 비교는 [정규화 비교 기록](AE_NORMALIZATION_COMPARISON.md)을 확인하세요.
`normalize`의 기본 `per_frequency`는 종전 방식이며, `--normalization global_train`은
RPM별 Train 전체 값의 공통 통계만 사용합니다. 모델 학습 설정은 바꾸지 않습니다.
실제 Run 본 실험은 확정 대기 중이고 별도 승인된 날짜 그룹 예비 실험은 완료했습니다.
결과는 [날짜 예비 실험 보고서](DATE_PRELIMINARY_RESULTS.md), 본 실험 선행 조건은
[개발 계획](DEVELOPMENT_PLAN.md)을 확인하세요.
OCSVM·IF·공정 비교 구조는 [비교 모델 안내](COMPARISON_MODELS.md)에 설명합니다.
`--stage all`은 아래 1~7단계만 실행합니다. 8~11단계는 선택적 추가 분석입니다.

## 처리 순서

| 순서 | 파일 | 하는 일 |
| --- | --- | --- |
| 1 | `vibration/check.py` | 원본 파일의 형식과 손상 의심 값을 검사 |
| 2 | `vibration/split.py` | 정상 파일을 학습·검증·평가로 분할, 이상 파일은 평가에 배정 |
| 3 | `vibration/spectra.py` | 파일을 8개 구간으로 나누고 주파수 스펙트럼 생성 |
| 4 | `vibration/normalize.py` | 정상 학습 데이터의 기준으로 모든 분할을 정규화 |
| 5 | `vibration/train.py` | 정상 데이터를 학습하고 최적 모델 저장 |
| 6 | `vibration/evaluate.py` | 구간 단위 이상 점수와 판정 계산 |
| 7 | `vibration/file_evaluate.py` | 구간 점수를 평균하여 파일 단위로 평가 |
| 8 | `vibration/baseline.py` | 정상 평균 스펙트럼과의 거리로 비교 성능 계산 |
| 9 | `vibration/features.py` | 회전 주파수 주변의 채널별 최대 진폭 추출 |
| 10 | `vibration/auxiliary.py` | 회전 주파수 특징으로 보조 이상 점수 계산 |
| 11 | `vibration/combined.py` | 오토엔코더와 보조 판정을 OR로 결합하여 평가 |

추가 선택 단계는 `ocsvm`, `iforest`, `compare`입니다.
`one_class.py`는 비교 모델 공통 특징·학습·Q95·저장을,
`experiment.py`는 세 모델의 Run·NPZ 일치 검사를,
`comparison_metrics.py`는 구간·파일 지표와 동일 시험 집합 비교를 담당합니다.
AE의 모델 수식은 유지하고 train·evaluate·file_evaluate에 공통 검사를 추가했습니다.
실행 전 네 분할의 NPZ를 모두 검사하며 학습에는 정상 train 배열만 전달합니다.

`vibration/model.py`는 오토엔코더 구조를 정의합니다. 별도 처리 단계가 아니라 학습과 평가에서 함께 사용합니다.

## 주요 용어

- **train**: 모델과 정규화 기준을 만드는 정상 학습 데이터
- **validation**: 모델 선택과 임계값 설정에 사용하는 정상 검증 데이터
- **test_normal**: 정상 파일이 이상으로 잘못 판정되는 비율(오탐률)을 확인하는 데이터
- **test_anomaly**: 이상 파일이 이상으로 판정되는 비율(탐지율)을 확인하는 데이터
- **재구성 오차**: 모델이 복원한 스펙트럼과 입력 스펙트럼의 차이
- **임계값**: 점수가 이 값보다 크면 이상으로 판정하는 기준

## 결과와 추가 분석

결과 위치는 `outputs/{RPM}rpm/{실험이름}/`입니다.
기본 실험 이름은 `ch3_runs`이며 새 실험은 `--run-name`으로 구분합니다.

```powershell
python run_project.py --rpm 1800 --stage inventory --run-name run_setup
```

`scripts/analyze_normal_1800rpm.py`는 `1800rpm/ch3_runs`의 정상 점수를 분석합니다.
저장된 분석 CSV로 그래프만 다시 만들 때는 다음 명령을 사용합니다.

```powershell
python -m scripts.analyze_normal_1800rpm --plot-only --run-name ch3_runs
```

현재 공통 실행은 `run_project.py`를 사용하며 테스트는 `tests/`, 추가 분석은 `scripts/`에 있습니다.


## CH3 단일 채널 전환 (1단계)

- 원본 MAT: `(6, 65536)`를 유지하며 CH3(Python index 2)만 선택합니다.
- 파일당 FFT 결과: `(8, 1, 3201)`, 모델 입력·출력: `(batch, 1, 3201)`.
- FFT, log1p, 학습 기준 Z-score, MSE 계산식은 유지합니다.
- 회전 주파수 보조 분석도 CH3만 사용합니다.
- 기존 6채널 NPZ, 모델, 임계값, CSV는 CH3 결과가 아닙니다. 재사용하지 않습니다.
- Run 분할 점검 전에는 전체 재학습·평가를 실행하지 않습니다.
- 형태 검증: `python -m tests.smoke_test_ch3` (임시 폴더만 사용, 학습 없음).

## Run 단위 분할 준비 (2단계)

기본 분할 방식은 `--split-mode run`입니다. Run ID를 모르면 자동으로 생성하거나
Normal1/Normal2를 임의로 쪼개지 않습니다. 기존 파일 시간순 60/20/20 방식은
`--split-mode file`로 명시해야 하며 Run 독립성을 보장하지 않습니다.

별도 날짜 예비 모드는 `date_preliminary`이며 run_id는 빈칸,
date_group은 `date:YYYYMMDD`로 저장합니다. 이를 실제 Run으로 표시하지 않습니다.
9/12 정상 Train, 9/16 정상 Validation, 9/19 정상/결함 Test 배정을 고정 검사합니다.
`python -m scripts.run_date_preliminary --run-name 새이름`은 모든 RPM 사전 검사 후
전처리·정규화·AE·OCSVM·IF·동일 시험 목록 비교를 실행합니다. DAE는 실행하지 않습니다.

### 1. 원본 파일 목록 만들기

```powershell
python run_project.py --rpm 1800 --stage inventory --run-name run_setup
```

`outputs/1800rpm/run_setup/run_inventory_1800rpm.csv`를 만듭니다.
900·1200RPM도 `--rpm`만 바꾸면 됩니다. 목록 생성은 신호를 변환하거나 모델을 학습하지 않습니다.
현재 데이터의 통합 초안은 `manifests/run_inventory_unconfirmed.csv`에 준비되어 있습니다.
4,871개 원본 파일의 날짜·RPM·상태를 기록했으며 Run ID와 split은 모두 빈칸입니다.

| 열 | 내용 |
| --- | --- |
| `file` | 프로젝트 기준 원본 MAT 상대 경로 |
| `date` | 측정일, 예: `20260916` |
| `rpm` | 공칭 RPM |
| `condition` | 정상 상태 `Normal1`/`Normal2`/`Normal3` 또는 결함 상태. Normal3는 기본 제외 |
| `source_condition` | 원본 폴더의 상태 표기. Looeness 표기도 보존 |
| `is_normal` | Normal1·Normal2·Normal3이면 1 |
| `default_included` | Normal3는 0, 다른 상태는 1. 품질 통과 여부와는 별개 |
| `run_id` | 실험 기록으로 확인한 Run의 전역 식별자. 처음에는 빈칸 |
| `split` | `train`, `validation`, `test_normal`, `test_anomaly`. 처음에는 빈칸 |

### 2. Run 확인 후 목록 작성

원본 목록을 프로젝트 안의 `manifests/confirmed_1800rpm.csv` 등으로 복사하여 보관하고,
실험 기록을 확인한 뒤 **기본 포함 행**의 Run ID와 배정을 입력합니다. 결함 행도 포함합니다.
Normal3는 정상 상태이지만 기본 실험에서 제외하므로 split을 비워 둡니다.
Normal3에 train·validation·test_normal·test_anomaly를 배정하면 기본 분할 단계에서 오류가 납니다.
품질검사에서 제외될 파일도 목록에는 유지합니다. 실제 분할 결과에서만 제외합니다.

- 같은 실제 Run은 날짜·RPM이 달라져도 같은 ID를 사용합니다.
- 서로 다른 날짜의 ID를 같게 써서 충돌하거나, 같은 Run을 다른 ID로 쪼개지 않도록 합니다.
- 한 Run은 한 split에만 배정합니다. 정상은 학습·검증·정상 시험, 결함은 이상 시험만 허용합니다.
- 여러 날짜·RPM을 하나의 CSV로 관리할 수 있습니다. 코드가 전체 CSV의 Run 충돌을 검사한 뒤
  요청한 RPM·날짜만 선택합니다. 서로 다른 CSV 간 충돌까지 자동 비교하지는 않습니다.
- 빈 ID, 미확정 배정, 중복 파일, 누락 파일, 경로와 날짜·RPM·상태 불일치는 오류입니다.
- 학습·검증·정상 시험 중 하나가 없으면 저장하지 않습니다. 정상 Run이 두 개뿐이면
  세 분할을 억지로 만들지 말고 추가 데이터 또는 평가 설계 확정을 기다립니다.
- 이 검사는 **입력된 ID의 중복 배정**을 검증합니다. 실제 운전의 독립성은 실험 기록으로 확인해야 합니다.

### 3. 품질검사와 Run 분할

Run 정보가 확정된 뒤 실행하는 예시입니다. 현재 단계에서 전체 학습·평가는 실행하지 않습니다.

```powershell
python run_project.py --rpm 1800 --stage check --run-name ch3_runs
python run_project.py --rpm 1800 --stage split --run-name ch3_runs --split-mode run --run-manifest manifests/confirmed_1800rpm.csv
```

품질검사는 문제 후보 CSV와 `quality_checked_1800rpm.csv`(검사한 원본 목록)를 저장합니다.
분할은 `split_1800rpm.csv`와 `split_config_1800rpm.json`을 저장합니다.
검사 대상에 없던 파일이 추가되면 check를 다시 실행해야 합니다.
이후 같은 `--run-name`으로 spectra → normalize를 실행하면 날짜·RPM·Run ID가 각 구간에 보존됩니다.
전처리에서도 Run의 분할 간 중복을 다시 검사합니다.

### 추가 날짜 데이터가 들어오면

현재 형식은 `data/20260912/20260912_1800RPM/Normal1/Vib/*.mat`입니다.
기존 `data/20260912_1800RPM/Normal1/Vib/*.mat` 형식도 지원합니다.
`20260919` 및 다른 RPM도 같은 규칙입니다. 날짜를 코드에 추가할 필요는 없습니다.
`--dates`를 생략하면 해당 RPM의 모든 날짜를 탐색합니다.

특정 날짜만 대상으로 할 경우 inventory·check·split에 같은 날짜를 지정합니다.

```powershell
python run_project.py --rpm 1800 --stage inventory --run-name selected_setup --dates 20260912 20260916 20260919
```

목록을 새로 만들고 기존의 확인된 Run 배정을 병합한 뒤, 추가 파일의 실제 Run을 확인합니다.
새 실험 이름으로 check와 split을 진행하는 것을 권장합니다. 기존 이름으로 다시 실행하려면
`--overwrite`가 필요하며, 기존 전처리 결과도 나중에 새 분할에 맞춰 다시 만들어야 합니다.
ZIP은 자동 해제하지 않으며, 원본 MAT 파일의 수정·삭제는 하지 않습니다.
상위 날짜와 RPM 폴더의 날짜가 다르면 오류가 납니다. 두 구조에 같은 원본을 중복 복사하지 마세요.
상태명 Looeness0.05·Looeness0.1·LooenessFull은 내부적으로 Looseness 계열로 통일합니다.

### 전체 품질검사와 통합 목록 갱신

`python -m scripts.prepare_data`는 전체 MAT 탐색 수를 실제 파일 수와 대조한 뒤,
모든 파일을 검사하고 통합 미확정 목록을 갱신합니다. 모델 학습·평가와는 별개입니다.
`outputs/data_preparation/quality_all_files.csv`는 전체 파일,
`quality_issues.csv`는 문제 후보만 기록합니다. RPM별 상세·요약·제외 후보도 함께 저장합니다.
품질 정책은 기존처럼 원본 6채널 중 하나라도 비정상 값이 있으면 보수적으로 제외 후보로
기록하며, CH3의 비정상 값 수를 별도 열로 제공합니다. 모델 입력은 계속 CH3 한 채널입니다.
SHA256과 검사 전후 크기·수정시간 비교도 기록합니다. SHA256은 읽은 바이트의 식별값이며
원본 데이터의 물리적 측정 정확성이나 Run 독립성을 증명하지는 않습니다.

### 저장 모델 단일 MAT 추론

2026-11-10 추가한 `vibration/detail_ui.py`는 기존 FFT log1p 결과·모델별 8구간 점수와 저장 구간 임계값을 표시합니다.
추론 API와 파일 평균 판정은 변경하지 않습니다. [상세 화면 안내](DETAIL_ANALYSIS_SCREEN.md)를 참고하세요.

`streamlit_app.py`는 업로드·미리보기·분석·결과 표시의 최소 웹 화면이며,
`vibration/web_inference.py`는 메모리 업로드 검증과 오류 안내를 담당합니다.
`vibration/history.py`와 `vibration/history_ui.py`는 SQLite 분석 이력 저장·중복 방지·검색·상세·CSV를 담당합니다. [이력 안내](SQLITE_HISTORY.md)를 참고하세요.
실행은 `python -m streamlit run streamlit_app.py`이며 [화면 안내](STREAMLIT_APP.md)에 자세히 기록했습니다.

`vibration/preprocessing.py`는 기존 배치 FFT와 단일 추론의 CH3 전처리를 공유합니다.
`vibration/inference_bundle.py`는 기존 주파수별 정규화 실험을 복사·검증하며,
`vibration/inference.py`는 패키지와 MAT만 읽어 AE·OCSVM·IF 점수·판정을 반환합니다.
`scripts/infer_mat.py`가 CLI이고 `scripts/verify_single_file_inference.py`가 기존 배치 결과 대조 도구입니다.
공통 정규화 비교안은 채택하지 않았으며 원래 두 실험은 보존합니다.
자세한 호출 방법과 한계는 [추론 안내](SINGLE_FILE_INFERENCE.md)를 확인하세요.

### 회귀 검사 실행

```powershell
python -m unittest discover -s tests -v
python -m tests.smoke_test_ch3
```

검사는 임시 폴더에서 다중 날짜 처리, Run 충돌·미확정·누락 거부, 파일 모드,
메타데이터 전달과 CH3 입출력을 확인합니다. 테스트의 가상 Run ID는 실제 데이터에 적용하지 않습니다.
