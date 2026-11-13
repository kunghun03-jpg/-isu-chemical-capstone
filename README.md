# CH3 정상 진동 기반 이상감지

개발가이드를 기준으로 정상 데이터만 학습하는 1D Convolutional Autoencoder 프로젝트입니다.
CH3 전처리·AE·OCSVM·IF와 Run 단위 분할 기능을 구현하고 전체 원본 품질검사를 완료했습니다.
현재는 책임님께 진행 상황과 향후 계획을 보고하고 연구 방향 피드백을 기다리는 단계입니다.
실제 Run ID와 독립성은 아직 미확정이므로 Run 단위 본 실험은 대기 중입니다.
2026-11-06 사용자 지정 날짜 그룹 예비 실험으로 AE·OCSVM·IF를 RPM별 학습·평가했습니다.
[예비 실험 결과](docs/DATE_PRELIMINARY_RESULTS.md)에 분할·지표·한계를 정리했습니다.
9월 12일·16일·19일 데이터의 날짜 상위 폴더 구조와 기존 평면 구조를 모두 지원합니다.

## 폴더 구조

```text
isu_chemical/
├─ run_project.py      # 공통 실행 진입점
├─ vibration/          # 검사·목록·분할·전처리·모델·평가
├─ data/               # 원본 MAT와 ZIP, 변경하지 않음
├─ manifests/          # 파일 목록과 확정 Run 배정
├─ outputs/            # 현재 CH3 실험 결과: RPM/실험이름
├─ tests/              # 가상 데이터 기본 검사, 실제 데이터 모델 학습 없음
├─ scripts/            # 선택적인 추가 분석
└─ docs/               # 코드 안내와 개발가이드 대응 현황
```

## 지금 할 수 있는 작업

프로젝트 루트에서 실행합니다. Python에는 NumPy·SciPy·PyTorch가 필요합니다.
그래프 생성 시 Matplotlib도 필요합니다.
비교 모델은 scikit-learn·joblib을 사용합니다. 패키지 목록은 `requirements.txt`에 있습니다.

```powershell
python run_project.py --rpm 1800 --stage inventory --run-name run_setup
python -m unittest discover -s tests -v
python -m tests.smoke_test_ch3
```

기존 통합 목록은 `manifests/run_inventory_unconfirmed.csv`입니다.
Run ID와 split은 비어 있으며 실험 기록 확인 전에는 임의로 채우지 않습니다.
Normal3도 정상 데이터로 집계하지만 기본 학습·검증·시험에서는 제외합니다.
9월 19일 원본 폴더의 Looeness 표기는 목록의 condition에서 Looseness로 통일하고,
source_condition과 file에는 원본 표기를 보존합니다.

## 전체 원본 품질검사

```powershell
python -m scripts.prepare_data
```

모델을 실행하지 않고 모든 MAT의 읽기·형태·실수 배열·샘플링 주파수·NaN·Inf·극단값을 검사합니다.
보고서는 `outputs/data_preparation/`, 갱신된 미확정 목록은 `manifests/`에 저장합니다.
상세 결과는 [전체 데이터 검사 보고서](docs/DATA_QUALITY_REPORT.md)에 정리합니다.
원본 MAT와 폴더명은 바꾸지 않습니다. 품질 통과와 Run 독립성 확정은 서로 다른 절차입니다.

기본 결과 위치는 `outputs/{RPM}rpm/ch3_runs/`입니다.
기본 분할 모드는 `run`으로, 확정 Run CSV가 없으면 분할하지 않습니다.
기존 파일 비율 분할은 비교용 옵션 `--split-mode file`로만 사용합니다.

## 다음 단계

1. 책임님 피드백 확인 및 기본 실험 방향 확정
2. 확보된 3개 날짜 데이터의 실제 Run 확인과 분할 기준 결정
3. 품질 결과를 반영한 공통 분할·CH3 전처리 후 세 모델 실제 학습·평가
4. 동일 시험 목록으로 지표·그림·한계 분석 및 최종 발표자료 정리

단계별 입력·실행 조건·산출물·완료 기준은 [개발 계획](docs/DEVELOPMENT_PLAN.md)에 정리했습니다.

Run 분할 기능만으로 실제 Run의 독립성이 증명되지는 않습니다.
FFT·log1p·학습 기준 Z-score·MSE·검증 Q95를 기본 방법으로 유지합니다.
회전 주파수 보조 점수와 OR 결합은 선택적 확장 기능입니다.
`--stage all`은 기본 오토엔코더 흐름(check부터 file_evaluate까지)만 실행합니다.
평균 스펙트럼 비교·회전 주파수 분석·OR 결합은 개별 `--stage`로 선택합니다.
Run 확정 전에는 `all`을 실행하지 않습니다.

## 비교 모델

OCSVM·IF는 AE와 같은 CH3 정규화 스펙트럼을 2차원 배열로 바꾸어 사용합니다.
정상 학습 구간만 fit하고, 정상 검증 Q95로 구간·파일별 임계값을 정합니다.
아래는 **Run 확정 및 전처리 이후 본 실험에 사용할 명령**입니다.
날짜 예비 실험에서는 같은 모델 단계를 별도 date_preliminary 설정으로 실행했습니다.

```powershell
python run_project.py --rpm 1800 --stage ocsvm --run-name ch3_runs
python run_project.py --rpm 1800 --stage iforest --run-name ch3_runs
python run_project.py --rpm 1800 --stage compare --run-name ch3_runs
```

`compare`는 같은 실험에서 AE의 evaluate·file_evaluate와 두 비교 모델의 결과가 모두 필요합니다.
분할·전처리 데이터 식별값과 시험 파일·구간이 다르면 중단합니다.
기본 all에는 비교 모델을 추가하지 않아 AE 실행 순서는 유지됩니다.
일반 미확정 Run·file 모드는 차단합니다. 별도 `date_preliminary` 모드는 실제 Run ID를 비워 두고
승인된 날짜 배정(9/12 정상 학습, 9/16 정상 검증, 9/19 정상·결함 시험)을 검사하여 실행합니다.
DAE는 팀원 담당으로 현재 실행 대상에서 제외합니다.

```powershell
python -m scripts.run_date_preliminary --run-name date_prelim_new
```

기존 폴더가 있으면 덮어쓰지 않습니다. 전체 분할·품질 SHA256·중복 검사를 통과해야만 전처리·학습을 시작합니다.

가상 데이터 검사: `python -m unittest tests.test_one_class -v`.
상세 설정과 결과 구조는 [비교 모델 안내](docs/COMPARISON_MODELS.md)를 확인하세요.

## AE 정규화 개발용 비교

채택 기준은 기존 주파수별 Train 정규화(`per_frequency`)입니다. 공통 정규화 비교안은 채택하지 않으며 두 실험 결과는 모두 보존합니다.
별도 개발 비교는 `global_train`으로 해당 RPM의 CH3 Train log1p 스펙트럼 전체에서
단일 평균·표준편차를 구합니다. 날짜 분할·품질 제외·FFT/log1p·AE 구조·학습 설정은 유지합니다.
시험 데이터를 이미 살펴본 뒤 설계한 비교이므로 독립적인 최종 성능 평가로 보고하지 않습니다.
상세 지표와 상태별 비교는 [AE 정규화 비교](docs/AE_NORMALIZATION_COMPARISON.md)에 기록합니다.
이 개발 비교에서는 AE만 실행하며 기존 OCSVM·IF·DAE 결과를 변경하지 않습니다.

## 저장 모델 단일 MAT 추론

Streamlit 최소 화면: `python -m streamlit run streamlit_app.py`.
업로드 → CH3 미리보기 → 분석 → 결과 표시를 지원합니다. [웹 화면 안내](docs/STREAMLIT_APP.md)를 확인하세요.
분석 후 결과 저장과 날짜·RPM·모델·판정별 검색, 상세·CSV 다운로드를 지원합니다. [SQLite 이력 안내](docs/SQLITE_HISTORY.md)를 확인하세요.
결과 아래 CH3 log1p 스펙트럼·모델별 8구간 점수 상세를 표시합니다. [표시 값·검증 방법](docs/DETAIL_ANALYSIS_SCREEN.md)을 참고하세요.

AE·OCSVM·IF의 기존 모델·주파수별 정규화·파일/구간 임계값을 불러오는 공통 API와 CLI를 제공합니다. 추론 중 재학습·통계 계산·Q95 재설정은 없습니다.

```powershell
python -m scripts.infer_mat --file "path/to/uploaded.mat" --rpm 900 --model all
```

[사용 방법과 패키지 구조](docs/SINGLE_FILE_INFERENCE.md), [배치 일치 검증](docs/SINGLE_FILE_INFERENCE_VALIDATION.md), [성능 한계](docs/INFERENCE_LIMITATIONS.md)를 확인하세요.

자세한 명령은 [코드 안내](docs/CODE_GUIDE.md), 완료·미완료 항목은
[개발가이드 대응 현황](docs/GUIDE_STATUS.md)을 확인하세요.
