# WSL DAE 실제 성능 예비 평가 도구

이 도구는 기존 가중치·mean/std·threshold를 고정하고 실제 MAT를 오프라인 평가합니다.
웹앱·이력 DB·기존 세 모델·DAE 번들·RPM 지원 설정은 읽거나 변경하지 않습니다.
legacy 진단 adapter를 사용하며, 평가 대상 RPM 목록은 지원 RPM 선언이 아닙니다.

## 실행

도구가 포함된 프로젝트 루트에서 **기존 WSL DAE Python 환경**으로 실행합니다.
코드만 WSL에 전달하고 실제 MAT/NPZ/가중치는 Git에 추가하지 않습니다.

```bash
cd /path/to/-isu-chemical-capstone
OMP_NUM_THREADS=2 python -B -m scripts.evaluate_dae_wsl --source "$HOME/capstone" --data-root "/mnt/c/Users/kungh/OneDrive/바탕 화면/캡스톤 디자인"
```

`--data-root`는 실제 원본 MAT의 공통 상위 폴더로 바꿉니다. 날짜 상위 폴더가 있는 구조와
평면 구조를 모두 탐색합니다. 예: `20260919/20260919_1200RPM/IR/Vib/*.mat` 또는
`20260919_1200RPM/IR/Vib/*.mat`. 파일명에서 결함이나 RPM을 추측하지 않습니다.

- `--dates 20260919`: 해당 날짜만 평가 (지정하지 않으면 모든 날짜)
- `--normal-conditions Normal1 Normal2`: 정상 평가 범위 변경
- `--output "$HOME/dae-performance/new-experiment"`: 새로운 외부 출력 폴더 지정

기본 정상 목록은 Normal1/Normal2/Normal3이며 각각의 오탐률도 출력합니다.
기본 결함 목록은 IR/OR/Roller/Looseness0.05/Looseness0.1/LoosenessFull/
Unbalance_-1/Unbalance_-2입니다. Looeness 원본 폴더 표기는 보존하고 집계명만 Looseness로
통일합니다. 다른 상태는 `--fault-conditions ...`로 명시해야 하며, 알 수 없는 상태를
자동으로 결함으로 분류하지 않습니다. 명시적으로 선택하지 않은 파일도 제외 목록에 남습니다.

## 고정 계산과 출력

- CH3 전체 신호 → Welch PSD → sqrt → 0≤f<10000Hz → 4-bin 평균 → 800특성
  → log10(x+1e-12) → 저장 mean/std → float32 DAE → 파일 전체 MSE
- 저장 threshold를 그대로 사용하며 `MSE > threshold`일 때 이상
- 정상 오탐률: 이상으로 판정된 정상 파일 수 / 평가 성공 정상 파일 수
- 결함 탐지율: 이상으로 판정된 결함 파일 수 / 평가 성공 결함 파일 수
- RPM별 합계 및 정상 상태/결함 종류별 수치를 모두 출력. 자료가 없으면 `평가 불가`/null

원본 모델·통계·정상/결함 NPZ·선택적 대역 통계의 SHA256을 전후 비교합니다.
재학습·임계값 재조정·정규화 재계산·모델 변환은 하지 않습니다.

기본 출력: `~/dae-performance/<UTC timestamp>/`

| 파일 | 내용 |
| --- | --- |
| report.json | 고정 threshold·실행 버전·원본 해시·집계·제외 사유·제한사항 |
| provenance.json | NPZ header/메타데이터·분할/RPM 단서·중복/누출 위험·미확인 사항 |
| metrics.csv | RPM/정상·결함/상태별 발견·평가·제외·이상 판정 수, 비율, 평가 날짜 |
| files.csv | 각 MAT의 원본 경로·날짜·RPM·상태·해시·MSE·판정 또는 제외 사유 |

산출물은 새 디렉터리에만 저장합니다. Git 체크아웃 내부나 원본/데이터 경로와 겹치는
출력은 거부합니다. 업로드·Git 명령은 없습니다. 보고서에도 로컬 파일명/메타데이터가
있으므로 공개 저장소에 추가하지 마세요.

## 학습 출처와 누출 조사

정상 NPZ의 X_train/X_val(또는 X_validation)/X_test header와 행별 메타데이터를 조사합니다.
예: train_files/files_train, train_file_names/file_names_train, train_groups/groups_train,
train_rpms/rpms_train, train_dates/dates_train, train_run_ids/run_ids_train 및 val/test 대응 키.
메타데이터는 해당 X 배열과 행 수가 맞는 문자열/정수 배열일 때만 단서로 사용합니다.
다른 키는 header에는 기록되지만 자동 연결하지 않습니다. 크기가 같다는 사실만으로
원본 행 대응이 검증된 것은 아닙니다. object 배열은 pickle 없이 건너뛰고 사유를 기록합니다.

fault_dataset.npz의 labels/groups/file_names/RPM/date/run 메타데이터도 조사합니다.
결함 자료의 RPM 목록을 정상 학습 RPM으로 간주하지 않습니다. fault NPZ는 메타데이터
조사용이며 실제 점수는 원본 MAT에서 다시 계산합니다.

- 정상 분할 간 파일 ID, float32 입력 특성 중복, 날짜 및 명시적 Run ID 겹침 기록
- 결함 NPZ 파일 목록과 정상 분할의 겹침 기록
- 평가 MAT가 Train/Validation 파일 목록 또는 입력 특성과 일치하면 제외
- 동일 CH3 복사본은 한 번만 평가. 상태/RPM이 충돌하는 동일 신호는 모두 제외
- 손상·형식 오류·지원하지 않는 경로/상태·평가 날짜 밖의 파일은 사유와 함께 제외
- 탐색 권한 오류/추적하지 않은 symlink 디렉터리는 탐색 불완전으로 처리하고 비정상 종료

일부 큰 정상 X 배열은 중복 fingerprint 계산을 위해 분할별로 하나씩 메모리에 로드합니다.
한 번에 약 `분할 행 수 × 800 × 저장 dtype 크기` 이상의 메모리가 필요합니다.

### 해석의 제한

항상 **독립성 미확인 예비 평가**로 표시하고 supported_rpms=null, web_activation_authorized=false를
기록합니다. 이 도구만으로 실제 Run 독립성을 인증하지 않습니다.

정규화된 X 배열만으로 mean/std의 Train 전용 계산을 증명할 수 없습니다. 파일명·Run ID가
없으면 확인 불가로 남기며, 중복을 못 찾았다고 누출이 없다고 결론 내리지 않습니다.
같은 Run의 서로 다른/인접 측정 신호는 해시가 달라도 상관될 수 있습니다.
같은 날짜가 겹친다는 사실도 누출 확정이 아닌 위험 단서입니다.

기본 실행은 발견된 모든 날짜를 평가합니다. Train/Validation과 확인된 일치만 제외하므로
남은 파일 전체가 독립 Test라고 주장할 수 없습니다. 수치는 **명시된 제외 정책 이후의 비율**입니다.
원본 분할 기록이 확보되면 별도 폴더 또는 --dates로 시험 범위를 제한할 수 있지만,
이 옵션 자체가 기존 모델의 학습 누출을 해결하지는 않습니다. 향후 독립 시험 설계·성능
기준 승인 전까지 웹 판정 차단을 유지합니다.

실행 종료코드 0은 일부 실제 MAT 평가와 보고서 작성 완료를 뜻합니다. 특정 RPM 또는 클래스
자료가 없는 경우 null을 확인해야 합니다. 평가 성공 0개·탐색 불완전·실행 실패·원본 변경은
종료코드 1이며, 출력 생성 후 실패하면 report.json에 오류를 남깁니다. 기존 출력은 덮어쓰지 않습니다.

## 클라우드 검증 범위

`python -m unittest tests.test_dae_evaluation -v`

합성 MAT·NPZ와 임시 가중치로 집계 공식, 원래 추론 점수와 일치, 제외·중복·누출 단서·
출력 보호·원본 불변을 검사합니다. 실제 WSL 파일에는 접근하지 못했으므로 실제 성능 수치를
보고한 것이 아닙니다. 실제 실행 후에는 metrics.csv와 provenance.json을 함께 검토해야 합니다.
