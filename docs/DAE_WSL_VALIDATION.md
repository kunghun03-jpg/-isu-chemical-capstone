# WSL에서 DAE 실제 파일 검증 준비

클라우드에서는 WSL의 `~/capstone` 파일에 접근하지 못했습니다. 실제 모델 변환·추론·성능을
검증한 기록이 아닙니다. 아래 도구를 WSL에서 실행한 뒤 생성된 보고서를 확인해야 합니다.
검증에 필요한 코드만 WSL 체크아웃에 전달하세요. 모델·NPZ·MAT는 Git에 추가하지 않습니다.

## 실행 (프로젝트 루트에서)

이 도구와 vibration/dae_*.py가 포함된 체크아웃 및 원본 dae_legacy/ 코드가 필요합니다.
WSL의 기존 DAE 실행 Python 환경을 사용하세요. 패키지를 임의로 업그레이드하지 않습니다.
소스 경로 기본값은 `~/capstone`, legacy 코드 기본값은 체크아웃의 `dae_legacy/`입니다.
사용자가 신뢰하는 legacy 코드만 지정하세요. 비교 과정은 그 코드를 별도 Python 프로세스로 실행합니다.

```bash
cd /path/to/-isu-chemical-capstone
OMP_NUM_THREADS=2 python -B -m scripts.validate_dae_wsl --source "$HOME/capstone" --mat "/path/to/actual_measurement.mat"
```

기존 실행기가 `python3` 또는 가상환경의 절대 경로라면 `python`만 교체합니다.
실제 MAT 없이 구조·통계·합성 입력의 계산 호환성만 먼저 확인하려면 `--mat`을 생략합니다.
legacy 코드가 다른 곳에 있으면 `--legacy-code /path/to/legacy_code`를 추가합니다.

출력 기본값: `~/dae-validation/<UTC timestamp>/`.
`--output /outside/checkout/new_directory`로 변경할 수 있습니다. 기존 디렉터리와 원본 파일은
덮어쓰지 않으며, 출력 경로가 체크아웃 내부이거나 원본 폴더와 중첩되면 거부합니다.
모든 산출물은 로컬에만 저장되며 업로드·Git 명령은 실행하지 않습니다.

## 실제 파일 검사 내용

- autoencoder_model.pth: raw state_dict를 동일 구조에 strict load, 유한값 검사
- normal_dataset_normalized.npz: 저장 mean/std의 dtype·shape·유한값·양의 std·800특성 broadcast 검사
- normal_threshold.npz: 저장 threshold가 유한한 음이 아닌 단일 값인지 검사; Q95 재계산 없음
- band_statistics.npz: 있으면 low/mid/high 순서·3개 평균/표준편차·유한값·양의 std 검사
  (선택 파일, 핵심 추론 번들에 포함하지 않음)
- 실제 legacy model.py, web_preprocessing.py, dae_inference.py를 별도 프로세스에서 실행
- 같은 MAT에서 legacy/new 전처리·정규화·복원 배열·float32 MSE·threshold·판정의 정확한 일치 검사
- 기존 convert_legacy_bundle을 호출해 학습 배열을 제외한 추론용 번들 생성·검증
- 번들 수치 커널과 legacy 결과도 비교. 이 진단에서만 base 계산 함수를 직접 호출하며
  번들 manifest의 RPM을 바꾸거나 공개 추론의 RPM 차단을 해제하지 않음
- 검사 전후 원본 artifact SHA256 비교; legacy 비교 코드와 MAT의 해시도 보고서에 기록

검사는 임계값·모델을 새로 학습하지 않습니다. 일치 실패 시 허용 오차를 자동으로 늘리거나
판정을 강제로 맞추지 않습니다. stderr와 report.json의 실패 항목을 확인하세요.

## RPM 근거 조사와 현재 결론

GitHub의 기존 train.py는 정규화 NPZ의 X_train/X_val/X_test를 읽지만, 정상 데이터 생성·분할
코드는 제공되지 않았습니다. prepare_fault_dataset.py와 evaluate_fault.py의 900/1200/1800
목록은 결함 자료 수집·평가 대상이며 정상 학습 범위를 증명하지 않습니다.
추론 코드는 단일 가중치·통계·임계값 파일을 사용하며 RPM을 입력받지 않습니다.

도구는 실제 정상 NPZ의 배열 이름·shape·dtype를 header로 조사합니다. 큰 학습 배열이나
pickle/object 배열을 로드하지 않습니다. 작은 group/file/date/run/rpm 메타데이터가 있으면
RPM 단서를 기록하고, legacy 코드의 관련 줄도 기록합니다. 이것은 근거 수집이며
그룹별 독립성·정규화 누수 여부·RPM별 탐지 성능 검증은 아닙니다.

따라서 생성 번들은 항상 `supported_rpms: null`을 유지합니다. 지원 RPM 설정 옵션도 없습니다.
정상 학습/검증/시험의 원본 파일 목록·날짜·Run 분할·정규화 계산 기록과 RPM별 실측 검증을
확인한 뒤 별도 절차로 지원 범위를 결정해야 합니다.

## 예상 결과와 해석

성공한 경우에만 다음과 비슷한 출력이 표시됩니다.

```text
PASS: compatibility_checks_passed_rpm_unconfirmed; input=provided_mat
보고서·번들: /home/<user>/dae-validation/<timestamp>
실제 탐지 성능 검증 아님. 지원 RPM 미확인; 공개 추론 차단 유지.
```

`report.json`의 주요 항목:

- checks: 구조·통계·전처리·계산 일치·RPM 미확인 차단 검사 결과
- statistics: mean/std 크기·dtype, 실제 저장 threshold 값
- rpm_investigation: 실제 NPZ header·메타데이터 단서·관련 코드 줄
- input_kind: synthetic_mat 또는 provided_mat (사용자가 제공한 MAT임을 뜻함)
- source_unchanged: true여야 함
- supported_rpms: null, performance_validated: false

파일/코드 누락·가중치 불일치·잘못된 통계·계산 불일치 등은 종료코드 1입니다.
검사 도중 실패하면 생성된 report.json에 실패 이유를 남깁니다. 출력 생성 전의 경로·파일
검사에서 실패하면 콘솔 오류만 남습니다. 실패한 출력은 덮어쓰지 말고 새 경로로 재검사합니다.

실제 파일을 사용해 모든 호환성 검사가 통과해도 **탐지 성능이나 지원 RPM이 검증됐다는 뜻은
아닙니다.** 학습 당시 라이브러리 버전도 별도로 확인해야 하며 번들에 기록되는 버전은 변환 환경입니다.
보고서·reference NPZ·번들에도 모델/로컬 정보가 포함되므로 GitHub에 추가하지 않습니다.
