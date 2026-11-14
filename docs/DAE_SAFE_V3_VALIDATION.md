# 기존 DAE 번들의 안전한 V3 연결 및 MAT 검증

## 현재 확인된 계약

진입점은 `streamlit_app_v3.py`이다. DAE는 `vibration/dashboard_v3.py`에서 기존 `predict_dae` → `SavedDAEBundleInference` 경로로 실행한다. 환경변수 `ISU_CHEMICAL_DAE_BUNDLE`을 그대로 사용한다. 미설정이면 기존 `outputs/dae_bundle` 기본 경로이며 명시적 빈 문자열은 경로 누락으로 처리한다. V2의 기존 DAE 어댑터 및 AE/RF/DB 구현은 변경하지 않는다.

사용자가 제공한 실제 번들은 다음이다. 이 경로는 사용자 WSL에 있으며 Codex에서 접근했다고 주장하지 않는다.

```text
/home/kungh/dae-validation/20261113T041136515404Z/bundle
```

제공된 report.json의 compatibility_checks_passed_rpm_unconfirmed / provided_mat / performance_validated=false는 legacy 계산 호환성 근거다. threshold=0.054707884788513184, supported_rpms=null, training_provenance=unverified를 변경하지 않는다. 이 문서의 값은 사용자 제공 정보이며 새 실행으로 측정한 값이 아니다.

## 준비 상태

V3는 manifest·해시 검사 → 실제 모델/통계 로드 → RPM 검사 순서로 확인한다. 준비 상태를 미확인으로 표시하기 전에 손상된 모델을 탐지한다.

| code | 화면 | 의미 |
|---|---|---|
| path_missing | 모델 미준비 | 빈 경로/폴더 없음 |
| file_missing | 모델 미준비 | manifest/필수 파일 없음 |
| hash_error | 모델 검증 오류/실행 오류 | manifest 또는 파일 SHA256 불일치 |
| manifest_error | 모델 검증 오류/실행 오류 | 버전·스키마·런타임 등 규격 오류 |
| model_load_error | 모델 로드 오류/실행 오류 | 통계·threshold·가중치 구조/수치 로드 실패 |
| rpm_unconfirmed | 지원 범위 미확인 | 모델 로드 가능하지만 supported_rpms=null |
| rpm_unsupported | 선택 RPM 미지원 | 해당 RPM이 선언 지원 목록에 없음 |
| ready | 실제 점수·임계값·판정 | 기존 공개 로더 검증 통과 |

지원 범위 미확인에서는 V3 모델 표의 점수·임계값·판정은 대시/지원 범위 미확인이다. 임의 정상값이나 추정 점수를 채우지 않는다. AE 기준 판정, RF 정상 생략/보류, 기존 저장·이력은 그대로다. 점수 절댓값은 모델 간 공통 척도나 정확도가 아니다.

## WSL 연결

기존 서버를 종료하고 저장소에서 실행한다. 기존 미커밋 작업은 보존한다.

```bash
git fetch origin
git switch --track origin/feat/dae-safe-v3-validation
# 브랜치가 있으면 git switch feat/dae-safe-v3-validation
export ISU_CHEMICAL_DAE_BUNDLE="$HOME/dae-validation/20261113T041136515404Z/bundle"
export ISU_CHEMICAL_V2_MODEL=autoencoder
export ISU_CHEMICAL_CANDIDATE_LABELS=0
# 기존 AE/RF 번들 및 DB 설정 유지
"$HOME/capstone-web-venv/bin/python" -B -m streamlit run streamlit_app_v3.py --server.port 8501
```

이 번들은 정상 로드되어도 **지원 범위 미확인**이어야 한다. 이것은 연동 실패가 아니라 기존 안전 정책이다. 정상 로드 이전에 버전/해시 오류가 나오면 그 오류부터 조사한다. 지원 목록/manifest/패키지 검사를 임의 수정하지 않는다.

## 오프라인 실제 MAT 검사

먼저 차단 정책 확인만 수행할 수 있다. 이 모드에서는 미확인 RPM의 MAT를 읽거나 MSE를 계산하지 않는다.

```bash
read -r -p '실제 MAT 절대 경로: ' dae_mat
read -r -p '명목 RPM (900/1200/1800): ' dae_rpm
dae_audit="$HOME/dae-bundle-audit/$(date -u +%Y%m%dT%H%M%S%NZ)"
"$HOME/capstone-web-venv/bin/python" -B scripts/validate_dae_bundle_mat.py \
  --bundle "$ISU_CHEMICAL_DAE_BUNDLE" --mat "$dae_mat" --rpm "$dae_rpm" \
  --source "$HOME/capstone" --data-kind actual --output "$dae_audit"
```

미확인 상태는 blocked 및 종료 코드 1로 기록된다. 명시적으로 숫자 계산만 확인하려면 **다른 새 출력 폴더**에서 다음을 실행한다.

```bash
dae_diagnostic="$HOME/dae-bundle-audit/$(date -u +%Y%m%dT%H%M%S%NZ)-diagnostic"
"$HOME/capstone-web-venv/bin/python" -B scripts/validate_dae_bundle_mat.py \
  --bundle "$ISU_CHEMICAL_DAE_BUNDLE" --mat "$dae_mat" --rpm "$dae_rpm" \
  --source "$HOME/capstone" --data-kind actual --diagnostic-unconfirmed \
  --output "$dae_diagnostic"
```

`--diagnostic-unconfirmed`은 CLI 전용 명시적 계산 확인이다. 기존 수치 커널을 호출하지만 공개 predict_bytes의 RPM 검사를 변경하지 않는다. 상태는 diagnostic_only_rpm_unconfirmed, public_inference_allowed=false, prediction 공란이다. diagnostic_mse와 저장 threshold는 진단용 숫자이며 정상/이상·성능·지원 RPM으로 해석하지 않는다. 종료 코드 0이어도 지원 범위가 검증됐다는 뜻이 아니다. 이 옵션은 오류/손상/RPM 미지원 모델을 우회하지 않는다.

이미 근거 있는 지원 메타데이터를 가진 번들에 대해서만 기존 공개 경로가 정상/이상 결과를 출력한다. 지원 근거 없이 새 supported_rpms를 작성하는 기능은 없다.

다중 파일은 `--mat/--rpm` 대신 `--cases /실제/cases.csv --data-root "$PWD"`를 사용한다. 필수 열은 path/rpm이며 date/condition 및 raw_prediction은 선택이다. 원래 조건은 감사 메타데이터이며 추론 입력이 아니다. 기존 discover_looseness_cases.py의 목록도 사용할 수 있다.

### 산출물

- dae_mat_results.csv: 경로/RPM/날짜/조건/데이터 종류, 준비 상태, 공개 허용 여부, 진단 MSE/저장 threshold/허용된 경우 판정, 오류
- dae_validation.json: 파일별 결과, 번들 SHA256/원본 변경 검사, 학습 메타데이터 조사, 한계

출력은 저장소 밖 새 폴더만 사용한다. 운영 DB에 접근하지 않는다. 기존 번들과 MAT는 읽기 전용이며 결과/모델/데이터를 Git에 추가하지 않는다. 모델 로드 가능 범위에서만 계산하고 비유한값은 기존 추론 커널이 거부한다.

## 학습 출처 조사: 사실과 미확인

저장소 코드에서 확인한 내용:

- dae_legacy/train.py: normal_dataset_normalized.npz의 X_train/X_val/X_test 로드. 실제 학습은 X_train, 검증은 X_val을 사용한다.
- dae_legacy/calculate_threshold.py: 정상 X_val 복원 MSE의 percentile 95로 threshold 계산. 기존 저장 threshold를 이 작업에서 재계산하지 않는다.
- dae_legacy/check_normal_data.py: 배열 이름과 shape 확인용이며 분할 생성 코드는 아니다.
- dae_legacy/prepare_fault_dataset.py: DATES=[20260912,20260916,20260919], RPMS=[900,1200,1800]로 결함 평가 데이터 준비. 이 목록을 정상 학습 또는 모델 지원 범위로 취급하지 않는다.
- 정상 NPZ의 날짜/RPM별 split을 생성한 코드 및 해당 실제 가중치와의 대응은 저장소만으로 확인하지 못했다. AE의 날짜 분할을 DAE에 대입하지 않는다.

CLI --source는 기존 NPZ의 NPY header를 조사해 split 배열 크기를 기록한다. group/file/date/run/rpm 메타데이터의 작은 비객체 배열만 읽어 키별 날짜/RPM 관측치를 기록한다. object/pickle 배열 및 10,000개 초과 배열은 값 읽기를 생략한다. fault_dataset.npz가 있으면 같은 방식으로 별도 조사한다. 키에 train/val/test가 보이는 사실과 실제 독립 분할 검증은 다르다. 원본 파일 매핑/Run 독립성/학습 provenance가 없으면 supported_rpms=null, split_independence=unconfirmed를 유지한다.

## 검증 및 남은 한계

합성 번들·MAT만 사용하는 테스트를 추가했다. 경로/파일/해시/로드/RPM 상태, 합법적 합성 지원 번들 계산 일치, 명시적 오프라인 진단 뒤에도 공개 차단 유지, 메타데이터 관측과 지원 선언의 분리를 검사한다. 실제 WSL 번들/NPZ/MAT는 이 환경에 없으므로 성능 또는 실제 threshold/정규화 호환성을 새로 검증했다고 주장하지 않는다.

병합 전 WSL에서 위 기존 번들을 지정해 실제 상태/해시/모델 로드 확인, Normal1/2 및 결함별/날짜별/세 RPM의 독립 데이터 관계 조사, 진단 결과와 기존 legacy 보고서 비교를 수행해야 한다. AE·RF·DB 회귀 및 기존 결과 보존도 확인한다. 추론 시간의 score나 단일 MAT 계산을 탐지 정확도로 보고하지 않는다.


### 실행 환경 버전 주의

기존 DAE 번들을 생성한 NumPy/SciPy/PyTorch와 웹 전용 환경 버전이 같다는 사실은 아직 확인하지 못했다. 버전이 다르면 RPM 미확인보다 먼저 manifest_error가 발생할 수 있다. manifest의 runtime_versions와 실제 웹 환경을 비교하고, 검사를 우회하거나 원본 manifest를 편집하지 않는다. 필요한 경우 기존 legacy 산출물을 웹 환경에서 기존 validate_dae_wsl.py로 새 외부 폴더에 변환·호환성 검사한다(원본 가중치/통계/threshold 보존). 변환 성공도 RPM 검증이 아니다.

자동 검증은 기존 66개와 신규 6개, 총 72개 고유 테스트 범위다. 전체 71개 통과 후 추가된 V3 비공개 점수 차단 검사를 포함해 신규 6개를 재실행한다. 실제 WSL 결과는 아직 없다.
