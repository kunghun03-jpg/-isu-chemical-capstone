# DAE 버전 불일치와 수치 호환성 검증

## 직접 원인과 확인 범위

기존 load_manifest는 manifest.runtime_versions와 현재 NumPy/SciPy/PyTorch를 정확히 비교했다. 사용자 확인 조합은 생성 환경 2.5.4 / 1.18.1 / 2.15.0+cpu, 웹 환경 2.4.6 / 1.18.1 / 2.14.1+cpu다. 이 불일치는 차단 이유이지만 실제 계산 비호환의 증거는 아니다.

**이 조합은 Codex에서 수치 검증하지 못했다. 실제 MAT·WSL 번들과 생성 환경이 없으므로 자동 허용하지 않는다.** 공용 가상환경의 버전을 설치/변경하지 않고 아래 두 환경 절차를 준비했다.

## 허용 정책

- manifest/파일 SHA256·스키마·전처리·정규화·가중치 검증은 유지한다.
- 생성 환경과 정확히 같은 버전은 기존 정책을 유지한다.
- 버전 불일치에는 기대/실제 버전과 bundle_id를 Python logging WARNING으로 기록한다.
- 불일치 환경은 기본 차단이다. 운영자가 `ISU_CHEMICAL_DAE_RUNTIME_REPORT`로 명시한 외부 수치 근거가 있어야 한다.
- 근거는 원본 manifest/가중치/통계/threshold 해시·bundle_id·원본/대상 버전·대상 Python/OS/아키텍처·실제 MAT SHA256·현재 코드 해시에 묶인다. 다른 번들/버전/코드/합성 자료는 허용 근거로 사용하지 않는다.
- 통과 flag만 믿지 않는다. source/target JSON과 NPZ 해시를 재검사하고 저장된 전처리·복원·MSE를 허용 오차로 다시 비교한다. 배열 dtype과 threshold는 정확히 같아야 하며 진단 경계 판정이 달라져도 차단한다.
- 합성 결과는 도구/정책 시험용이며 운영 불일치 허용 근거가 될 수 없다.
- 이 보고서는 운영자가 신뢰하고 관리하는 로컬 검증 자료다. 암호학적 서명이나 외부 인증은 아니다. JSON/NPZ를 임의 생성·수정해 신뢰 근거를 위조하지 않는다. 보고서 폴더를 보호하고 Git에 올리지 않는다.
- 단일 파일 통과는 그 입력에서의 수치 호환성 근거이며 모든 신호/하드웨어에서의 완전 동일성 보장이 아니다. 다양한 실제 입력을 별도로 검증하고 검토해야 한다. 모델 정확도·RPM 지원 검증은 별개다.

supported_rpms=null과 training_provenance=unverified는 수정하지 않는다. **실행 버전 검증을 통과해도 V3는 지원 범위 미확인으로 일반 추론을 계속 차단한다.** 런타임 보고서는 RPM 검사를 우회하지 않는다. AE/OCSVM/IF/RF/DB 및 보류 정책은 변경하지 않는다.

## 허용 오차 (고정, Test로 조정 금지)

| 비교 값 | rtol | atol |
|---|---:|---:|
| log spectrum·저장 정규화 입력 | 1e-6 | 1e-7 |
| 복원 스펙트럼 | 1e-5 | 1e-6 |
| float32 커널에서 계산한 파일 MSE | 1e-5 | 1e-7 |
| 저장 threshold | 정확 일치 | 정확 일치 |

float32 처리의 작은 플랫폼 오차를 확인하기 위한 사전 고정 기준이며, 해당 실제 모델에서 충분하다고 검증된 성능 오차 범위는 아니다. 하나라도 실패하면 보고서를 허용하지 않는다. 두 MSE가 threshold를 사이에 두고 갈리면 수치 오차 이내여도 실패한다.

## WSL 절차

먼저 최신 브랜치를 사용하고 기존 미커밋 작업을 보존한다. 번들/가중치/정규화/임계값/패키지 버전은 바꾸지 않는다.

```bash
git fetch origin
git switch --track origin/feat/dae-runtime-attestation
# 이미 브랜치가 있으면 git switch feat/dae-runtime-attestation
export ISU_CHEMICAL_DAE_BUNDLE="$HOME/dae-validation/20261113T041136515404Z/bundle"
read -r -p '기존 보고서 검증에 사용한 동일 MAT 절대 경로: ' same_mat
reference_dir="$HOME/dae-runtime-check/$(date -u +%Y%m%dT%H%M%S%NZ)-source"
target_dir="$HOME/dae-runtime-check/$(date -u +%Y%m%dT%H%M%S%NZ)-web"
```

### 1. 생성 환경에서 기준 수치 내보내기

아래 source_python은 **manifest와 정확히 같은 버전 조합인 기존 Python**이어야 한다. `~/capstone/.venv`가 실제로 그 환경인지는 확인이 필요하다. 다른 환경이면 기존 정확한 실행 파일 경로로 바꾼다. 새 패키지를 설치하지 않는다. CLI가 생성 환경 버전 일치를 강제한다.

```bash
source_python="$HOME/capstone/.venv/bin/python"
"$source_python" -c 'import numpy, scipy, torch; print(numpy.__version__, scipy.__version__, torch.__version__)'
"$source_python" -B scripts/check_dae_runtime_compat.py export \
  --bundle "$ISU_CHEMICAL_DAE_BUNDLE" --mat "$same_mat" --data-kind actual \
  --legacy-report "$HOME/dae-validation/20261113T041136515404Z/report.json" \
  --output "$reference_dir"
```

기존 report.json의 mat_sha256/bundle_id/runtime_versions와 실제 입력을 비교하고, 인접한 legacy_reference.npz의 features/input_spectrum/reconstructed_spectrum/mse/threshold와도 비교한다. 같은 MAT가 아니거나 예전 수치 파일이 없으면 실패한다. 기존 report.json의 상태 문구만으로 수치를 검증하지 않는다.

다른 실제 MAT를 새로 검증할 때에는 --legacy-report를 생략하여 새로운 source 기준을 만들 수 있다. 이 경우 옛 보고서와 일치했다고 주장하지 않는다. 생성 환경에 접근할 수 없으면 기준 수치를 새로 만들 수 없으므로 대상 조합을 허용하지 않는다.

### 2. 현재 웹 환경에서 같은 MAT 비교

```bash
"$HOME/capstone-web-venv/bin/python" -B scripts/check_dae_runtime_compat.py verify \
  --bundle "$ISU_CHEMICAL_DAE_BUNDLE" --mat "$same_mat" --data-kind actual \
  --reference "$reference_dir/reference.json" \
  --legacy-report "$HOME/dae-validation/20261113T041136515404Z/report.json" \
  --output "$target_dir"
```

원본 무결성을 검사한 뒤 기존 DAE 구조·정규화·float32 MSE를 오프라인으로 계산한다. 공개 번들 추론에 버전/RPM 우회 옵션을 추가하지 않는다. source와 같은 입력 SHA256·동일 코드·동일 번들인지 확인한다. 비교 실패 시 유효 runtime_compatibility.json이 생성되지 않으며 계속 차단된다. 로그의 max_abs_errors와 기존 보고서 비교 결과를 검토한다.

### 3. 실제 수치 통과 자료를 검토한 후만 지정

```bash
export ISU_CHEMICAL_DAE_RUNTIME_REPORT="$target_dir/runtime_compatibility.json"
export ISU_CHEMICAL_V2_MODEL=autoencoder
export ISU_CHEMICAL_CANDIDATE_LABELS=0
"$HOME/capstone-web-venv/bin/python" -B -m streamlit run streamlit_app_v3.py
```

이후 상태는 버전 오류에서 **지원 범위 미확인**으로 바뀌어야 한다. 공개 점수·판정이 나오면 안 된다. 새 환경/코드/번들에는 기존 보고서를 재사용하지 않는다. 참조 폴더/target JSON/NPZ는 런타임 재검사에 필요하므로 이동·삭제하지 않는다. 보고서를 미설정하면 불일치 환경은 다시 차단된다.

## 파일과 테스트

reference.json/reference.npz: 생성 버전, 실제 입력 SHA256, 코드/번들 해시와 전처리·복원 수치.
target.json/target.npz: 웹 환경의 동일 입력 결과.
runtime_compatibility.json: 비교 허용 오차·최대 절대 오차와 source/target 해시·버전, performance_validated=false/rpm_support_validated=false.

출력은 저장소/번들 밖의 새 폴더만 허용한다. 수치 배열에는 실제 데이터에서 나온 정보가 있으므로 외부 공유/Git 업로드하지 않는다.

단위 테스트는 합성 가중치/MAT와 모의 버전 식별자를 사용한다. 모의 조합은 정책 테스트이며 실제 NumPy/PyTorch 교차 버전 검증으로 보고하지 않는다. 실제 MAT 경로·SHA256과 요청한 실제 버전 조합의 허용 근거는 WSL 실행 후에만 채울 수 있다.

## 병합 전 확인

- 원본 번들/통계/threshold 및 환경 버전 불변
- 동일 실제 MAT와 SHA256, 생성/웹 버전 로그 확인
- 기존 legacy_reference 수치 및 새 두 환경의 모든 비교 통과
- 근거 파일/코드 해시, 고정 허용 오차와 threshold 경계 일치 확인
- 자료 없는 버전/다른 모델/다른 입력의 자료는 차단
- supported_rpms=null 공개 추론 차단 및 training_provenance 유지
- AE/RF/기존 DB와 보류 정책 불변
- 실행 호환성 통과를 RPM 지원·진단 성능으로 표현하지 않음


검증 결과: 전체 84개 합성/회귀 테스트 통과 후 최종 정책 변경에 대해 신규 호환성 테스트 7개를 재실행한다. 실제 사용자 MAT 및 요청된 실제 교차 버전 조합은 미검증이며 자동 허용 조합은 추가하지 않았다.
