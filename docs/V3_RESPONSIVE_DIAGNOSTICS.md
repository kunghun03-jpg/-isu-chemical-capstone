# V3 레이아웃 및 모델 진단

## 화면 수정

V3만 별도 CSS를 적용한다. 1920px에서는 AE·모델 표·RF의 세 열을 유지한다. 1051–1500px에서는 AE·모델 표를 첫 줄, RF를 두 번째 줄 전체 폭으로 배치한다. 1050px 이하에서는 기존 세로 배치와 함께 하단 요약도 줄바꿈한다. 네이비 사이드바는 초기 expanded 상태로 표시하며 상단 네이비 헤더를 유지한다. 세로 스크롤은 필요하며 모든 카드를 한 화면 안에 축소하지 않는다. 카드의 가로 잘림·겹침을 방지하는 것이 목표다. 이전 캡처의 사이드바/헤더 미노출은 스크린샷의 잘린 범위 또는 접힘/스크롤 상태일 수 있어 원인을 단정하지 않는다.

## DAE ‘모델 미준비’

DAE는 `ISU_CHEMICAL_DAE_BUNDLE`이 설정되면 그 폴더를, 아니면 저장소 `outputs/dae_bundle`을 사용한다. 기존 bundle_status가 manifest/필수 파일에 FileNotFoundError를 받으면 모델 미준비다. 레거시 PTH/NPZ 파일만 ~/capstone에 있어도 이 경로의 번들로 자동 연결되지 않는다.

V3 모델 카드의 ‘DAE가 모델 미준비인 이유 확인’ 및 시스템 설정에서 실제 조회 폴더와 다음 파일 존재 여부를 확인할 수 있다.

- manifest.json
- autoencoder_model.pth
- normalization.npz (mean/std)
- threshold.npz

파일이 모두 있어도 해시·구조·런타임·RPM 검증을 통과해야 한다. 실제 WSL 파일에 접근하지 못한 Codex는 해당 PC에서 어느 파일이 없는지 확정하지 않는다. 합성 테스트에서는 미설정 폴더의 manifest 부재를 확인했다.

### 기존 학습 산출물이 있는 경우

가중치·통계·임계값을 바꾸거나 재학습하지 않고 기존 변환/호환성 도구를 사용한다. 실제 입력 MAT를 선택해 아래를 실행한다.

```bash
read -r -p '실제 DAE 호환성 검사 MAT 절대 경로: ' dae_mat
dae_report="$HOME/dae-validation/$(date -u +%Y%m%dT%H%M%S%NZ)"
PYTHONPATH="$PWD" "$HOME/capstone-web-venv/bin/python" -B scripts/validate_dae_wsl.py \
  --source "$HOME/capstone" --legacy-code "$PWD/dae_legacy" \
  --mat "$dae_mat" --output "$dae_report"
export ISU_CHEMICAL_DAE_BUNDLE="$dae_report/bundle"
```

source에는 기존 autoencoder_model.pth, normal_dataset_normalized.npz(mean/std), normal_threshold.npz(threshold)가 필요하다. band_statistics.npz는 기존 검사 도구의 선택적 통계 검사용이며 새 번들의 필수 추론 파일은 아니다. 검사 실패 시 report.json을 확인하고 연결 완료로 주장하지 않는다.

이 절차는 지원 RPM을 확정하지 않는다. 정상 변환 후에도 ‘RPM 적용 범위 미확인’으로 차단되는 것이 올바르다. 기존 학습 출처·날짜/Run 분리·RPM별 성능 근거가 없는 상태에서 manifest를 편집해 RPM을 임의 추가하지 않는다.

### 산출물이 없는 경우

이번 작업은 DAE 학습을 수행하지 않는다. 기존 원본 학습 산출물을 확보하는 것이 우선이다. 별도 승인된 후속 연구에서 레거시 구조/전처리를 유지한 정상 Train 학습, 분리된 Normal Validation 통계·임계값 산출, 미사용 Test/RPM별 평가와 출처 기록이 필요하다. 그런 새 모델은 기존 검증 모델과 다른 버전으로 별도 보관하고 자동 운영 적용하지 않는다. 없는 가중치나 통계를 임의로 만들어 연결하지 않는다.

## OCSVM 점수 0과 이상 판정

기존 SavedModelInference는 다음 계산을 사용한다.

1. CH3 8구간 → Hann FFT → log1p → 저장 Train mean/std 정규화
2. 각 구간의 `score_samples(features)` 호출
3. 부호 변환 `segment_anomaly_score = -score_samples`
4. 파일 점수 = 8구간 평균
5. **파일 점수 > 저장 file_threshold**이면 이상

따라서 0 > -0.37434146이면 이상이다. 점수가 음수일 수 있고 0은 확률/정상 기준이 아니다. V3는 새로운 임계값이나 부호 변환을 추가하지 않는다. UI는 .8g로 표시하므로 정확한 값을 아래 도구의 repr와 구간 값으로 확인한다. RBF kernel underflow나 학습 범위 이탈은 가능한 조사 가설이지 점수 0만으로 확정되는 원인이 아니다. 실제 사용자 파일의 원점수는 WSL에서 검사해야 한다.

## 실제 MAT 진단 실행

기존 V3 브랜치가 main PR #17에 병합된 상태를 기반으로 한다. 기존 미커밋 작업을 강제로 덮어쓰지 않는다.

```bash
git fetch origin
git switch --track origin/feat/v3-responsive-model-diagnostics
# 로컬 브랜치가 있으면 git switch feat/v3-responsive-model-diagnostics
read -r -p '분석할 실제 MAT 절대 경로: ' mat_path
read -r -p '측정 RPM (900/1200/1800): ' mat_rpm
model_report="$HOME/v3-model-diagnostics/$(date -u +%Y%m%dT%H%M%S%NZ).json"
"$HOME/capstone-web-venv/bin/python" -B scripts/diagnose_v3_models.py \
  --mat "$mat_path" --rpm "$mat_rpm" \
  --bundle-root "${ISU_CHEMICAL_V2_BUNDLE_ROOT:-$PWD/outputs/inference_bundles/date_prelim_20261106}" \
  --data-kind actual --output "$model_report"
```

JSON에는 AE/OCSVM/IF의 기존 기본 로더 경로와 V3 선택 로더 경로 결과의 exact_match, OCSVM 8개 score_samples/부호 변환 점수/파일 평균/저장 임계값/repr/판정, 정규화 입력 범위, 번들 버전, DAE 폴더/파일 상태를 기록한다. 참고 모델 파일 누락으로 기본 로더가 실패하면 classic_error를 숨기지 않는다. 종료 코드 1은 클래식 비교 실패 또는 계산/입력 오류이며 자료 부족 상태를 성공으로 해석하지 않는다.

OCSVM은 원래 오프라인 vibration.one_class.anomaly_scores 및 comparison_metrics.file_scores/predictions도 직접 호출하여 값과 판정을 비교한다. 이것은 같은 저장 번들 계산 경로의 일치 검사다. 과거 오프라인 평가에서 다른 모델/정규화/파일을 사용했는지는 과거 평가 해시와 추가 비교해야 하며, 이 명령만으로 과거 Test 전체 동일성을 증명하지 않는다. 운영 DB에 접근하지 않고 출력 JSON은 저장소 밖 새 파일에만 쓴다. 실제 MAT와 모델 파일은 읽기 전용이다.

```bash
export ISU_CHEMICAL_V2_MODEL=autoencoder
export ISU_CHEMICAL_CANDIDATE_LABELS=0
# 기존 번들/DB 설정 유지. 운영 DB 사용 전 백업 권장.
"$HOME/capstone-web-venv/bin/python" -B -m streamlit run streamlit_app_v3.py --server.port 8501
```

## 검증 및 병합 전 확인

합성 테스트와 Chromium으로 확인하며 실제 사용자 MAT는 이 환경에 없다. RF 정상 생략/최종 보류 정책 및 모델 학습·임계값은 변경하지 않는다. 실제 Normal1/2, IR/OR/Roller, Unbalance, Looseness와 세 RPM에서 진단 JSON을 저장한 뒤 웹 점수와 비교한다.

- 1920×1080에서 네이비 헤더/사이드바 및 세 열 확인
- 1366×768에서 AE/모델 표와 다음 줄 RF 카드의 가로 잘림 없음 확인
- DAE 조회 경로/필수 파일/해시/정규화/임계값과 RPM 차단 확인
- OCSVM 원점수·부호·8구간 평균·임계값 및 exact_match 확인
- AE 정상 RF 생략, 이상 원시 예측과 최종 보류 분리 확인
- 파형/FFT/특징, 저장/중복 저장/재시작 후 이력 확인
- 모델 개선·확률·진단 신뢰도를 입증했다고 표현하지 않음

Codex에서는 신규 3개와 기존 63개, 총 66개 합성/회귀 테스트가 통과했다. 합성 MAT CLI 실행에서 AE/OCSVM/IF 기본 경로와 V3 결과가 일치했고 OCSVM 원래 오프라인 helper 비교도 통과했다. 실제 사용자 MAT 점수 0의 원인은 아직 WSL 검사가 필요하다. Chromium 두 해상도의 카드 좌표·가로 넘침·저장/이력을 검사했고 오류가 없었다.
