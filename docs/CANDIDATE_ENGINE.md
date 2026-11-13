# 학습 가능한 결함 후보 엔진과 V2 연결

## 구현 범위와 현재 검증 상태

MAT → 기존 AE → **AE 이상일 때만 저장된 RF 후보 엔진 추론** → 실제 특징·보류 이유 → DB·이력 경로다. AE 정상은 후보 엔진을 로드/호출하지 않고 ‘AE 정상 · 후보 추정 생략’으로 표시한다. 파형·FFT·관측 특징은 정상에서도 제공한다.

기존 정상 학습 AE의 전처리·가중치·정규화·임계값은 바꾸지 않는다. 새 후보 RF는 네 범주의 레이블을 사용하는 별도 지도학습이며, 기존 이상 탐지 모델과 구분한다. DAE·OCSVM·IF를 웹에 추가하거나 3X~5X 실험을 사용하지 않는다.

Codex에서는 실제 WSL EDA CSV·MAT에 접근하지 못했다. 합성 CSV·MAT와 임시 모델로 학습/아티팩트/웹/DB 연결을 검사했으며 **실제 결함 진단 성능은 검증하지 않았다**. 기존 LODO 점수를 새 배포 모델의 검증 성능으로 재사용하지 않는다.

## Train / Validation / Test

CLI에 세 날짜 목록을 반드시 명시하며 서로 겹치면 중단한다. 예시:

| 역할 | 날짜 | 사용 |
| --- | --- | --- |
| Train | 20260912 | median imputer·RF·훈련 min/max 범위 학습 |
| Validation | 20260916 | RPM·후보 클래스별 보류 margin 선택 |
| Test | 20260919 | 고정 모델·고정 정책의 성능을 한 번 평가 |

선택한 Train 날짜의 가용 RPM을 함께 학습하는 단일 RF다. 기존 RPM별 LODO 모델들과는 다른 배포 모델이며 평가 결과를 구분한다. 지정하지 않은 날짜는 학습에 몰래 추가하지 않는다. 모델은 **Train만 학습한 상태 그대로** 저장하고 Validation/Test를 합친 재학습을 하지 않는다. 학습 날짜가 하나뿐이라는 한계와 Run 독립성 미확인을 기록한다. 기존 연구에서 이미 관찰했던 날짜를 재사용하는 후속 실험이므로 새로운 최종 시험이라고 주장하지 않는다.

`split_files.csv`에 각 파일의 경로·날짜·RPM·원래 조건·네 범주·기록된 MAT 해시와 split을 저장한다. EDA 로더의 중복 경로·중복 해시·메타데이터 검사도 재사용한다. 이는 기록 일관성 확인이며 학습 시 원본 MAT를 재계산하지 않는다. Train에는 네 클래스가 각각 최소 2개 필요하다.

## 동일 특징·전처리

기존 9개 순서를 고정한다: RMS, Peak, Crest Factor, 1X, 2X, 저·중·고주파 평균제곱, 엔벨로프 AC RMS. 경로·파일명·날짜·원래 조건·RPM은 숫자 입력 특징에 포함하지 않는다. RPM은 적용 범위 및 Validation 정책을 선택할 때만 사용한다.

웹은 기존 MAT 로더와 EDA `analyze_signal`을 사용한다. CH3, fs=25600Hz, 65536샘플, 전체 기록 Hann FFT, 명목 1X·2X ±1Hz, 대역 [0,200)/[200,2000)/[2000,10000]Hz, 엔벨로프 carrier 2–8kHz·양끝 .1초 제거 설정을 고정한다. 다른 EDA 설정은 학습 단계에서 거부한다. 명목 RPM은 실제 회전수 계측값이 아니다.

훈련 fold의 median imputer와 RF를 하나의 Pipeline으로 저장한다. 전체 결측 훈련 열은 기존과 같이 0을 사용한다. 별도 표준화·특징 선택은 하지 않는다. 일부 결측도 Pipeline 계산은 가능하지만 자동 후보명은 보류한다. NaN/빈 값 외 Inf·음수·범위 오류, 전체 결측, 특징 스키마 오류는 거부한다.

RF는 기존 설정인 200 trees, max_depth=5, min_samples_leaf=2, class_weight=balanced, random_state=42, n_jobs=1을 유지한다. 웹에서 fit이나 정책 조정은 수행하지 않는다.

## 보류 정책과 활성화

원시 네 범주: Normal, Bearing, Looseness, Unbalance. 표시가 허용된 경우 각각 베어링 결함 의심 / 느슨함 의심 / 불평형 의심으로 변환한다. **확정 진단이 아닌 탐색적 후보**다.

점수는 RF의 `predict_proba` 출력이지만 **보정된 확률·신뢰도가 아니다**. margin은 최고 점수−두 번째 점수다. Validation에서 각 RPM·후보 클래스마다:

1. 실제 해당 클래스 표본이 최소 5개 있어야 한다.
2. 후보 예측이 양수 margin, 결측 없음, Train 전체 min/max 안인 표본을 사용한다.
3. Validation에서 관측한 margin들을 후보 경계값으로 비교한다.
4. 제시 표본 최소 5개·경험적 precision ≥0.8인 경계 중 제시 수가 최대인 것을 선택한다.
5. 충족하지 못하면 그 RPM·클래스 규칙은 생성하지 않는다.

5개와 0.8은 사전 고정한 탐색적 최소 조건이며 충분한 통계적 검증을 보장하지 않는다. 같은 Validation에서 경계를 선택했으므로 그 precision은 낙관적이다. Test에서는 경계나 클래스 활성 상태를 조정하지 않고 평가만 한다. 선택된 규칙은 전반적 진단 성능 보장이나 확률 보정이 아니다.

웹 보류 사유:

- 모델 경로 미설정, 파일 누락·손상, 버전/스키마 오류
- AE 이상인데 RF가 Normal 예측: **판정 불일치**
- 해당 RPM이 Train에 없거나 해당 RPM·클래스의 Validation 정책이 없음
- 점수 차이가 선택된 margin 미만이거나 동률
- 특징 결측 또는 훈련 min/max 이탈
- 자동 후보명 표시 설정이 꺼져 있음

훈련 min/max 검사는 범위 밖 입력을 거르는 보수적 안전 조건이며 도메인 적합성을 증명하지 않는다. 범위 안이어도 오분류할 수 있다. Looseness의 기존 일반화 실패를 고려해 검증 부족 규칙을 임의로 활성화하지 않는다.

**자동 후보명은 기본 비활성(`ISU_CHEMICAL_CANDIDATE_LABELS=0`)이다.** 이 상태에서도 AE 이상 시 실제 RF 추론을 실행하며 원시 분류·미보정 점수·margin·보류 이유를 상세에서 확인한다. 실제 Test의 보류율과 선택 정확도를 검토한 후 명시적으로 1을 설정할 수 있다. 1을 설정해도 표본 부족·미검증 클래스 정책을 강제로 우회할 수 없다. 성능이 낮으면 0을 유지하고 새 독립 데이터 검증을 준비해야 한다. 시연 파일 정답에 맞춰 policy/manifest를 편집하지 않는다.

## 아티팩트와 오류 처리

새 폴더에 아래 파일을 생성한다. 기존 결과/모델 폴더는 덮어쓰지 않는다.

```text
model.joblib                 # Train 전용 fitted imputer + RF
manifest.json                # 버전·특징 순서·계산 설정·패키지 버전·model SHA256
                             # Train 범위/RPM, split 날짜·파일 수·클래스 구성, 보류 정책
training_report.json         # 출처 SHA256·상태·검증/시험 지표·imputer 통계·manifest 해시
split_files.csv              # Train/Validation/Test 전체 파일 목록, 해시
unused_eda_files.csv
excluded_eda_rows.csv
validation_predictions.csv
test_predictions.csv
selective_metrics.csv        # 보류율, 제시 수, 제시한 예측의 정확도
class_metrics.csv            # 원시 네 범주 precision/recall/F1/support
raw_confusion.csv            # 원시 네 범주 행렬
policy_confusion.csv         # ABSTAIN을 포함한 제시 결과
```

최소 `model.joblib`, `manifest.json`, `training_report.json`, `split_files.csv`가 같은 폴더에 있어야 한다. 로더는 완료 상태, manifest/모델/목록 해시, 9개 특징 순서·설정, numpy/scipy/sklearn/joblib 버전, Pipeline 구조, 실제 RF 설정을 검증한다. 불일치하면 후보만 보류하고 기존 AE 결과를 유지한다. 직접 생성한 신뢰할 수 있는 로컬 아티팩트만 사용한다. MAT 업로드로 모델 pickle을 받지 않는다.

평가 JSON에는 원시 네 범주의 Macro F1/Balanced Accuracy/클래스별 지표도 저장한다. 보류율과 선택 정확도는 **모든 Test EDA 파일**을 후보 단계에 넣은 모의 평가다. 실제 AE 이상으로 필터링한 집합이 아니므로 최종 시스템 전체의 성능으로 주장하지 않는다. Normal 예측은 이 평가에서 보류가 되며, 실제 웹에서 AE 정상이면 후보 실행 자체가 생략된다. 제시 표본이 0이면 선택 정확도는 null이다. Validation 선택 정확도와 Test 선택 정확도를 구분한다.

## WSL 정확한 명령

기존 서버를 `Ctrl+C`로 종료한 뒤 실제 소스 Git 저장소에서 새 브랜치를 가져온다. 이미 로컬 브랜치가 있으면 `git switch feat/trainable-candidate-engine`을 사용한다. 로컬 변경을 강제 초기화하지 않는다.

```bash
git rev-parse --show-toplevel
git fetch origin
git switch --track origin/feat/trainable-candidate-engine
```

### 1. 학습·정책 선택·시험·저장: 한 명령

```bash
candidate_bundle="$HOME/candidate-engines/$(date -u +%Y%m%dT%H%M%S%NZ)-rf-eda9"
"$HOME/capstone-web-venv/bin/python" -B -m scripts.train_candidate_engine \
  --eda-dir /home/kungh/fault-signatures/20261113T131205447206Z \
  --train-dates 20260912 \
  --validation-dates 20260916 \
  --test-dates 20260919 \
  --output "$candidate_bundle"
```

세 날짜가 실제 CSV에 없거나 입력 구성이 다르면 중단한다. 임의로 날짜를 합쳐 오류를 우회하지 않는다. 입력 CSV·JSON은 읽기 전용이며 원본 MAT/AE 파일은 변경하지 않는다. 모델/데이터/리포트를 Git에 올리지 않는다.

### 2. 평가 검토

```bash
"$HOME/capstone-web-venv/bin/python" -B - "$candidate_bundle" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
r = json.loads((p / 'training_report.json').read_text())
print('status:', r['status'], 'complete:', r['completed'], 'unchanged:', r['inputs_unchanged'])
print('splits:', r['split_counts'], 'classes:', r['class_counts'])
for split in ('validation', 'test'):
    for m in r[split + '_metrics']:
        raw = m['raw_four_class_metrics']
        print(split, 'RPM=', m['rpm'] or 'ALL', 'n=', m['n'],
              'F1=', raw['macro_f1_fixed_four_classes'],
              'BA=', raw['balanced_accuracy_present_classes'],
              'abstain=', m['abstention_rate'], 'accepted=', m['accepted_n'],
              'selective accuracy=', m['selective_accuracy'])
print('Validation policy availability:')
for d in r['policy']['diagnostics']:
    print(d)
print('Test files:', p / 'test_predictions.csv')
PY
```

전체 정확도만 보지 말고 Normal·Looseness·Unbalance Recall, RPM별 결과, 클래스 누락과 보류 수를 확인한다. Test 성능이 낮으면 자동 표시를 활성화하지 않는다. Test를 보고 임계값을 조정한 뒤 같은 Test를 독립 검증이라고 주장하지 않는다.

### 3. 웹 실행: 우선 보류 모드

```bash
export ISU_CHEMICAL_V2_MODEL=autoencoder
export ISU_CHEMICAL_CANDIDATE_BUNDLE="$candidate_bundle"
export ISU_CHEMICAL_CANDIDATE_LABELS=0
"$HOME/capstone-web-venv/bin/python" -B -m streamlit run streamlit_app_v2.py \
  --server.address 127.0.0.1 --server.port 8501
```

새 터미널이면 `candidate_bundle` 변수를 실제 생성 폴더의 절대 경로로 다시 지정한다. 기존 AE 번들·DB 환경 설정은 유지한다. `http://localhost:8501`에서 AE 이상 파일을 분석하고 ‘후보 엔진 출력 · 보류 판단 근거’를 열어 실제 RF 출력·버전을 확인한다.

**실제 평가를 검토해 표시를 허용하기로 한 경우에만**, 서버를 종료한 후 동일 아티팩트로 다음과 같이 실행한다.

```bash
export ISU_CHEMICAL_CANDIDATE_LABELS=1
"$HOME/capstone-web-venv/bin/python" -B -m streamlit run streamlit_app_v2.py \
  --server.address 127.0.0.1 --server.port 8501
```

보류 정책을 충족한 경우만 후보명이 나온다. 기준 미달 클래스/입력은 설정 1에서도 불명확이다. 신규 모델이나 설정이 바뀌면 이전 세션 결과를 지우고 재분석하며 기존 DB 후보를 덮어쓰지 않는다.

## UI·DB 호환

`candidate_reports`의 테이블/PK는 바꾸지 않는다. JSON schema 2(`ch3_candidate_engine_v2`)에 engine 종류·모델 버전/해시·원시 예측·미보정 점수·보류 여부/이유·Validation 정책과 실제 특징을 기록한다. 분석 시각·RPM은 기존 analysis 행과 연결한다. schema 1 근거 전용 기록도 계속 읽고, 같은 분석에 둘 다 있으면 v2를 우선 표시한다. 기존 AE-only 기록은 후보 없음으로 표시된다.

AE 점수·임계값·판정은 원래 result를 수정하지 않는다. 기존 원자적 저장과 중복 방지를 유지하며 데이터·모델·기존 이력을 삭제/갱신하지 않는다. 운영 DB는 이전 통합 문서의 SQLite backup 절차로 먼저 백업한다.

## 실제 MAT 시연 검증

1. **성능 확인용**으로는 `test_predictions.csv`에 기록된 Test 날짜 파일을 사용한다. Train/Validation 파일 시연은 동작 확인이며 독립 성능 검증으로 제시하지 않는다.
2. Normal1/Normal2, IR/OR/Roller, Unbalance_-1/-2, 각 Looseness 조건을 고르고 측정 RPM을 맞춘다. 다른 날짜·RPM에서도 반복한다.
3. 기존 AE 점수·임계값·판정이 종전과 같은지 비교한다. AE 정상은 RF 미호출·추정 생략이어야 한다.
4. AE 이상은 실제 RF 버전과 원시 출력이 나타나야 한다. 모델이 없으면 오류 사유와 불명확을 표시해야 한다.
5. RF Normal + AE 이상은 판정 불일치로 보류해야 한다. 낮은 margin/범위 이탈/결측/미검증 클래스도 보류해야 한다.
6. 파일명만 바꾼 같은 바이트는 동일 특징·후보를 내야 한다. 정답 이름에 맞춰 파일이나 기준을 바꾸지 않는다.
7. 후보를 제시한 경우에도 ‘확정 진단이 아닌 탐색적 후보’ 안내가 보여야 한다. 점수를 신뢰도 %로 설명하지 않는다.
8. 저장·중복 저장·서버 재시작 후 이력에서 AE 값과 후보 버전·보류 이유·특징이 같은지 확인한다. 과거 schema 1·AE-only 이력도 확인한다.

## 자동 테스트·한계

```bash
MPLCONFIGDIR=/tmp/candidate-engine-mpl \
"$HOME/capstone-web-venv/bin/python" -B -m unittest \
  tests.test_candidate_engine tests.test_presentation_v2 tests.test_dashboard_v2 \
  tests.test_ae_v2_parity \
  tests.test_history.HistoryTests.test_group_filters_detail_csv_and_repeated_save \
  tests.test_history.HistoryTests.test_atomic_rollback_and_concurrent_duplicate \
  tests.test_history.HistoryTests.test_separate_process_restart_preserves_and_deduplicates -v
```

합성 신호에서 실제 저장 Pipeline을 읽고 추론한 결과는 배선·계산 호환 검증이다. 물리적 Bearing/Looseness/Unbalance 판별 성능이 아니다. 기존 V1 이력 UI의 실 MAT 의존 테스트는 실제 파일 부재로 이 환경에서 수행하지 못한다. 실제 WSL 학습·평가 보고서와 MAT 시연을 확인한 뒤 배포 및 자동 후보 표시를 결정한다.

2026-11-13 클라우드 확인: 신규 10개와 기존 회귀 23개로 총 33개 테스트 통과(중복 실행 제외). Python 3.14.4 / NumPy 2.4.6 / SciPy 1.18.1 / scikit-learn 1.9.0 / Streamlit 1.67.0에서 검사했다. Chromium/Playwright에서 실제 저장된 합성 RF 모델로 MAT→AE→후보→DB→이력 재조회가 통과했고 page error는 0개였다. 실제 사용자 CSV·MAT 검증 결과가 아니다.
