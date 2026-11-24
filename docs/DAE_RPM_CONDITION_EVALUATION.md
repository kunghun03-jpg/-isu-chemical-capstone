# DAE 전체 RPM·조건 고정 임계값 오프라인 검증

## 목적과 현재 상태

V3의 공개 RPM 차단은 유지하면서 900/1200/1800RPM, Normal1/Normal2, IR/OR/Roller, Looseness0.05/0.1/Full, Unbalance_-1/-2의 실제 MAT를 평가한다. 계산 가능성, 정상/결함 MSE 분포, 기존 임계값의 관측 결과와 지원 범위 인증은 별개다.

사용자 WSL 번들/보고서/실제 data는 Codex에 없어 실제 RPM별 결과를 생성하지 않았다. 사용자 보고의 실행 환경 호환성 통과를 부정하거나 성능 근거로 확대하지 않는다. 동일 보고서를 기존 load_manifest가 다시 검증하며 자동 우회 옵션은 없다.

별도 main 기반 worktree에서 신규 스크립트/테스트/문서만 추가한다. 기존 미커밋 DAE 평가 코드 등 6개 파일은 수정하지 않는다. AE/RF/DB/Streamlit 및 모델·threshold·manifest에 쓰지 않는다.

## 기존 경로 재사용

1. SavedDAEBundleInference 생성: manifest/가중치/정규화/threshold 해시와 실행 환경/수치 호환성 근거 검증
2. 별도 오프라인 CLI에서 기존 SavedDAEInference.predict_bytes 수치 커널 호출: Welch 전처리 → 저장 mean/std → float32 파일 복원 MSE
3. 고정된 저장 threshold와 `MSE > threshold` 비교

이는 기존 오프라인 진단 경로이며 일반 V3 predict_bytes/check_rpm은 수정하거나 호출 옵션으로 해제하지 않는다. offline_prediction은 평가용 임계값 비교이며 V3 사용 허가가 아니다. supported_rpms=null, rpm_evidence=null, training_provenance=unverified를 변경하지 않는다. 버전/해시 검증 실패 시 비검증 레거시 모델로 fallback하지 않는다.

## WSL 실행

기존 환경과 코드를 유지하고 실제 프로젝트 루트에서 실행한다.

```bash
cd "/mnt/c/Users/kungh/OneDrive/바탕 화면/isu_chemical"
git fetch origin
git switch --track origin/feat/dae-rpm-condition-evaluation
# 로컬 브랜치가 있으면 git switch feat/dae-rpm-condition-evaluation
dae_output="$HOME/dae-evaluation/$(date -u +%Y%m%dT%H%M%S%NZ)"
"$HOME/capstone-web-venv/bin/python" -B scripts/evaluate_dae_rpm_conditions.py \
  --data-root "$PWD/data" \
  --bundle "$HOME/dae-validation/20261113T041136515404Z/bundle" \
  --runtime-report "$HOME/dae-runtime-check/20261114T072450568395435Z/web-verification/runtime_compatibility.json" \
  --source "$HOME/capstone" \
  --files-per-stratum 10 \
  --data-kind actual --output "$dae_output"
```

--runtime-report는 기존 ISU_CHEMICAL_DAE_RUNTIME_REPORT 규약을 이 프로세스에서만 설정하고 복원한다. 기존 보고서의 참조 JSON/NPZ와 코드 해시가 그대로 있어야 한다. 코드/번들/환경 근거가 다르면 실패하며 보고서나 manifest를 편집해서 통과시키지 않는다.

기본 선택은 **RPM×날짜×원래 조건별 정렬된 경로의 등간격 인덱스 최대 10개**다. 각 집단의 처음/중간/끝 구간을 포함하며 모델 결과를 보기 전에 선정한다. 가능한 표본이 적으면 전부 선택하고 부족 수를 기록한다. 날짜 제한이 없으면 발견된 모든 날짜를 포함한다. 특정 날짜만은 `--dates 20260912 20260916 20260919`를 추가한다.

전체 파일을 계산하려면 새 출력 폴더를 사용해 --files-per-stratum 대신 **--all-files**를 지정한다. 원본 MAT를 메모리에 한꺼번에 적재하지 않고 CPU 모델 한 개로 순차 처리한다. 중단 시 완료된 CSV와 가능한 부분 보고서를 보존한다. 기존 출력 폴더는 덮어쓰지 않으며 재개 기능은 없다. 같은 출력에 여러 프로세스를 실행하지 않는다.

## 산출물

- dae_results.csv: 경로/SHA256/날짜/RPM/조건/범주, MSE·저장 threshold·평가용 비교, 상태·오류, 모델 버전/data_kind
- dae_rpm_summary.csv: 정상/결함 별 유효 표본·이상 수·비율·Wilson95 구간, MSE 분포 및 관측 AUROC/AUPRC
- dae_condition_summary.csv: 원래 조건×날짜×RPM별 점수 분포·이상 비율·오류/중복
- dae_validation_report.json: 범위·샘플 선정·모델/환경/해시·출처 조사·누락/제한·별도 활성화 계획
- README.md: 한국어 관측 결과와 필수 질문/활성화 판단 한계
- discovery_inventory.csv: 전체 탐색·선정/미선정·제외 사유
- dae_mse_*.png: 날짜/조건별 MSE 분포와 저장 threshold, 실제 n 표시

실행 결과에 data_kind=actual/synthetic를 명시한다. 파일별 오류·NaN/Inf는 점수/판정 공란으로 남기고 비율 분모에서 제외한다. 동일 SHA256는 첫 성공 결과와 같다는 진단값을 기록하되 duplicate_content로 표시하고 중복본은 비율 분모에서 제외한다. 오류·중복이 특정 날짜/조건에 몰리면 결과에 선택 편향이 생길 수 있다.

성공 종료 코드 0은 선택 파일의 기술적 계산 완료이며 성능 인증이나 모든 조건 커버리지 통과가 아니다. 모델 검증 실패/파일 오류/원본 변경/중단은 종료 코드 1 또는 중단 상태다. 빈 집단의 비율·AUROC는 공란이며 0%로 표현하지 않는다. JSON의 coverage/sampling_shortfalls 및 README의 제한을 함께 확인한다.

## 학습 출처·중복

--source의 기존 normal_dataset_normalized.npz/fault_dataset.npz를 안전한 header와 작은 비객체 메타데이터 위주로 조사한다. 기존 provenance 함수를 재사용한다. 학습/검증/시험 크기·키, 날짜/RPM 관측을 기록하되 큰 행렬·pickle/object 자료를 로드하지 않는다. source가 없거나 손상돼도 조사 오류를 기록하고 수치 평가와 출처 불확실성을 구분한다.

평가 날짜가 메타데이터에 나타나면 potential_date_overlap으로 기록한다. 이것은 같은 파일/Run 또는 실제 학습 참여의 확정이 아니다. 반대로 날짜 교집합이 없다고 독립성을 증명하지 않는다. 정확한 Train/Validation/Test 파일 매핑과 Run 독립성은 이 도구가 인증하지 않으며 independent_test_performance_validated=false다.

저장소에서 확인된 코드 근거:

- `dae_legacy/train.py`는 `normal_dataset_normalized.npz`의 X_train/X_val/X_test를 읽으며, X_train에만 잡음을 추가해 정상 원본을 복원하도록 학습한다. 이 파일에는 날짜/RPM별 분할을 생성하는 코드가 없다.
- `dae_legacy/calculate_threshold.py`는 정상 X_val 복원 MSE의 95백분위수를 사용한다. 이번 도구는 이 계산을 재실행하거나 저장 threshold를 재작성하지 않는다.
- `dae_legacy/prepare_fault_dataset.py`는 20260912/20260916/20260919와 900/1200/1800RPM의 결함 목록을 열거한다. 이것은 결함 평가 데이터 수집 범위이며 정상 학습 참여 증거가 아니다.
- 정상 NPZ를 만드는 원본 분할 절차와 실제 파일별 Train/Validation/Test·Run 매핑은 현재 코드만으로 확정할 수 없다.

결함 준비 코드에 있는 세 날짜/세 RPM은 평가 데이터 목록이지 DAE 학습/지원 범위가 아니다. 동일 가중치가 어떤 정상 데이터에서 학습됐는지, mean/std와 threshold 출처가 그 분할과 일치하는지는 별도 자료가 필요하다.

## 결과를 보고 답할 질문

1–3. 각 RPM에서 status=ok인 **실제 고유 파일**이 있어야 해당 선택 파일의 계산 성공을 말할 수 있다. 합성 결과나 입력 크기가 같다는 사실만으로 실제 데이터 검증을 대신하지 않는다.

4. 정상/결함 MSE 중앙값·IQR·날짜/하위 조건 분포, 관측 AUROC/AUPRC를 함께 본다. 평균 하나나 특정 OR 파일만으로 전체 결함 분리를 주장하지 않는다.

5. 저장 threshold의 정상 관측 이상 비율과 결함 관측 이상 비율, Wilson 구간과 오류 제외를 함께 본다. threshold가 유효하다는 운영 결론은 사전 목표와 독립 검증 없이는 확정하지 않는다. Test에서 threshold를 최적화하지 않는다.

6–7. 이 실행은 지원 RPM을 등록하지 않는다. 모든 RPM은 blocked_pending_independent_review이며, 자료 부족/낮은 성능 RPM은 계속 차단한다. 다음은 **별도 변경 계획**이다.

- 누락 조건·날짜, 작은 표본, 수치 오류·중복 원인 해결
- 실제 정상 학습 출처·정규화/threshold 출처·날짜/Run 분리 확정
- 사전에 정상 오경보 허용치와 결함별 최소 검출 목표 설정 (Test에 맞춰 선택하지 않음)
- 미사용 날짜/Run에서 RPM별 Normal1/2·베어링 하위 조건·약한 불평형/느슨함을 평가
- 관측 성능이 낮거나 불확실하면 차단 유지
- 충분한 근거가 있는 RPM만 모델/데이터/코드 해시와 결과를 첨부한 별도 PR로 검토

동일 수치 계산 가능성과 탐지 성능, 단일 호환성 입력과 운영 범위는 구분한다. DAE MSE를 AE/OCSVM/IF 점수 절댓값 또는 공통 신뢰도로 비교하지 않는다.

## 테스트/병합 전

합성 3RPM·2날짜·10조건의 수치 계산, 기존 커널 일치, 공개 RPM 차단 유지, 데이터/모델 해시 불변, 런타임 실패 시 fallback 금지, NaN/Inf·중복/오류 분모, 표본 선정의 재현성을 확인한다. 실제 WSL 실행은 아직 미완료다.

- 기존 번들/threshold/운영 DB/Streamlit 불변
- 호환성 근거와 현재 코드/환경 일치
- 실제 coverage와 날짜/조건별 표본·오류 확인
- 정상/결함 비율·불확실성·중복과 학습 출처를 함께 보고
- 낮은 성능이나 미확인 RPM은 공개 추론 차단 유지

신규 합성 테스트 8개 통과. 기존 경로를 포함한 통합 실행 101개 통과 후 PNG 생성 테스트를 추가하여 신규 8개를 재실행했다. 실제 MAT 성능 결과는 포함하지 않는다.
