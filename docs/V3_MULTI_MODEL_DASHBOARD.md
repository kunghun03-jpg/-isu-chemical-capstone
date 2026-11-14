# V3 다중 AI 모델 대시보드

`streamlit_app_v3.py`를 새 진입점으로 추가한다. V2 앱 파일과 UI는 변경하지 않는다. 기존 V2/통합 검증/오류 조사 변경은 main PR #14–16에 병합되어 있다.

## 운영 판정과 참고 결과

- 운영 기준은 기존 AE다. 같은 SavedModelInference의 CH3·평균 제거·Hann FFT·log1p·Train 정규화·8구간 복원 MSE 평균·저장 임계값을 사용한다.
- RF는 기존 analyze_with_engine을 호출한다. AE 정상 시 RF 생략, AE 이상 시 실제 원시 예측과 기존 보류 정책 결과를 구분한다. 미보정 모델 점수와 실제 사유·버전을 표시한다.
- OCSVM/IF는 기존 RPM별 번들과 score_samples의 부호 변환을 그대로 사용한다. -score_samples 8구간 평균 > 저장 임계값이며 AE 점수와 직접 비교하거나 합산하지 않는다.
- DAE는 기존 전용 Welch 전처리/파일 MSE 경로와 bundle_status/predict_dae를 재사용한다. 지원 RPM 미확인 또는 미지원이면 실행하지 않는다. 가중치 존재나 계산 호환성만으로 RPM 지원을 선언하지 않는다.
- 파일 없음/추론 오류/실행 차단/정상·이상을 구분하고 없는 점수에는 대시를 표시한다. 참고 모델 오류가 성공한 AE/RF 판정을 변경하지 않는다.

## 선택 모델 파일 검증

공통 SavedModelInference에 `selected_artifacts_only=False` 옵션을 추가했다. 기본값에서는 기존대로 모든 manifest 아티팩트를 검사하므로 V1/V2 동작이 유지된다. **V3만 True**를 전달해 선택 모델 및 공통 정규화 파일의 경로·해시를 검사한다. manifest 자체 해시, 전처리 계약, 해당 모델 버전·임계값·가중치 검증은 유지한다. 다른 참고 모델 파일 누락이 AE를 막지 않도록 한 검증 범위 옵션이며 점수 계산 변경이나 manifest 수정이 아니다.

## 화면 및 입력

네이비 사이드바/헤더·밝은 배경·흰 카드 디자인을 재사용한다. 홈/업로드/이력/통계/모델 성능 비교/설정의 6개 메뉴는 실제 페이지로 연결된다. 로그인 기능은 없으며 관리자 계정을 가짜로 표시하지 않는다. 상태는 분석 대기/완료/실패이며 모든 모델·DB가 정상이라고 일괄 선언하지 않는다. 시각은 실제 분석 timestamp의 KST 표시다.

65,536샘플, 25,600Hz, CH3 입력 계약을 유지한다. 파일/RPM/번들 변경 시 이전 결과를 초기화한다. 같은 성공 입력은 분석 버튼을 비활성화해 중복 실행을 막는다. 파일/RPM을 변경하면 새 분석이 가능하다. AE 실패 시 오류가 표시되며 추가 모델에는 임의 결과를 채우지 않는다.

실제 특징 표, 시간 파형/FFT 및 기존 전체·저주파 선택을 유지한다. 시간 파형 확대용 드래그/휠 그래프를 추가했다. 원본 FFT에서 기존 peak_component로 명목 RPM/60 × 1–5차의 실제 진폭을 측정한다. ±1Hz 탐색에 bin 해상도를 고려하며 실제 회전수 계측값이나 결함 판정선이 아니다. 1200RPM의 1X=20Hz다. 추가 고조파는 RF 학습 특징이나 보류 정책에 투입하지 않는다. 센서 단위는 raw로만 표시한다.

## 저장·이력

기존 analyses/model_results/candidate_reports에는 운영 AE/RF를 그대로 저장한다. 추가 참고 결과는 `v3_reference_results(analysis_id, payload_json, sha256)` 보조 테이블에 저장한다. 테이블은 명시적 저장 버튼에서만 생성되며 과거 기록을 삭제하거나 마이그레이션하지 않는다. 기존 V2는 이 테이블을 무시하고 V3는 없는 경우 참고 결과 없음으로 표시한다. 같은 결과 중복 저장은 기존 ID를 사용하고 다른 참고 결과로 덮어쓰지 않는다. 입력/점수/버전/실행 상태를 JSON으로 보존한다.

AE/RF 저장과 참고 결과 저장은 두 트랜잭션이다. 두 번째 저장이 실패하면 AE 기록만 저장됐을 수 있다고 화면에 명시하며 재시도할 수 있다. 운영 DB에서 시연하기 전 서버 종료 및 SQLite backup 절차는 V2_INTEGRATION_VALIDATION.md를 따른다. Codex에서는 임시 DB만 사용했다.

## 실행

기존 웹 전용 가상환경과 실제 AE/RF 번들 경로를 유지한다. 기존 미커밋 작업을 강제 덮어쓰지 않는다.

```bash
git fetch origin
git switch --track origin/feat/v3-multi-model-dashboard
# 브랜치가 이미 있으면 git switch feat/v3-multi-model-dashboard
export ISU_CHEMICAL_V2_MODEL=autoencoder
export ISU_CHEMICAL_CANDIDATE_LABELS=0
# 기존 ISU_CHEMICAL_V2_BUNDLE_ROOT, ISU_CHEMICAL_CANDIDATE_BUNDLE 설정 유지
# 선택: 검증된 DAE 번들만 ISU_CHEMICAL_DAE_BUNDLE로 지정 (미확인 RPM 차단 유지)
"$HOME/capstone-web-venv/bin/python" -B -m streamlit run streamlit_app_v3.py --server.port 8501
```

기본 DB는 기존 V2 설정을 재사용한다. V3 시연을 별도로 저장하려면 실행 전에 `export ISU_CHEMICAL_HISTORY_DB="$HOME/v3-demo/history.sqlite3"`를 지정한다. 이전 V2 기록과 함께 조회하려면 기존 DB를 백업한 뒤 그 경로를 유지한다. 앱 실행만으로 과거 기록을 삭제하지 않는다.

## 오프라인 성능 비교 페이지

`ISU_CHEMICAL_V3_EVALUATION_CSV`에 실제 평가 CSV 경로를 지정한다. 없으면 평가 자료 없음이다. 필수 열은 아래와 같고 모델 키는 autoencoder/dae/ocsvm/iforest다.

```text
model,rpm,date,n,split,metric,value
```

각 행에는 실제 평가의 RPM/날짜/파일 수/분할 방식/지표/값을 기록한다. 모델 버전·출처 등 추가 열도 그대로 표시한다. 파일 한 개의 운영 점수를 이 표에 넣거나 없는 DAE 평가를 만들어 넣지 않는다. 기존 평가 결과의 형식이 다르면 원본을 보존하고 이 열을 갖는 외부 CSV를 준비해야 한다. 이 페이지는 재평가·모델 학습·성능 보증 기능이 아니다.

## 검증 범위

합성 V3 신규 9개 및 기존 54개, 총 63개 테스트가 통과했다. RF.fit을 금지했다. AE 기본/V3 점수 일치, OCSVM/IF 실제 저장 추론 값 일치, 합성 DAE 준비/미확인/미지원, 선택 모델 해시 오류, 참고 파일 누락의 AE 격리, RF raw/보류 불변, 중복 실행/입력 초기화, 1–5X, 저장/중복 저장/새 Python 프로세스 재조회/구버전 이력을 검사했다.

Chromium 1920×1080과 1366×768에서 합성 MAT 업로드 → AE 이상/RF Bearing 원시 예측/최종 불명확 → 저장 → 이력을 확인했다. 가로 페이지 넘침과 브라우저 오류는 없었다. 세로 스크롤로 특징·그래프·하단 요약에 접근한다. 앞서 공유한 네이비·화이트 디자인을 기준으로 했으며 이번 요청에는 새 이미지 첨부가 없었다.

실제 사용자 MAT·학습 AE/RF/DAE 번들은 이 환경에 없어 실데이터 진단 성능이나 실제 DAE RPM 지원을 검증하지 않았다. 합성 화면 수치는 테스트용 모델에서 실제 계산된 것이며 하드코딩한 시연 결과가 아니다.

## 병합 전 WSL 점검

1. 실제 900/1200/1800RPM 파일에서 기존 V2와 AE 점수·임계값·판정 일치 확인
2. 정상 RF 생략 및 이상 RF 원시·보류·사유 불변 확인
3. OCSVM/IF 실제 번들 점수와 기존 웹/오프라인 값 비교
4. DAE 없음/미확인 RPM 차단 확인; 실제 지원 근거 있는 번들에서만 실행 확인
5. 원본을 손상시키지 말고 복사 번들로 모델 누락/손상과 AE 유지 확인
6. 파일/RPM 변경 및 같은 입력 중복 클릭 방지 확인
7. 1X–5X 실제 회전 주파수와 그래프 단위 확인
8. DB 백업 후 V3 저장/재시작/조회 및 V2 과거 기록 호환 확인
9. 두 화면 크기에서 표·보류 사유·하단 저장 접근 확인
10. 추가 모델 비교를 운영 판정/신뢰도로 오인하지 않는지 확인
