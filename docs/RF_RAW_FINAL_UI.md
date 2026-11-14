# RF 원시 예측과 최종 후보 분리

V2의 대시보드, 상세 보기, 저장 이력은 다음 결과를 구분한다.

1. **AE 이상 탐지**: 기존 점수·임계값·판정을 그대로 사용한다.
2. **모델 참고 예측 / RF 원시 예측**: 실제 `engine.raw_prediction`만 표시한다. Bearing은 베어링 결함, Unbalance는 불평형, Looseness는 느슨함, Normal은 정상 클래스다. 확정 진단이 아니다.
3. **최종 결함 후보**: 기존 보류 정책 적용 결과를 그대로 표시한다. 실제 보류 사유와 관측 특징을 함께 확인한다.

`ISU_CHEMICAL_CANDIDATE_LABELS=0`은 기존 설계상 최종 자동 후보명만 비활성화한다. 원시 예측은 기존에도 저장되고 상세 정보에 노출되었다. 이번 변경은 이를 독립 컴포넌트로 강조하며 새 공개 설정이나 정책을 추가하지 않는다. 이 설정이 0이면 RF가 Bearing이어도 최종 후보는 불명확이다.

AE 정상에서는 RF를 호출하지 않고 실행 생략을 표시한다. AE 이상/RF Normal 충돌, 점수 차이 부족, 훈련 범위 이탈, RPM·클래스 검증 부족, 자동 후보명 비활성화, 모델 누락·호환 오류는 기존 엔진이 반환한 실제 사유만 표시한다. 점수를 신뢰도나 보정된 확률로 표현하지 않는다. 파일명·정답 레이블·최종 후보나 점수의 argmax로 원시 예측을 생성하지 않는다.

## 저장 및 호환성

기존 `candidate_reports.payload_json`은 `engine.raw_prediction`, `candidate`, `abstained`, `reasons`, 특징량을 이미 저장한다. DB 스키마·저장 경로·마이그레이션을 변경하지 않는다. 기존 schema 1/2 기록을 읽으며 원시 값이 없는 과거 기록에는 `원시 예측 없음`을 표시하고 재추론하지 않는다.

## 검증 범위

합성 데이터 기반 신규 9개 및 기존 회귀 29개, 총 38개 테스트 통과. RF.fit 호출을 금지한 상태에서 테스트했다. 이미 존재하던 합성 RF 아티팩트를 읽기 전용으로 로드한 추론, 아티팩트 해시 불변, 손상 복사본 처리, AE 정상 실행 생략, labels=0 보류 유지, DB 재조회, AE 결과 불변, 파일명 변경 불변을 확인했다. 실제 사용자 MAT와 학습 RF는 이 환경에 없어 실제 진단 성능을 검증하지 않았다.

Chromium에서 합성 MAT 업로드 → 분석 → 원시 Bearing/최종 불명확 분리 표시 → 저장 → 이력 재조회를 확인했다. 이는 기능 검증이며 실데이터 진단 성능 검증이 아니다.

사용자가 제공한 Test 61개 원시 정확도 29/61(약 47.5%), 후보 제시 6개, OR의 RPM별 3/3 Bearing 예측은 외부 실행 결과다. 이번 테스트의 측정치가 아니며 진단 성능이 충분하다는 근거가 아니다. IR·Roller·Normal·Unbalance 오분류와 표본 부족을 함께 설명해야 한다.

## WSL 실행

기존 Streamlit을 종료하고 Git 저장소에서 실행한다. 미커밋 작업이 있으면 보존하고 강제 checkout/reset을 하지 않는다.

```bash
git fetch origin
git switch --track origin/feat/rf-raw-final-separation
# 이미 로컬 브랜치가 있으면: git switch feat/rf-raw-final-separation
export ISU_CHEMICAL_V2_MODEL=autoencoder
export ISU_CHEMICAL_CANDIDATE_LABELS=0
# ISU_CHEMICAL_CANDIDATE_BUNDLE은 61개 평가에 사용했던 기존 번들 경로 유지
: "${ISU_CHEMICAL_CANDIDATE_BUNDLE:?기존 RF 번들 절대 경로를 먼저 설정하세요}"
"$HOME/capstone-web-venv/bin/python" -B -m streamlit run streamlit_app_v2.py --server.port 8501
```

번들 경로가 현재 셸에 없다면 기존 평가에 사용했던 경로를 확인해 `export ISU_CHEMICAL_CANDIDATE_BUNDLE='/실제/기존/번들/경로'`로 설정한다. 임의 최신 폴더를 선택하거나 모델을 다시 학습하지 않는다. 기존 AE 번들·DB 경로 설정도 유지한다.

## 실제 OR 파일 시연 확인

기존 번들의 Test 목록에서 OR 경로를 읽기 전용으로 확인한다. 이 목록의 조건명은 검증 파일 선택용이며 웹 추론 입력이 아니다.

```bash
"$HOME/capstone-web-venv/bin/python" -B - <<'PY'
import csv, os
from pathlib import Path
bundle = Path(os.environ['ISU_CHEMICAL_CANDIDATE_BUNDLE'])
with (bundle / 'test_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
    for row in csv.DictReader(f):
        if row['condition'] == 'OR':
            print({k: row[k] for k in ('rpm','date','path','raw_prediction','accepted','reasons')},
                  'MAT exists:', Path(row['path']).is_file())
PY
```

1. 동일 번들과 Test OR 파일을 사용하고 폴더 RPM에 맞춰 900/1200/1800을 선택한다.
2. 기존 오프라인/웹 AE 점수·임계값·판정이 그대로인지 확인한다.
3. AE 이상일 때만 실제 RF 원시 예측을 확인한다. 동일 조건에서 RF가 Bearing을 반환하면 `베어링 결함` 참고 예측이 표시된다. 결과가 다르면 그대로 기록하며 강제 변경하지 않는다.
4. labels=0에서는 최종 후보 불명확과 실제 보류 사유를 확인한다. 추가 사유 펼침도 확인한다. AE 정상일 때는 RF 생략이 올바른 동작이다.
5. 저장 후 이력에서 원시 예측·최종 후보·사유가 동일한지 확인하고, 앱 재시작 뒤 재조회한다. 기존 AE-only/구버전 기록도 확인한다.
6. Normal1/2, IR/Roller, Unbalance, Looseness 및 다른 날짜·RPM도 반복한다. OR 소수 사례를 전체 결함 성능으로 일반화하거나 Test를 보고 정책을 완화하지 않는다.
