# 11월 17일 발표용 Streamlit V2 통합

이 문서는 근거 전용 v1 통합 기록이다. 학습 가능한 후보 엔진 v2의 현재 학습·보류·활성화 절차는 [CANDIDATE_ENGINE.md](CANDIDATE_ENGINE.md)를 따른다. 기존 v1 이력은 유지된다.

최신 카드형 대시보드의 레이아웃·메뉴·실행 절차는 [DASHBOARD_V2_DESIGN.md](DASHBOARD_V2_DESIGN.md)를 참고한다.

## 현재 제공하는 기능

실제 MAT 업로드 → RPM 선택 → **기존 AE 이상 탐지**와 **별도의 진동 특징·결함 후보 검토** → 동일 화면 결과 → 선택 저장 → 이력 재조회 흐름을 제공한다.

- AE는 기존 `dashboard_v2.analyze`와 모델 번들을 그대로 사용한다. CH3, 8구간 FFT/log1p, Train 정규화, 복원 MSE, 파일 평균, 저장 임계값을 수정하지 않았다.
- V2 신규 분석은 AE만 허용한다. 모델 선택 UI는 없고 DAE·OCSVM·IF를 호출하지 않는다. 기존 V1·오프라인 API와 과거 다중 모델 이력은 유지한다.
- 후보 특징은 기존 EDA `analyze_signal`을 재사용한다. CH3, 25,600Hz, 65536샘플, 전체 기록 Hann FFT와 1X·2X, RMS·Peak·Crest Factor, 대역 평균제곱, 엔벨로프 AC RMS를 계산한다.
- 시간 파형, 후보 검토용 전체 FFT(0–10kHz/0–200Hz), 기존 AE의 8구간 log1p 입력 스펙트럼을 구분해 표시한다. FFT의 명목 1X·2X 선은 진단 임계값이 아니다.
- 후보 검토 계산이 실패해도 성공한 AE 결과는 유지한다. 후보는 계산 불가 사유를 표시하고 해당 상태로 저장할 수 있다.

## 결함 후보가 불명확인 이유

현재 웹에서 사용할 **검증·등록된 결함 분류 아티팩트가 없다**. 이전 RF 평가 도구는 실행 중 학습·평가하는 도구이며 웹용 학습 모델을 배포한 것이 아니다. Looseness Recall도 매우 낮았다. 따라서 이번 구현은 Random Forest가 존재한다고 가정하거나, 임의의 1X/고주파 임계값으로 결함명을 생성하지 않는다.

정상적인 특징 계산 후 최종 후보는 **‘결함 후보 불명확’**이다. 이는 시연 파일명을 보고 고정한 정답이 아니라, 결함 종류를 결정할 검증된 기준이 없다는 명시적 보류 정책이다. 파일명·폴더·정답 조건은 후보 분석 함수에 전달하지 않는다. 실제 데이터에서 계산한 관측값과 다음 검토 근거를 함께 보여준다.

| 검토 대상 | 관측 특징 | 결론을 보류하는 이유 |
| --- | --- | --- |
| 베어링 | 고주파 평균제곱, 엔벨로프 RMS, Crest Factor | 정상 대비 증가·형상·결함 주파수·센서 대역 적합성 미확인 |
| 불평형 | 명목 1X·2X 진폭, RMS | 회전 성분의 존재만으로 불평형 확정 불가; 정상 기준·부하·방향 필요 |
| 느슨함 | 2X 진폭, Crest Factor, Peak | 기존 일반화 성능이 낮아 단순 고조파로 판정하지 않음 |

세 검토 항목은 후보 순위나 확정 진단이 아니다. AE 이상은 특정 결함을, AE 정상은 결함 부재를 보장하지 않는다. 엔벨로프 2–8kHz는 기존 탐색 설정이며 특정 베어링 결함 주파수로 해석하지 않는다. 3X~5X 개선 실험, 분류기 재학습·추가 모델 비교는 이번 경로에 포함하지 않는다.

## DB 호환성과 저장

기존 `analyses`, `model_results`, 검색 필드, request_key·중복 방지 규칙은 그대로다. `HistoryStore.save`에 선택적 `candidate_report` 인자를 추가했으며, 기존 호출은 종전과 동일하게 동작한다.

새 V2에서 저장하면 같은 SQLite DB에 **보조 `candidate_reports` 테이블**을 최초 저장 시 생성한다. 기존 테이블 ALTER나 기존 기록 갱신·삭제는 하지 않는다. 후보 특징·근거·제한·버전·입력 SHA256·RPM을 JSON으로 저장한다. 원본 MAT·파형·전체 FFT 배열은 DB에 저장하지 않는다.

- AE 행과 후보 행은 같은 트랜잭션으로 저장한다. 후보 삽입 실패 시 새 AE 행도 rollback한다.
- 같은 분석·같은 후보 버전의 다른 payload로 기존 후보를 덮어쓰지 않는다.
- 기존 AE-only 기록에 동일 입력의 후보를 보충할 수 있으나 기존 AE 행은 바꾸지 않는다.
- 과거 기록은 후보가 없어도 정상 조회된다. 후보 읽기는 read-only 연결이며 테이블을 만들지 않는다.
- 후보가 없는 기록은 ‘이 기록에는 결함 후보 근거가 없습니다’로 표시한다.
- 기본 CSV 다운로드의 기존 모델 이력 형식은 유지한다. 새 후보 근거는 상세 화면 및 후보 JSON 다운로드에서 조회한다.

**운영 DB 첫 사용 전 별도 백업을 권장한다.** 실행 중 DB 파일을 단순 복사하기보다 SQLite backup을 사용한다. 기존 환경변수 `ISU_CHEMICAL_HISTORY_DB`가 설정돼 있으면 그 경로를 그대로 사용한다.

## WSL 실행

먼저 기존 Streamlit을 실행하던 **실제 소스 Git 저장소**에서 진행한다. 데이터 폴더가 Git 저장소라고 가정하지 않는다.

기존 웹 서버는 실행 터미널에서 `Ctrl+C`로 종료한 후 브랜치를 전환하고 새 프로세스로 시작한다. 내부 모듈·DB API 변경이 있어 기존 프로세스의 자동 새로고침에만 의존하지 않는다.

```bash
git rev-parse --show-toplevel
git fetch origin
git switch --track origin/feat/presentation-v2-candidate-ui
```

이미 로컬 브랜치가 있으면 `git switch feat/presentation-v2-candidate-ui`를 사용한다. 로컬 변경이 있으면 강제 전환·초기화를 하지 않는다. main에 병합된 뒤에는 최신 main에서도 실행할 수 있다.

기존 웹 전용 환경과 기존 번들·DB 설정을 사용한다. 새로운 가상환경 설치나 기존 `~/capstone/.venv` 변경은 필요 없다.

```bash
export ISU_CHEMICAL_V2_MODEL=autoencoder

# 실제 적용되는 번들과 DB 경로 확인
"$HOME/capstone-web-venv/bin/python" -B - <<'PY'
from vibration.dashboard_v2 import configuration, history_store
c = configuration()
print("model:", c.model)
print("bundles:", c.bundle_root)
print("history:", history_store().path)
PY

"$HOME/capstone-web-venv/bin/python" -B -m streamlit run streamlit_app_v2.py \
  --server.address 127.0.0.1 --server.port 8501
```

브라우저: `http://localhost:8501`. 포트가 사용 중이면 기존 서버를 종료하거나 다른 포트를 명시한다. 900·1200·1800RPM 번들의 실제 가중치·정규화·manifest 및 버전 검사에 필요한 파일은 기존 위치에 있어야 한다. 이 작업은 모델 파일을 Git에 추가하지 않는다.

DB 백업이 필요하면 웹 서버 실행 전 소스 저장소에서 아래 명령을 사용한다. 새 타임스탬프 파일에만 저장한다.

```bash
"$HOME/capstone-web-venv/bin/python" -B - <<'PY'
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from vibration.dashboard_v2 import history_store
source = history_store().path.resolve()
if source.is_file():
    folder = Path.home() / "capstone-db-backups"
    folder.mkdir(exist_ok=True)
    target = folder / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".sqlite3")
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as src:
        with closing(sqlite3.connect(target)) as dst:
            src.backup(dst)
    print("backup:", target)
else:
    print("기존 DB 없음:", source)
PY
```

## 실제 MAT 시연 점검 절차

### 1. 실제 파일 목록 확인

기존 EDA CSV에서 각 RPM·원래 조건별 파일 한 개와 실제 존재 여부를 출력한다. 이 선택은 테스트 준비용이며 웹은 이 레이블을 사용하지 않는다.

```bash
"$HOME/capstone-web-venv/bin/python" -B - <<'PY'
import csv
from pathlib import Path
p = Path('/home/kungh/fault-signatures/20261113T131205447206Z/file_features.csv')
with p.open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
conditions = ['Normal1', 'Normal2', 'IR', 'OR', 'Roller', 'Unbalance_-1',
              'Unbalance_-2', 'Looseness0.05', 'Looseness0.1', 'LoosenessFull']
for rpm in (900, 1200, 1800):
    for condition in conditions:
        matches = sorted((r for r in rows if int(r['rpm']) == rpm and r['condition'] == condition),
                         key=lambda r: (r['date'], r['path']))
        if matches:
            r = matches[0]
            print(rpm, condition, r['date'], 'exists=' + str(Path(r['path']).is_file()), r['path'])
        else:
            print(rpm, condition, 'CSV에 파일 없음')
PY
```

### 2. 분석 화면

1. Normal1/2, IR, OR, Roller, Unbalance, Looseness 각 실제 파일을 차례로 업로드한다. **측정 당시 RPM**을 선택한다.
2. AE 판정·점수·임계값이 기존 오프라인/웹 결과와 같은지 비교한다. 이미 확인한 900RPM `20260912_144039_Vib.mat`는 약 48.3072 / 0.324214 / 이상을 참조하되, 값을 코드에 넣어 맞추지 않는다. 다른 번들·파일·RPM이면 동일 결과를 요구하지 않는다.
3. 같은 화면에 별도의 ‘결함 후보 불명확’과 실제 1X·2X·고주파 관측값이 나타나는지 확인한다. ‘실제 진동 특징량 9개와 측정 조건’ 및 ‘후보별 관측 근거와 점검 한계’를 연다.
4. 시간 파형, 전체 FFT, 저주파 확대와 1X·2X 표시, AE 입력 스펙트럼을 확인한다. 서로 다른 FFT 집계가 같은 값이어야 한다고 오해하지 않는다.
5. 이름만 IR/OR/Normal로 바꾼 같은 바이트의 파일도 후보 특징·상태가 같아야 한다. 이름만 보고 결함을 출력하면 실패다.
6. 다른 파일이나 RPM으로 바꾸면 이전 AE·후보·FFT가 지워지고 재분석을 요구해야 한다.
7. AE 정상인 경우에도 후보 검토는 별개이며 ‘결함 없음’이라고 단정하지 않는다. IR/OR/Roller 파일에서 후보 불명확은 분류 근거 미확인 정책에 따른 정상 동작이다.

### 3. 저장·이력

1. 저장 전 기존 DB 백업을 확인한다.
2. ‘이 결과 저장’을 두 번 눌러 분석 건과 후보가 중복되지 않는지 확인한다.
3. ‘분석 이력’에서 날짜·RPM·판정으로 검색하고 새 기록의 AE 값과 후보 특징·근거가 동일한지 확인한다.
   이력 날짜는 MAT 측정 날짜가 아니라 **분석 실행 날짜(KST)**다.
4. 원본 MAT 업로드 없이 웹을 재시작해 이력 상세와 후보 JSON을 다시 조회한다.
5. 과거 AE-only·다중 모델 기록도 그대로 읽히고 후보 없음으로 표시되는지 확인한다.

### 4. 오류 상태

잘못된 MAT/NaN·Inf·잘못된 fs·길이는 입력 단계에서 차단돼야 한다. 번들 누락·해시 오류 시 AE 결과를 생성하지 않는다. 후보 특징 계산 실패 시 기존 AE는 보존되고 후보는 계산 불가로 표시된다. 저장 권한·디스크 문제 시 성공 메시지 없이 오류를 표시해야 한다. 실운영 번들을 수정해 오류를 시험하지 말고 별도 복사본/테스트 폴더를 사용한다.

## 자동 테스트·검증 제한

합성 MAT와 임시 가중치/DB로 실제 V2 흐름과 기존 API를 검사한다. 테스트 파일명 Normal/IR/OR/Roller/Unbalance/Looseness는 레이블 누출 방지용이며 실제 해당 결함 신호를 뜻하지 않는다.

```bash
MPLCONFIGDIR=/tmp/presentation-mpl \
"$HOME/capstone-web-venv/bin/python" -B -m unittest \
  tests.test_presentation_v2 tests.test_dashboard_v2 tests.test_ae_v2_parity \
  tests.test_history.HistoryTests.test_group_filters_detail_csv_and_repeated_save \
  tests.test_history.HistoryTests.test_atomic_rollback_and_concurrent_duplicate \
  tests.test_history.HistoryTests.test_separate_process_restart_preserves_and_deduplicates -v
```

실제 프로젝트 MAT·학습 번들은 Codex에 없어 실데이터 추론 검증을 완료했다고 주장하지 않는다. 기존 `test_history`의 V1 웹 저장 테스트 1개는 실제 MAT를 찾는 단계에서 데이터 부재로 실행되지 않았다. 합성 V2 업로드·저장·이력, 기존 V1 추론 연결, DB 원자성·동시 중복 방지·재시작은 별도로 검사한다. 모델 채택이나 실제 결함 판별 성능 검증은 이번 테스트 범위가 아니다.

병합 전에는 AE 수치 유지, 기존 테이블·기록 호환, 보조 테이블 추가에 대한 운영 백업, 실제 조건별 MAT 시연을 확인한다. 기능 코드 병합과 실제 데이터 검증 완료를 구분한다.

클라우드 검증 결과: 신규 8개 + 기존 V2 6개 + 기존 DB 3개 + AE 경로 비교 6개, **23개 테스트 통과**. 화면 정리 후 UI 14개도 재실행해 통과했다. Python 3.14.4, NumPy 2.4.6, SciPy 1.18.1, PyTorch 2.14.1+cpu, scikit-learn 1.9.0, Streamlit 1.67.0을 사용했다. Chromium/Playwright에서도 합성 MAT 업로드→AE·후보 동시 표시→특징량·FFT→저장→이력 재조회를 확인했고 page error는 0개였다. 임시 랜덤 가중치·합성 신호를 사용했으며 실제 결함 판별 성능 검증이 아니다.
