# Streamlit V2 대시보드 리디자인

## 화면 구성

제공된 참고 이미지의 네이비 사이드바·헤더, 밝은 회백색 배경, 흰색 카드와 파란색 강조색을 적용했다. 가로 입력 카드 → 3열 분석 결과 → 3열 실제 그래프 → 하단 4개 요약과 저장 패널 순서다. 좁은 화면에서는 카드가 세로로 배치되고 상단 `☰ 메뉴`에서 페이지를 이동할 수 있다.

참고 이미지에서 다음 내용은 실제 구현에 맞게 수정했다.

- 로그인·admin 계정은 표시하지 않는다.
- 시스템 표시는 선택 RPM의 AE **파일 존재** 상태이며 하드웨어 수집 연결이나 성능 검증 완료 표시가 아니다. 버전·SHA256은 분석 시 기존 로더가 검사한다.
- 시각은 실제 마지막 분석 시각을 KST로 표시한다. 예시 날짜·시각은 사용하지 않는다.
- AE 정상은 기존 정책에 따라 후보 엔진을 생략한다. 정상인데 특정 결함을 확정하는 예시를 재현하지 않는다.
- 후보 점수는 실제 엔진 출력이 있을 때만 보여주며 미보정 점수임을 명시한다. 가짜 72.3%나 신뢰도를 넣지 않는다.
- 엔진이 없으면 실제 1X·2X·고주파 관측 특징과 보류 사유를 표시한다.
- CH3만 분석하며 CH1/CH2 독립 분석을 주장하지 않는다.
- mm/s·m/s², 검증되지 않은 정상 범위/정상·주의 열, 미계산 3X~5X는 표시하지 않는다.
- 회전 성분은 900RPM=15Hz, 1200RPM=20Hz, 1800RPM=30Hz의 명목 1X 기준이다. 1200RPM에서 600Hz로 표시하지 않는다.

## 실제 동작하는 메뉴

| 메뉴 | 기능 |
| --- | --- |
| 홈 대시보드 | 업로드·RPM·분석·후보·특징·그래프·저장 |
| 데이터 업로드 | 같은 실제 업로드·입력 검증·분석 흐름에 접근 |
| 분석 이력 | 기존 DB 날짜/RPM/판정 검색, 상세, CSV 및 후보 JSON |
| 통계 및 리포트 | 저장 분석 ID별 정상/이상 포함 건수, RPM 집계, 실제 기록 CSV |
| 시스템 설정 | 현재 AE 경로/파일 상태, 후보 엔진 경로/표시 설정, DB 경로, CH3 입력 계약 |

통계는 실제 저장 이력의 기술 집계이며 모델 정확도·설비 가동률·실시간 감시 수치가 아니다. 설정은 읽기 전용이며 모델 선택·임계값 수정·관리자 계정 기능을 추가하지 않았다.

## 구조와 보존 범위

- `streamlit_app_v2.py`: 기존 입력 identity/초기화, AE·후보 API 호출, 결과 저장을 조합한다.
- `vibration/dashboard_view.py`: CSS, 헤더, 상태 원, 결과/특징 카드, 실제 그래프, 요약 표시만 담당한다.
- `vibration/dashboard_pages.py`: 기존 API로 이력·통계·설정을 렌더링한다.
- `tests/test_dashboard_redesign.py`: 메뉴·상세 그래프·시각화 축약·실제 특징/단위·회전 주파수·HTML escape 검증.

추론·후보 정책·모델 아티팩트·학습·정규화·임계값·DB 스키마는 변경하지 않는다. `candidate_analysis.py`, `candidate_engine.py`, `candidate_history.py`, `history.py`의 로직은 그대로다. 기존 `candidate_ui.py`의 전체 근거 화면도 상세/이력에서 재사용한다. 새 UI는 후보 정책을 우회하거나 분류 성능을 올리지 않는다.

CSS는 앱에서 지정한 container key와 제한적인 Streamlit data-testid를 사용한다. 생성되는 emotion 클래스 이름에는 의존하지 않는다. 기능은 네이티브 Streamlit uploader·button·radio·popover·download를 사용하며 HTML 카드는 읽기 전용 표시다. 파일명 등 사용자 문자열은 HTML escape한다. Streamlit 1.67.0으로 검사했다.

## 그래프와 상세 보기

- 시간 파형: 실제 65536샘플에서 화면용 구간 최솟값·최댓값 샘플을 선택한다. 실제 시각·진폭과 첨두를 보존하며 분석 입력을 바꾸지 않는다.
- FFT: 기존 전체 FFT 값 중 표시 bin별 최대값을 골라 피크를 보존한다. 전체 0–10kHz/저주파 0–200Hz 전환과 명목 1X·2X 선을 제공한다.
- 회전 성분: 기존 추출기의 실제 1X·2X 진폭만 표시한다. 새로운 3X~5X 계산은 하지 않는다.
- 하단 `상세 분석 · 원본 그래프와 전체 판단 근거`를 열고 `상세 그래프와 근거 표시`를 켜면 전체 원본 파형, 축약하지 않은 FFT, 기존 8구간 AE log1p 스펙트럼, 구간 점수와 후보 전체 근거를 볼 수 있다.

개요 그래프의 표시용 축약은 MSE·특징량·분류·임계값 계산에 사용하지 않는다. 결측값은 그래프나 표에서 임의의 정상값으로 바꾸지 않는다.

## WSL 실행

기존 서버를 Ctrl+C로 종료한 후 **실제 소스 Git 저장소**에서 실행한다. 데이터 폴더와 Git 저장소가 같은 경로라고 가정하지 않는다.

```bash
git rev-parse --show-toplevel
git fetch origin
git switch --track origin/feat/dashboard-v2-redesign

export ISU_CHEMICAL_V2_MODEL=autoencoder
"$HOME/capstone-web-venv/bin/python" -B -m streamlit run streamlit_app_v2.py \
  --server.address 127.0.0.1 --server.port 8501
```

이미 로컬 브랜치가 있으면 `git switch feat/dashboard-v2-redesign`을 사용한다. 로컬 변경을 강제 초기화하지 않는다. 최신 main으로 병합된 뒤에는 업데이트한 main에서 실행할 수 있다.

`http://localhost:8501`에서 확인한다. 기존 AE 번들·DB 환경 설정을 유지한다. 현재 후보 엔진을 쓰고 있었다면 `ISU_CHEMICAL_CANDIDATE_BUNDLE`, `ISU_CHEMICAL_CANDIDATE_LABELS`도 기존 검토한 값을 유지한다. UI 리디자인을 위해 후보명 표시를 강제 활성화하지 않는다. 후보 학습·평가·활성화는 [CANDIDATE_ENGINE.md](CANDIDATE_ENGINE.md)를 따른다. 모델이 없으면 불명확과 실제 오류/보류 사유가 표시된다.

## 시연 검증

1. 실제 MAT와 측정 당시 RPM으로 분석한다. 이전 AE 점수·임계값·판정과 같아야 한다.
2. 정상은 후보 생략, 이상은 연결된 실제 엔진의 후보 또는 보류를 표시해야 한다.
3. 주요 특징값, 전체/저주파 FFT, 회전 성분의 주파수, 상세 전체 그래프를 확인한다.
4. 저장하고 이력 버튼으로 이동해 같은 AE·후보·근거를 조회한다. 반복 저장·과거 AE-only/v1/v2 기록 호환을 확인한다.
5. 통계/리포트가 저장 기록을 집계하며 CSV가 내려받아지는지 확인한다. 설정의 실제 경로도 확인한다.
6. 1920×1080, 1366×768, 좁은 화면에서 잘림·겹침·가로 넘침을 확인한다. 좁은 화면에서는 상단 `☰ 메뉴`로 이동한다.
7. 실제 Normal1/2, IR/OR/Roller, Unbalance_-1/-2, Looseness 조건별 파일과 다른 날짜/RPM을 검사한다. 파일 선택과 DB 백업 절차는 [PRESENTATION_V2.md](PRESENTATION_V2.md)를 따른다.

## 테스트 범위

```bash
MPLCONFIGDIR=/tmp/dashboard-redesign-mpl \
"$HOME/capstone-web-venv/bin/python" -B -m unittest \
  tests.test_dashboard_redesign tests.test_candidate_engine \
  tests.test_presentation_v2 tests.test_dashboard_v2 tests.test_ae_v2_parity \
  tests.test_history.HistoryTests.test_group_filters_detail_csv_and_repeated_save \
  tests.test_history.HistoryTests.test_atomic_rollback_and_concurrent_duplicate \
  tests.test_history.HistoryTests.test_separate_process_restart_preserves_and_deduplicates -v
```

실제 사용자 MAT·학습 가중치는 Codex에 없다. 임시 합성 MAT·가중치·후보 모델·DB를 사용한 기능 테스트이며 실데이터 성능 검증 완료가 아니다. 기존 실제 MAT 의존 V1 이력 UI 테스트 1개는 이 환경에서 실행할 수 없다. 스크린샷의 SYNTHETIC 파일과 수치를 실제 진단 성능 자료로 사용하지 않는다.

2026-11-14(KST) 검증: 신규 6개와 기존 33개, 총 **39개 자동 테스트 통과**. Chromium에서 1920×1080, 1366×768의 3열 배치와 390×844의 세로 배치·모바일 메뉴를 확인했다. 세 크기 모두 실제 차트 4개가 렌더링됐고 가로 넘침과 page/console error는 0개였다. 저장·이력·통계 CSV·설정 이동도 검사했다. 스크린샷과 `browser-checks.json`은 완료 보고에 별도 첨부한다.
