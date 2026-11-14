# Looseness 전체 파일 날짜 차이 조사

## 범위와 검증 상태

신규 scripts/analyze_looseness_date_shift.py는 20260912/20260916의 세 RPM·세 Looseness 조건을 전수 탐색한다. 907개 또는 기존 표본 관측값을 정답으로 하드코딩하지 않는다. 실제 WSL MAT는 Codex에 없어 실제 파일 수·탐지율·날짜 비율을 새로 측정하지 못했다. 합성 테스트와 실제 결과는 data_kind로 구분한다.

AE는 V2가 사용하는 SavedModelInference(models=('autoencoder',))를 그대로 사용하고, CPU에서 RPM당 한 번 로드해 재사용한다. 기존 번들/버전/해시 검증을 유지한다. 전처리·정규화·복원 MSE·8구간 평균·저장 임계값을 바꾸지 않는다. 특징은 기존 candidate_analysis의 계산 부분만 사용하며 analyze_with_engine/RF 예측·학습·운영 DB를 호출하지 않는다. V2/V3 웹과 운영 정책은 수정하지 않는다.

## 1. WSL 파일 수만 먼저 확인

```bash
cd "/mnt/c/Users/kungh/OneDrive/바탕 화면/isu_chemical"
git fetch origin
git switch --track origin/feat/looseness-date-shift-analysis
# 브랜치가 이미 있으면 git switch feat/looseness-date-shift-analysis
web_python="$HOME/capstone-web-venv/bin/python"
stamp="$(date -u +%Y%m%dT%H%M%S%NZ)"
"$web_python" -B scripts/analyze_looseness_date_shift.py \
  --data-root "$PWD/data" --data-kind actual --inventory-only \
  --output "$HOME/v2-error-analysis/looseness-inventory-$stamp"
```

inventory.csv는 날짜×RPM×Looseness/Normal1/Normal2별 발견 수다. 이 단계는 MAT를 읽지 않아 readable/read_failed는 not_tested다. 실제 읽기 성공/실패 수는 본 분석 group_summary.csv에서 확인한다. discovery_exclusions.csv에는 날짜·RPM·조건·Vib·파일 날짜가 모호하거나 충돌하는 경로를 기록한다. 파일명에 문자열이 들어간다고 다른 폴더의 파일을 포함하지 않는다. 심볼릭 링크/루트 밖 파일은 제외한다. 실제 데이터 루트가 없으면 에러이며 실제 파일 수 0으로 보고하지 않는다.

기본 분석은 Normal을 실행하지 않는다. Normal 발견 수를 먼저 표시하고 Looseness 실측 처리시간을 얻은 뒤 analysis_report.json의 normal_seconds_estimate를 확인한다. 이것은 동일 길이 신호의 대략적 CPU 추정이며 장비·캐시·오류에 따라 달라진다.

## 2. 전체 Looseness + 기존 표본 수치 비교

```bash
loose_output="$HOME/v2-error-analysis/looseness-full-$(date -u +%Y%m%dT%H%M%S%NZ)"
analysis_args=(
  --data-root "$PWD/data"
  --bundles "$PWD/outputs/inference_bundles/date_prelim_20261106"
  --data-kind actual
  --reference "$HOME/v2-error-analysis/20261114T063407214292556Z-sample36/integration_results.csv"
  --reference "$HOME/v2-error-analysis/normal_sample_36_run/integration_results.csv"
)
"$web_python" -B scripts/analyze_looseness_date_shift.py "${analysis_args[@]}" --output "$loose_output"
```

기본값에서는 Normal을 분석하지 않으므로 Normal 참조 행은 비교 범위에서 제외한다. 참조 파일 자체의 해시는 기록한다. 참조 CSV가 없으면 해당 경로를 확인해야 하며 일치 검증을 수행했다고 주장하지 않는다. --reference 없이 실행은 가능하지만 parity_verified/conclusions_ready는 false다.

같은 실제 경로·날짜·RPM·조건·MAT SHA256·AE 점수·저장 임계값·판정·9개 특징을 비교한다. 고정 rtol=1e-10/atol=1e-12이며 판정은 정확 일치해야 한다. 같은 환경/입력의 재사용 경로를 기대하는 엄격한 기준이다. 불일치 시 모델 해시·패키지·MAT SHA·전처리 출처를 조사하고 오차를 Test 결과에 맞춰 늘리지 않는다. reference_parity의 이유와 differences를 확인한다. CSV 숫자 반올림 등으로도 불일치가 날 수 있으나 근거 없이 허용하지 않는다.

종료 코드 0은 완료·표본 비교 통과·두 날짜×세 RPM×세 강도 모두 유효 표본 존재·대상 탐색 제외·처리 오류/중복/원본 변경 없음일 때만 나온다. 누락/불일치/오류가 있으면 종료 코드 1이며 숫자 관측 자료는 남지만 전체 결론은 보류한다. 이 조건이 통과해도 독립 Test 성능이나 물리적 원인이 확정되는 것은 아니다.

## 3. 중단 및 재개

```bash
"$web_python" -B scripts/analyze_looseness_date_shift.py "${analysis_args[@]}" \
  --output "$loose_output" --resume
```

같은 셸의 analysis_args/loose_output을 유지한다. 다른 셸이면 이전 정확한 인자·출력 경로를 다시 설정한다. 재개 없이 기존 출력 폴더를 지정하면 거부한다. 번들/참조/구현 코드/패키지 버전/선택 목록/이미 처리한 MAT가 달라져도 거부한다. --include-normal을 재개 도중 추가하지 않는다.

파일별 progress.jsonl에 flush/fsync하며 Ctrl+C에서는 부분 CSV/JSON/요약도 남긴다. 강제 종료 시에는 완료된 journal 행이 재개 기준이다. journal이 중간에 잘렸거나 손상되면 보수적으로 거부하고 원본을 삭제/자동 편집하지 않는다. 실패한 파일도 처리 기록으로 유지하며 자동 재시도하지 않는다. 오류 원인을 조사한 후 새 출력 폴더에서 다시 실행한다. 현재 실행의 CSV/보고서는 재개 결과로 갱신하지만 이전 별도 실험 폴더를 덮어쓰지 않는다.

메모리에 모든 MAT를 적재하지 않고 한 파일씩 처리한다. 파일별 SHA256을 저장하고 같은 콘텐츠는 duplicate_content로 표시해 이후 복사본을 성공 분모에서 제외한다. 읽기 성공은 MAT 로더의 구조 검증 통과이며, AE/특징 계산 실패와 구분한다.

## 4. Normal 대조군 (명시적 선택)

먼저 inventory와 위 Looseness 실행의 파일 수/예상 시간을 검토한다. 필요하면 별도의 새 실행에만 --include-normal을 지정한다. 이 모드는 Looseness도 다시 계산하므로 시간을 고려한다.

```bash
control_output="$HOME/v2-error-analysis/looseness-normal-full-$(date -u +%Y%m%dT%H%M%S%NZ)"
"$web_python" -B scripts/analyze_looseness_date_shift.py "${analysis_args[@]}" \
  --include-normal --output "$control_output"
```

Normal 진입 전 파일 수와 현재 Looseness 처리시간 기반 추정을 출력한다. Normal1/Normal2는 구분한다. normal_control_comparison.csv에는 같은 RPM에서 각 Normal 대조군과 Looseness의 날짜별 중앙값 차이 및 날짜 차이의 차이를 기록한다. 이는 날짜 변화의 결함 특이성을 조사할 단서이지 인과 효과 추정이 아니다. Normal의 학습/검증/threshold 산출 중복 가능성이 있으므로 독립 정상 FPR·일반화 성능으로 주장하지 않는다.

## 산출물

- file_results.csv: 경로·SHA·상태/오류·날짜/RPM/조건, AE 버전/번들 ID·점수·threshold·비율·판정, 9개 특징, 표본 비교·시간, AE 8구간 점수
- group_summary.csv: 전체/처리/성공/실패/중복/읽기 수, AE 이상 수·비율, 점수·비율·9개 특징의 n/mean/median/min/max/q05/q25/q75/q95
- date_comparison.csv: 같은 RPM/강도의 두 날짜 값·차이·평균/중앙값 비율
- analysis_report.json: 완료 상태, data_kind, Git·런타임·구현/번들 해시, 입력/그룹 수, 표본 일치/누락/불일치, 커버리지, 결론 사용 여부, 시간·Normal 추정
- analysis_summary.md: 한국어 결과/한계·날짜별 실제 관측 표·paired 방향 집계·다음 우선순위
- inventory.csv/discovery_exclusions.csv: 전체 경로 탐색 및 제외 기록
- segment_summary.csv: 기존 AE 8구간(각 0.32초)의 점수 분포; 구간 threshold를 파일 threshold로 대체하지 않음
- normal_control_comparison.csv: 명시적 Normal 분석 시 날짜 차이 대조 자료
- progress.jsonl/run_config.json: 중단·재개 감사 자료
- PNG: AE 이상 비율, 날짜/조건별 점수 비율·RMS·Peak·1X·2X·중고주파 분포, 파일명 측정 시간에 따른 점수

임계값이 0 또는 1e-12 이하면 score/threshold는 공란이다. 날짜 비율의 분모 절댓값≤1e-12도 공란이고 원래 값·차이는 유지한다. PNG는 symlog를 사용해 0 주변을 선형으로 처리하며 0/음수를 제거하지 않는다. 각 집단 n을 표시한다. 점수/threshold가 정의되지 않는 집단의 비율 그래프는 생성되지 않을 수 있다. 분위수는 기술 통계이며 새 운영 threshold를 산출하거나 선택하지 않는다.

## 특징 정의

CH3 전체 65,536샘플, 25,600Hz, 시간 평균 제거 AC RMS/Peak/Crest Factor. 센서 물리 단위 미확인으로 raw·raw²·무차원만 사용한다. 대칭 Hann 단측 FFT의 명목 RPM/60 및 2배 회전 주파수 ±1Hz 피크이며 bin 해상도는 25600/65536=0.390625Hz다. 대역 에너지는 기존 periodogram 적분으로 [0,200), [200,2000), [2000,10000]Hz다. 엔벨로프는 기존 4차 Butterworth 2–8kHz, sosfiltfilt, Hilbert와 양쪽 0.1초 제외를 유지한다.

이 9개 특징은 RF용 관측 지표와 같은 정의지만 AE의 직접 입력은 아니다. AE는 기존 8구간 FFT log1p/Train 정규화 스펙트럼을 사용한다. RMS 차이 하나로 AE 복원오차의 원인을 확정하지 않는다.

## 해석 및 제한

기존 사용자 표본의 날짜별 탐지 차이와 특징 비율은 배경 관측이며 이 스크립트의 결론으로 미리 입력하지 않는다. 전체 결과가 다르면 가설을 수정한다. 0이 아닌 표본이 있어도 다른 날짜/Run과 독립이라는 뜻은 아니다. 측정 시각은 파일명의 단서이고 Run ID가 아니다. 강도·RPM별 표본 수가 다르면 날짜별 단순 합계보다 층별 date_comparison을 우선한다.

추가 확인할 자료: 실제 체결 간극/토크, 부하·실측 RPM, 센서 위치/방향·감도/교정, 수집 게인/단위, 실험 순서·Run/라벨 로그, AE 학습/검증 출처. 이 자료 없이 날짜 학습·라벨 오류·센서 이동·결함 강도 증가를 단정하지 않는다. AE 정확도 100%나 임계값 변경으로 해결됐다고 주장하지 않는다.

## 테스트 및 병합 전

신규 합성 테스트는 탐색/18개 층, metadata/SHA, V2 점수·특징 일치, 잘못된 MAT, 중복, 표본 불일치, 안전한 비율, Ctrl+C 재개/입력 변경 차단, Normal opt-in, 그림 및 구간 요약, RF/DB 호출 금지를 검사한다. 실제 36개와의 일치는 WSL에서만 확인할 수 있다.

- 원본 MAT·가중치·정규화·threshold 불변
- 운영 DB 접근/RF 실행 없음
- 실제 파일 수·제외/실패·중복 및 18개 층 확인
- 기존 Looseness 36개 및 선택 시 Normal 36개 parity 확인
- 원본/구현/환경 해시와 중단 재개 확인
- 분석 결과·한계를 함께 보고, 대용량 파일 Git 제외
- V2/V3 회귀 테스트 통과


Codex 검증: 신규 10개와 기존 84개, 총 94개 합성·회귀 테스트가 통과했다. 최종 집계/안전성 수정 후 신규 테스트를 재실행한다. 실제 WSL 데이터 및 기존 두 표본 CSV가 이 환경에 없어 실제 파일 수·날짜별 탐지율·기존 36개 일치 여부는 아직 미확인이다.
