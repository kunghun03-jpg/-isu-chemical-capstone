# 1800RPM Unbalance_-1 50개 파일 AE 예비 검사

현재 Codex에는 실제 WSL MAT 폴더가 없어 실제 50개를 실행하지 못했다. 아래 명령은 WSL에서 실제 파일을 전부 읽고 결과를 만든다. 파일 수 50은 기대값이며 발견 수를 실제로 세고 다르면 표시한다. 임의로 50개로 자르거나 점수/판정을 채우지 않는다.

```bash
cd "/mnt/c/Users/kungh/OneDrive/바탕 화면/isu_chemical"
git fetch origin
git switch --track origin/feat/unbalance-preliminary-analysis
# 로컬 브랜치가 있으면 git switch feat/unbalance-preliminary-analysis
"$HOME/capstone-web-venv/bin/python" -B scripts/analyze_unbalance_preliminary.py \
  --data-dir "$PWD/data/20260919/20260919_1800RPM/Unbalance_-1/Vib" \
  --bundles "$PWD/outputs/inference_bundles/date_prelim_20261106" \
  --data-kind actual \
  --output "$HOME/v3-error-analysis/unbalance-preliminary"
```

기존 출력 폴더는 덮어쓰지 않는다. 이미 결과가 있다면 --output에 `"$HOME/v3-error-analysis/unbalance-preliminary/$(date -u +%Y%m%dT%H%M%S%NZ)"`처럼 새 하위 폴더를 지정한다. 기존 미커밋 작업을 강제 checkout/reset으로 덮어쓰지 않는다.

기존 V3 analyze_selected(1800RPM, autoencoder)를 그대로 사용하며 선택 모델·정규화 해시/버전 검사와 전처리·8구간 MSE 평균·저장 threshold를 유지한다. 추론 입력 이름은 uploaded.mat이며 실제 파일명/폴더 레이블은 모델 입력으로 사용하지 않는다. RF·DB·Streamlit 앱은 호출/수정하지 않고 모델·MAT를 읽기만 한다.

## 결과

- ae_results_50.csv: 실제 발견한 모든 MAT의 파일명/경로/SHA256, AE 점수/저장 threshold, 정상/이상, 누락 후보, 버전, 오류 및 data_kind
- summary.json: 발견/성공/오류/미처리 수, 정상·이상 수, 전체/성공 파일별 정상 비율, 점수 min/max/mean/median/q25/q75, 실제 reference 점수 및 일치 여부
- README.md: 한국어 실제 관측 요약과 한계

각 파일 후 CSV를 저장하고 중단되면 가능한 부분 요약을 남긴다. 모델/원본을 변경하지 않는다. 운영 DB에 접근하지 않는다. 경로 metadata는 20260919/1800RPM/Unbalance_-1/Vib인지 검증한다. 파일 수가 다르거나 파일 오류가 발생해도 오류를 정상으로 세지 않는다.

기준 20260919_132506_Vib.mat의 제공 표시값 0.37572573은 비교에만 사용한다. rtol=0, atol=5e-9+1e-15(표시 소수 8자리의 반 단위 및 부동소수점 여유)이며 threshold가 아니다. 실제 계산값은 reference_actual_score에 기록한다. 다른 파일 값을 강제 일치시키지 않는다.

normal_fraction_total = 정상 판정 수 / 전체 발견 수. 오류/미처리가 있으면 이는 확인된 정상 후보 비율의 하한이고 모든 파일의 판정이 알려진 비율이 아니다. normal_fraction_successful도 함께 기록한다. completed_and_checked는 실제 50개·전부 성공·기준 일치·모델 불변일 때만 true다. 그렇지 않으면 종료 코드 1이며 CSV/JSON의 오류를 조사한다.

정상으로 나온 디렉터리 레이블 Unbalance_-1은 탐지 누락 후보일 뿐 실제 물리 결함/라벨의 진위를 자동 확정하지 않는다. 학습·평가·같은 날짜/Run 중복이 확인되지 않아 독립 Test 성능이나 정확도라고 부르지 않는다. 합성 테스트는 실제 50개 결과가 아니다.
