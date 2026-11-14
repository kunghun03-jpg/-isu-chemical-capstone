# 기존 모델 정보 화면: 정상 조건 비교와 데이터 품질

기존 `streamlit_app_v3.py`의 **모델 정보** 메뉴에 읽기 전용 섹션을 추가한다. 실행 명령과 기존 추론·저장·이력·다운로드·DAE 차단은 유지한다. 새 앱이나 의존성을 추가하지 않는다.

## 입력 위치

프로젝트 루트를 기준으로 아래 파일을 읽는다. 현재 Codex 환경에는 실제 두 CSV가 없어 합성 테스트만 수행한다. 없는 파일을 예시 결과로 생성하지 않는다.

- `outputs/normal_condition_comparison/normal1_vs_normal2_summary.csv`
- `outputs/normal_condition_comparison/excluded_files.csv` (선택)

매번 화면을 다시 실행할 때 읽으며 파일·DB·모델에 쓰지 않는다. 자료가 없으면 배치 위치를 안내한다. 열/값 오류가 있으면 해당 자료 표시만 보류하고 앱을 중단하지 않는다. 비교 CSV가 없거나 손상돼도 제외 파일 자료는 독립적으로 표시한다.

## 비교 CSV 형식

실제 CSV의 열 형식은 아직 확인되지 않아 아래 두 가지 명시적 형식을 지원한다. 열 이름은 앞뒤 공백과 대소문자, UTF-8 BOM을 처리한다. 중간값·평균 등을 중앙값으로 임의 해석하지 않는다.

**조건별 행 형식** (각 RPM·조건·날짜/Run에 한 행):

```csv
rpm,condition,date,rms_median,one_x_amplitude_median,two_x_amplitude_median,three_x_amplitude_median
```

- `condition`: Normal1 또는 Normal2. `normal_condition`, `normal_type`도 가능.
- RPM: 900/1200/1800.
- RMS 열: `rms_median`, `rms_ac_median`, `median_rms` 중 하나.
- 1X 열: `one_x_amplitude_median`, `one_x_median`, `1x_median` 중 하나.
- 2X·3X도 같은 규칙으로 `two_x`/`three_x` 및 `2x`/`3x`를 지원한다.
- 같은 의미로 인식되는 열이 둘 이상이면 선택하지 않고 오류를 안내한다. RMS와 AC RMS 중 무엇인지 원본 보고서 정의를 확인한다.
- 날짜 `date`/`measurement_date`, Run `run`/`run_id`, 표본 수 `n`/`n_files`/`file_count`는 선택이다. 제공된 값만 표시한다.

**Normal1·Normal2가 나란한 열 형식**:

```csv
rpm,date,normal1_rms_median,normal2_rms_median,normal1_1x_median,normal2_1x_median,normal1_2x_median,normal2_2x_median,normal1_3x_median,normal2_3x_median
```

각 특징의 위 별칭에 `normal1_`/`normal2_`를 붙인 열을 지원한다. 두 형식 모두 CSV에 기록된 중앙값만 사용하며 원본 MAT에서 재계산하거나 중앙값을 다시 평균내지 않는다.

날짜가 여러 개면 날짜 선택으로 분리한다. 명시된 Run이 여러 개면 Run도 선택하며 서로 합치지 않는다. 같은 RPM·조건·날짜/Run의 중복 집계는 오류로 표시한다. 날짜 열이 없으면 날짜별 구분을 확인할 수 없다고 안내한다.

비유한값·잘못된 숫자·음수 RMS/진폭·지원 외 RPM·Normal1/2 외 조건은 자료 표시를 보류한다. 누락된 조건/RPM을 0으로 채우지 않고 누락 범위를 안내한다. 그래프는 특징별로 별도의 y축을 사용하며, 센서 물리 단위나 정상 범위를 임의로 붙이지 않는다.

## 제외 파일 CSV

필요한 정보:

- 파일명 또는 경로: `filename`, `file_name`, `file` / `path`, `file_path`
- RPM: `rpm`
- 조건: `condition`, `normal_condition`, `normal_type`
- 제외 사유: `reason`, `exclusion_reason`, `excluded_reason`, `error`

파일명과 경로가 함께 있으면 둘 다 보존한다. `reason`과 `error`가 함께 있으면 사유와 오류 상세를 각각 표시한다. RPM/조건 값이 비어 있으면 '미기록'으로 표시하고 파일명에서 추측하지 않는다. 데이터 품질 표는 현재 그래프의 날짜/Run 선택과 별개로 전체 제외 기록을 표시한다고 명시한다.

파일이 없는 경우 '제외 파일 자료 없음'이며 제외가 0건이라는 뜻은 아니다. 빈 CSV·열 형식 오류도 구분해 안내한다.

## 해석 제한

**날짜 기준 예비 평가이며 실제 Run 독립성은 확인되지 않았다**는 문구를 항상 표시한다. CSV에 Run 열이 있다는 사실도 독립성 검증 근거로 삼지 않는다. 이 자료는 정상 조건의 특징량 비교이며 탐지 성능·모델 정확도·임계값의 적합성 또는 DAE 지원 RPM 인증이 아니다.

이 추가 기능은 기존 실시간 업로드 결과를 수정하지 않는다. AE/OCSVM/IF/RF·DAE 로더, 점수 수식, 정규화, threshold, manifest, 모델 파일과 DB 스키마를 변경하지 않는다.

## 실행·테스트

```bash
"$HOME/capstone-web-venv/bin/python" -m streamlit run streamlit_app_v3.py
"$HOME/capstone-web-venv/bin/python" -B -m unittest tests.test_normal_comparison_ui
```

테스트용 CSV는 임시 디렉터리에만 만들며 실제 outputs에는 저장하지 않는다. 나란한 열/조건별 행 형식, 날짜·Run 분리, 누락·빈 파일·손상 자료·NaN/Inf·중복, 제외 사유 보존, 기존 앱 메뉴 연결을 검증한다. 실제 CSV 표시 결과의 검증은 WSL에서 파일을 배치한 뒤 수행해야 한다.

최종 확인: 신규 합성 테스트 11개를 포함한 회귀 테스트 83개 통과. RF 재학습 호출을 차단한 상태에서 기존 추론·보류·저장·이력 경로를 함께 확인했다. 실제 요청 CSV는 이 환경에 없어 실제 자료 표시 검증은 미수행이다.
