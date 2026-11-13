# 최소 Streamlit 웹 화면

프로젝트 루트에서 기존 추론 패키지와 같은 Python 환경으로 실행합니다.
이번 확인 환경은 `C:\Users\82106\anaconda3\python.exe`, Streamlit 1.58.0입니다.

```powershell
python -m streamlit run streamlit_app.py
```

현재 기본 Python과 패키지 설치 환경이 다르면 다음 명령을 사용하세요.

```powershell
& "C:\Users\82106\anaconda3\python.exe" -m streamlit run streamlit_app.py
```

브라우저에서 `http://127.0.0.1:8501`에 접속합니다. 종료는 실행한 터미널에서 Ctrl+C입니다.
Streamlit이 없는 환경에서는 해당 Python으로 `python -m pip install -r requirements.txt`를 먼저 실행합니다.
저장 모델과 NumPy·PyTorch·scikit-learn 버전이 맞아야 하며 임의 업그레이드로 버전 검사를 우회하지 않습니다.

## 사용 순서

1. 실제 측정 조건에 맞는 RPM과 AE·OCSVM·IF·전체 중 하나를 선택합니다.
2. 20MB 이하 MAT 파일 하나를 업로드합니다.
3. CH3 원본 파형, 25,600Hz 샘플링 주파수, 65,536개 샘플, 2.56초 길이를 확인합니다.
4. **분석 실행**을 누르면 기존 `SavedModelInference.predict_bytes`를 호출합니다.
5. 모델별 파일 점수·저장 임계값·정상/이상 판정을 확인합니다. 필요하면 JSON을 다운로드합니다.
6. **결과 저장**으로 SQLite에 저장하고 아래 이력에서 검색·상세 조회·CSV 다운로드를 사용합니다. [이력 안내](SQLITE_HISTORY.md)를 확인하세요.
7. 결과 아래 **상세 분석**에서 CH3 log1p 스펙트럼과 모델별 8개 구간 점수·저장 구간 임계값을 확인합니다. [상세 화면 안내](DETAIL_ANALYSIS_SCREEN.md)를 참고하세요.

파일·RPM·모델을 바꾸면 이전 결과는 지워집니다. 분석 버튼을 누르기 전에는 모델 추론을 실행하지 않습니다.
미리보기는 전체 CH3 샘플을 사용하며, 센서 단위가 확정되지 않아 물리 단위를 임의로 표시하지 않습니다.

## 저장 모델과 업로드 처리

- 패키지는 고정된 `outputs/inference_bundles/date_prelim_20261106/{RPM}rpm/`에서 읽습니다. 사용자가 모델 경로를 업로드하지 않습니다.
- 저장 모델만 프로세스 내에서 재사용합니다. 사용자 업로드는 전역 캐시에 넣지 않습니다.
- 업로드는 메모리 bytes로 처리하며 임시 디스크 파일을 만들지 않습니다. JSON 다운로드도 서버 디스크에 쓰지 않습니다.
- 입력 크기와 MAT 변수 형태를 먼저 확인하고 기존 전처리의 전체 6채널 유한값·극단값·샘플링 검사를 적용합니다. 입력은 CH3입니다.
- 파일명은 표시용으로만 전달합니다. 경로 생성이나 정상/결함 상태 추정에 사용하지 않습니다.
- 파일 점수는 기존과 같은 8개 구간 점수 평균, 판정은 저장 파일 임계값을 초과할 때 이상입니다.
- 재학습·Train 통계 재계산·Validation Q95 재설정은 수행하지 않습니다.
- 입력 형식 오류, 패키지 누락, 버전·해시 불일치는 오류 안내를 표시하고 판정을 생성하지 않습니다.
- `.streamlit/config.toml`은 로컬 접속과 20MB 업로드 제한을 설정합니다. 이번 화면은 외부 공개 배포용 인증·권한 기능을 포함하지 않습니다.

날짜 간 높은 정상 오탐률과 실제 Run 미확정은 화면 상단에 상시 표시합니다.
점수 일치는 실사용 탐지 성능 검증을 의미하지 않습니다. [성능 한계](INFERENCE_LIMITATIONS.md)를 함께 확인하세요.
SQLite 이력은 후속 구현으로 추가했습니다. 결함 후보 추정과 DAE는 포함하지 않습니다.

## 검증

SQLite 이력 추가 후 전체 **35개 테스트 통과**. 재시작·연결·중복·실패 검증은 [SQLite 이력 안내](SQLITE_HISTORY.md)에 기록했습니다.
아래 31개 기록은 이력 추가 전 최소 화면의 검증 기록입니다.

2026-11-06 확인 결과:

- 전체 회귀 테스트 **31개 통과**. 신규 웹 테스트 3개 포함.
- 900·1200·1800RPM의 실제 MAT 각 1개에서 AE·OCSVM·IF 전체 선택과 개별 선택 모두 기존 CLI의 파일 점수·저장 임계값·판정·구간 결과와 정확히 일치했습니다.
- 업로드 전 추론 미실행, 선택 변경 시 결과 삭제, 잘못된 MAT·모델 누락 안내, 원본 CH3 미리보기와 파일명에 무관한 판정을 확인했습니다.
- 로컬 임시 서버의 health 응답과 기본 페이지 HTTP 200을 확인하고 검사 서버를 종료했습니다.
- 기존 결과·추론 패키지 **336개 파일의 SHA256 변경 없음**을 확인했습니다. 원본 MAT는 읽기만 했습니다.

추가 파일은 `streamlit_app.py`, `vibration/web_inference.py`, `.streamlit/config.toml`, `tests/test_streamlit_app.py`, 이 문서입니다.
requirements에 Streamlit·pandas를 추가하고 기존 안내 문서의 웹 구현 상태를 갱신했습니다.
기존 CLI·추론 엔진·모델·정규화 기준·임계값은 변경하지 않았습니다.

`tests/test_streamlit_app.py`는 Streamlit AppTest로 업로드·미리보기·분석 버튼·결과 표와 오류 안내를 검사합니다.
세 RPM에서 실제 MAT 하나씩, 전체 및 각 단일 모델 선택 결과를 기존 CLI JSON과 대조합니다.
파일 점수·임계값·판정·구간 결과는 같아야 하며 업로드명 Normal1/IR에 따른 결과 변화도 없어야 합니다.
학습 함수·역전파·Q95 계산을 금지한 상태에서도 웹 분석이 완료되는지 확인합니다.

```powershell
python -m unittest tests.test_streamlit_app -v
```

브라우저의 파일 선택 대화상자 자체는 자동 검사 대상이 아니며 AppTest에는 메모리 업로드를 주입합니다.
이전 전체 1,484개 배치 대조 결과는 [단일 추론 검증](SINGLE_FILE_INFERENCE_VALIDATION.md)에 보존합니다.
