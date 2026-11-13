# WSL 웹앱 전용 환경

기존 `~/capstone/.venv`와 프로젝트 `.venv`를 변경하지 않고 외부에 새 웹 환경을 만듭니다.
모델 가중치·manifest·전처리·임계값·DAE·DB 코드는 수정하지 않습니다.

## 실행: 프로젝트 루트에서 두 명령

```bash
python3 -B scripts/setup_web_env.py
"$HOME/capstone-web-venv/bin/python" -m streamlit run streamlit_app.py
```

첫 명령은 Python 3.14를 우선 탐색합니다. PATH에 python3.14가 없더라도 실행 중인
python3가 3.14이면 그 인터프리터로 **새 환경만** 생성합니다. 기존 환경을 activate/deactivate할
필요가 없습니다. 특정 실행기를 쓰려면 `--python /path/to/python3.14`를 지정하세요.
모든 설치는 새 환경의 `bin/python -m pip`로 수행합니다.

고정 설치 버전:

| 패키지 | 버전 |
| --- | --- |
| NumPy | 2.4.6 |
| PyTorch | 2.14.1+cpu (공식 CPU index) |
| scikit-learn | 1.9.0 |
| SciPy | 1.18.1 |
| Streamlit | 1.67.0 |

추가 웹 의존성은 기존 requirements.txt를 사용하되 위 다섯 버전을 함께 제약합니다.
Python별 binary wheel 해결 가능 여부를 pip dry-run으로 검사하고 설치합니다.
소스 빌드·TLS 검사 해제·버전 검사 우회·자동 Python 다운그레이드는 하지 않습니다.
기존 pip target/prefix/user 설정은 새 환경 밖으로 설치를 돌리지 못하도록 제외하고,
proxy 및 CA 환경변수는 유지합니다. 사용자 pip 설정 파일은 읽지 않으므로 사설 CA가 필요한
WSL 환경은 PIP_CERT/REQUESTS_CA_BUNDLE 등 정상적인 CA 환경 설정을 제공해야 합니다.

## Python 3.12 대안

Python 3.14 설치/의존성 문제가 있을 때만, **이미 설치된** Python 3.12로 다른 새 경로를 사용합니다.

```bash
python3 -B scripts/setup_web_env.py --python python3.12 --env "$HOME/capstone-web-py312"
"$HOME/capstone-web-py312/bin/python" -m streamlit run streamlit_app.py
```

이 스크립트는 시스템 Python을 설치/교체하지 않습니다. python3.12가 없으면 먼저 WSL의
정상적인 Python 설치 절차로 준비해야 합니다. 기존 3.14 웹 환경도 삭제하지 않습니다.
번들 파일 누락·해시·모델 버전 오류는 Python을 바꾼다고 해결되는 문제가 아닙니다.

## 설치 후 실제 번들 로딩 검증

설치 후 pip check와 정확한 패키지 버전을 검사하고, 다음 세 디렉터리의 실제
SavedModelInference 로더를 호출해 **AE·OCSVM·IF 모두** 로드합니다.

```text
outputs/inference_bundles/date_prelim_20261106/{900,1200,1800}rpm/
```

각 manifest의 버전/전처리/집계 규격, artifact SHA256, 정규화 통계, 실제 AE checkpoint와
OCSVM/IF joblib 및 저장 threshold/데이터 식별값을 기존 로더로 검증합니다.
누락된 artifact를 재학습하거나 manifest를 바꾸지 않습니다.

- 종료 0: 설치와 세 RPM 번들 로딩 확인 완료. 실제 MAT 추론·성능까지 검증한 것은 아님
- 종료 2: 설치 후 버전/번들 로딩 실패 또는 미확인. 성공으로 표현하지 않음
- 종료 1: Python/설치/경로 보호 등 설정 실패

검사 결과는 새 환경의 `bundle-loading-report.json`에 저장됩니다. 필요한 경우
`bundle-loading-stderr.log`도 기록합니다. 설치가 끝나도 실제 모델 파일이 없으면 종료 2가 정상입니다.
문제를 해결한 뒤 같은 명령을 재실행할 수 있으며, 이 도구가 만든 환경만 재사용합니다.
기존의 일반 폴더/다른 가상환경/심볼릭 링크 경로는 덮어쓰지 않습니다.

검사만 다시 수행하려면:

```bash
"$HOME/capstone-web-venv/bin/python" -B -m scripts.verify_web_env
```

`--bundle-root`는 검사할 번들 루트를 별도로 지정하는 진단 옵션이며 웹앱의 고정 경로를
변경하지 않습니다. 실제 웹 실행에는 위 기본 위치에 올바른 번들이 있어야 합니다.

DAE 번들은 별도의 실행 버전을 요구할 수 있습니다. 새 환경에서 DAE 준비 상태가 달라져도
기존 DAE 환경/번들을 변경하거나 RPM 차단을 풀지 않습니다. 이 작업의 검증 대상은 기존 세 모델입니다.
실제 MAT로 세 모델 각각과 전체 분석을 확인하는 작업은 번들 로딩 성공 이후 필요합니다.

## 이번 검증 범위

클라우드의 별도 임시 Linux Python **3.14.4**에서 위 다섯 버전의 실제 설치와 pip check를
확인했습니다. 같은 환경에서 보호/버전/번들 로딩 테스트 5개와 전처리 테스트 2개가 통과했고,
합성 번들로 설정 스크립트 재실행도 성공했습니다. 이 합성 번들은 테스트용 가중치입니다.
실제 배포 번들은 클라우드에서 normalization/가중치 파일이 누락되어 로딩 검증이 실패했으며,
실제 WSL 파일의 로딩·MAT 추론을 검증했다고 주장하지 않습니다.
