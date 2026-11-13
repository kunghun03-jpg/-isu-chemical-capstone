# AE 단일 MAT: 오프라인 계산과 Streamlit V2 비교

## 현재 확인된 사실과 제한

- 사용자가 WSL 웹에서 확인한 파일: `20260912_144039_Vib.mat`, 900RPM / IR / Vib. 보고된 결과는 점수 약 48.3072, 임계값 약 0.324214, 이상이다. 이 수치는 사용자 제공 기록이며 이번 클라우드 실행 결과가 아니다.
- 저장소 inventory에 `data/20260912/20260912_900RPM/IR/Vib/20260912_144039_Vib.mat` 경로가 있다. 클라우드의 `/workspace`, `/tmp`에는 해당 MAT 원본이 없다. 실제 원본 가중치와 정규화 NPZ도 없어 실제 파일 점수 일치는 아직 검증하지 못했다.
- 이 Sep 12 IR 파일은 현재 `split_900rpm.csv`의 날짜 분할 시험 목록에 없다. 따라서 기존 시험 CSV의 점수와 직접 대조했다고 주장할 수 없다. 새 도구는 원본 체크포인트와 Train 통계를 사용해 해당 파일의 오프라인 점수를 다시 계산한다. 기존 평가 CSV에서 같은 파일명의 행이 있으면 참고용으로만 출력하며 동일 파일 증거로 간주하지 않는다.
- 저장소 CSV의 900RPM 임계값은 `0.3242139731159553`으로, 사용자가 원본 산출물 검증에서 보고한 `0.32421397830476884`와 미세하게 다르다. 어느 쪽도 덮어쓰거나 재계산하지 않는다. 도구는 WSL에서 지정한 원본 CSV와 실제 웹 번들 임계값을 정확히 비교하고 차이를 보고한다.

## 코드 조사 결과

오프라인 `vibration/spectra.py` → `normalize.py` → `evaluate.py` / `file_evaluate.py`와 웹 `dashboard_v2.analyze` → `SavedModelInference.predict_bytes`를 비교했다. FFT 전처리는 두 경로 모두 `preprocessing.ch3_log_spectra`를 공유한다. 정규화는 오프라인 `normalize.py`의 계산식과 웹 `apply_normalization`의 계산식이 동일하다.

| 단계 | 두 경로의 계산 |
| --- | --- |
| 입력 | signals (6,65536), fs=25600Hz; 전체 6채널 품질 검사 |
| 채널 | CH3, 인덱스 2 |
| 구간 | 순서대로 8개 × 8192샘플, (8,1,8192) |
| 평균 제거 | 구간별 평균 제거 |
| FFT | 대칭 `np.hanning(8192)`, rFFT 절댓값 / 윈도 합; 내부 bin 2배 |
| 주파수 | 0–10000Hz 양끝 포함, 3201 bin |
| 변환 | log1p 후 float32 |
| 정규화 | 저장된 Train mean/std, float64 연산 후 float32; 통계 재산출 없음 |
| 점수 | eval 모드 AE; 복원값과 입력을 double로 바꿔 구간 전체 MSE |
| 파일 집계 | 8개 구간 점수의 산술평균 |
| 판정 | 저장된 파일 임계값에 `score > threshold`; 같으면 정상 |

코드상 계산식 차이는 발견하지 못했다. 실제 입력·가중치·통계의 일치는 별도 검증이 필요하다. 기존 배치 평가의 배치 크기와 단일 파일 배치 크기 차이는 CPU 커널의 마지막 자리 오차를 만들 수 있다. 새 도구는 단일 파일 배치 8개로 오프라인 수식을 재현하며 과거 배치 평가를 그대로 재실행했다고 주장하지 않는다.

## WSL 실행

최신 코드가 있는 **Git 저장소**에서 실행한다. 데이터 폴더가 Git 저장소라고 가정하지 않는다. 웹 전용 가상환경을 그대로 사용하며 설치나 환경 변경은 하지 않는다.

```bash
cd /실제/Git/저장소
"$HOME/capstone-web-venv/bin/python" -B -m scripts.verify_ae_v2_parity \
  --search-root "$HOME/capstone" \
  --rpm 900 \
  --experiment-dir outputs/900rpm/date_prelim_20261106 \
  --output "$HOME/ae-parity/$(date -u +%Y%m%dT%H%M%S)-single-mat"
```

원본이 다른 위치에 있거나 같은 이름이 여러 개이면 `--search-root` 대신 `--file /실제/경로/20260912_144039_Vib.mat`를 지정한다. 원본 산출물 폴더에는 `autoencoder_900rpm.pt`, `normalization_900rpm.npz`, `evaluation_file_900rpm.csv`가 필요하다. 번들 기본 경로는 V2와 같은 환경변수 `ISU_CHEMICAL_V2_BUNDLE_ROOT` 또는 `outputs/inference_bundles/date_prelim_20261106`이다. 필요하면 `--bundle-root /실제/번들상위폴더`로 지정한다(그 아래 `900rpm/manifest.json`). V2 대표 모델 설정이 AE가 아니면 중단하며 자동 변경하지 않는다. 명시적 번들 경로를 지정하는 경우 실제 웹 설정과 같은 폴더인지 확인한다.

`--output`을 생략하면 stdout만 출력한다. 지정 시 저장소·입력 폴더 밖의 **새 폴더**에 `report.json`을 저장한다. 기존 결과 폴더는 거부한다. 파일 탐색 결과가 0개 또는 여러 개면 임의 선택하지 않는다.

## 결과 해석

마지막 두 줄에 `Offline | score=... | threshold=... | 정상/이상`, `V2 web | ...`가 나란히 출력된다. JSON에는 구간별 처리부터 복원 및 파일 점수까지 shape, 정확한 일치 여부, 최대 절대차, 첫 불일치 단계, 가중치 일치, 임계값 일치, 판정 일치가 기록된다.

- 전처리·정규화·모델 입력은 정확한 배열 일치를 요구한다.
- 복원·점수는 기본 rtol=1e-6, atol=1e-9 범위를 사용하고 정확한 일치 여부도 별도 보고한다. 임계값·판정은 정확히 같아야 한다.
- `passed_calculation_parity` / 종료 0: 이 파일의 두 계산 경로 호환. 성능·데이터 분할 타당성 검증은 아니다.
- `mismatch_requires_review` / 종료 2: `first_mismatch_stage`부터 조사한다. Train 통계, 가중치, 전처리, 저장 임계값 중 무엇이 다른지 분리해 표시한다.
- `NOT VERIFIED` / 종료 1: 파일 부재, 입력 오류, 번들 버전/해시 오류 등으로 검증 실패. 정상/이상 결과로 대체하지 않는다.

원본 MAT·체크포인트·통계·평가 CSV·번들 모든 artifact를 실행 전후 SHA256으로 확인한다. DB API, 학습·최적화, 임계값 추정, 기존 평가 결과 쓰기는 호출하지 않는다. 실제 웹 어댑터 내부 배열은 일시적인 Python trace와 읽기 전용 AE forward hook으로 관찰하며 성공·예외 모두 해제한다. 웹 서버 안에서 실행하는 도구가 아닌 독립 진단 CLI이다.

## 자동 검증

```bash
"$HOME/capstone-web-venv/bin/python" -B -m unittest tests.test_ae_v2_parity -v
```

합성 MAT와 임시 랜덤 가중치로 실제 V2 경로/독립 수식 일치, 통계 차이 위치, 임계값 차이, NaN 거부, 예외 시 trace 복원, 누락·중복·RPM 불일치, DB 미접근, 원본 보존, 결과 덮어쓰기 방지를 검사한다. 합성 테스트 통과는 실제 48.3072 점수 재현이나 탐지 성능 검증을 의미하지 않는다.

클라우드 실행 결과: Python 3.14.4 / NumPy 2.4.6 / PyTorch 2.14.1+cpu 환경에서 신규 6개와 기존 V2 6개, 총 12개 테스트가 통과했다. 기존 `tests.test_single_inference`의 실제 번들 로드 클래스는 `900rpm/normalization.npz` 부재로 실행하지 못했다. 실제 파일 검증 결과는 WSL 실행 후 생성되는 보고서로 확인해야 한다.
