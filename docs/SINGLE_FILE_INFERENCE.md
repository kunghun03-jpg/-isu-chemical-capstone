# 저장 모델 공통 단일 MAT 추론

현재 기준은 **기존 주파수별 Train 정규화**입니다. 공통 평균·표준편차 개발 비교안은 채택하지 않습니다.
`date_prelim_20261106`과 `ae_globalnorm_dev_20261106`의 결과는 모두 보존합니다.
추론 대상은 기존 날짜 예비 실험의 AE·OCSVM·IF이며 DAE는 제외합니다.

## 실행

프로젝트 루트에서 과학 계산 패키지가 설치된 Python 환경을 사용합니다.
추론 패키지는 이번 작업에서 세 RPM 모두 생성했습니다. 이미 있는 패키지에는 내보내기를 다시 실행하지 않습니다.

```powershell
python -m scripts.infer_mat --file "data/20260919/20260919_900RPM/Normal1/Vib/20260919_131643_Vib.mat" --rpm 900 --model all
```

`--model ae`, `--model ocsvm`, `--model iforest`로 하나만 선택할 수 있습니다.
`--output outputs/single_file_inference/example.json`을 추가하면 새 JSON에 저장합니다. 기존 파일 덮어쓰기는 거부합니다.
다른 MAT 파일은 `--file` 경로를 바꾸고 해당 RPM을 지정하세요. RPM은 측정값을 추정하는 기능이 아니라 호출자가 제공하는 조건입니다.

향후 다른 **주파수별 정규화 실험**을 패키지로 내보낼 때만 다음 명령을 사용합니다. 모델을 새로 학습하지 않습니다.

```powershell
python -m scripts.export_inference_bundles --run-name date_prelim_20261106
```

## 공유 API와 웹 연결 지점

```python
from vibration.inference import SavedModelInference

engine = SavedModelInference("outputs/inference_bundles/date_prelim_20261106/900rpm")
result = engine.predict_mat("path/to/uploaded.mat", rpm=900)
# 웹에서 받은 파일은 디스크 저장 없이 처리 가능:
# result = engine.predict_bytes(uploaded_bytes, rpm=900, filename="uploaded.mat")
```

모델은 엔진 생성 때 한 번 로드합니다. 웹에서는 RPM별 엔진을 재사용하고 반환된 JSON을 화면·이력 저장 기능에 연결하면 됩니다.
Python API와 CLI에 이어 [Streamlit 최소 업로드 화면](STREAMLIT_APP.md)과 [SQLite 이력](SQLITE_HISTORY.md)을 연결했습니다.
추론 패키지는 자체 완결형으로, 원래 Train·Validation·Test NPZ나 원본 실험 폴더를 추론 중 읽지 않습니다.
신뢰하는 프로젝트에서 만든 모델 파일을 사용해야 하며 NumPy·PyTorch·scikit-learn 버전이 패키지와 다르면 중단합니다.

## 저장 기준과 계산

패키지 위치: `outputs/inference_bundles/date_prelim_20261106/{RPM}rpm/`.

| 파일 | 용도 |
| --- | --- |
| manifest.json | RPM, 전처리 명세, 모델 식별값, 구간·파일 임계값, 버전, SHA256, 기존 성능과 한계 |
| normalization.npz | 기존 Train 주파수별 평균·표준편차의 바이트 동일 사본 |
| autoencoder.pt | 기존 AE 체크포인트 사본 |
| ocsvm.joblib / iforest.joblib | 기존 비교 모델 사본 |

원본 MAT는 `signals=(6,65536)`, `fs=25600`이어야 합니다. 전체 6채널에 기존 품질 조건을 적용한 후 CH3만 선택합니다.
CH3 → 8개 구간 → 구간별 평균 제거 → Hann FFT → 단측 진폭 보정 → 0–10kHz → log1p → float32 → 저장 Train 평균·표준편차로 Z-score 순서입니다.
배치와 추론은 `vibration/preprocessing.py`를 공유합니다. 결과는 `(8,1,3201)`입니다.

AE는 복원 MSE, OCSVM·IF는 `-score_samples`를 사용하여 모두 점수가 클수록 이상입니다.
파일 점수는 **8개 구간 점수의 산술평균**입니다. 파일·구간 판정은 각각 해당 수준의 저장 임계값을 사용하며 **점수 > 임계값**일 때만 Abnormal입니다.
AE 파일 임계값은 기존 `evaluation_file_...csv`의 단일 임계값을 내보냅니다. AE의 `threshold_...json`은 구간 임계값이므로 파일 판정에 대신 사용하지 않습니다.
비교 모델은 기존 저장 모델과 `thresholds.json`의 두 임계값을 대조합니다.
새 Q95 계산, 정규화 통계 계산, 파라미터 갱신, 재학습은 없습니다.

내보낼 때만 기존 실험 분할·식별값과 저장 정규화 적용 결과를 검사합니다. 추론할 때는 패키지와 입력 MAT만 검사합니다.
해시는 우발적인 파일 변경을 탐지하며 실제 Run 독립성이나 물리적인 센서 정확성을 증명하지 않습니다.

검증 결과는 [단일 추론 검증](SINGLE_FILE_INFERENCE_VALIDATION.md), 실제 적용 한계는 [별도 한계 문서](INFERENCE_LIMITATIONS.md)를 확인하세요.

## 이번 변경 파일

- 신규 코드: `vibration/preprocessing.py`, `vibration/inference_bundle.py`, `vibration/inference.py`.
- 신규 실행 도구: `scripts/export_inference_bundles.py`, `scripts/infer_mat.py`, `scripts/verify_single_file_inference.py`.
- 신규 테스트: `tests/test_single_inference.py`.
- 기존 코드 변경: `vibration/spectra.py`의 FFT 계산을 동일 계산식의 공유 함수 호출로 교체했습니다. 모델 구조·학습 설정·임계값은 변경하지 않았습니다.
- 문서: 이 안내와 검증·한계 문서를 추가하고 README, CODE_GUIDE, GUIDE_STATUS, 정규화 비교 문서에 채택 기준을 반영했습니다.
