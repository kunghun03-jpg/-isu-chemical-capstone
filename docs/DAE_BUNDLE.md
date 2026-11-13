# DAE 추론 번들 v1

DAE 전용 규격입니다. 기존 AE·OCSVM·IF 번들, Streamlit, 이력 DB는 변경하지 않습니다.
모델 구조와 Welch → sqrt(PSD) → 800특성 → log10 → 저장 Z-score → float32
복원 MSE → `MSE > threshold` 계산은 legacy와 같습니다. 재학습·통계 재계산은 없습니다.

## 파일

```text
<bundle>/
  manifest.json
  autoencoder_model.pth
  normalization.npz       # mean, std만; 원래 dtype/shape 유지
  threshold.npz           # threshold만; legacy의 float 변환값
```

manifest는 schema_version, model_type, architecture, model_version, preprocessing,
scoring, supported_rpms, rpm_evidence, training_provenance, runtime_versions,
artifacts(각 파일 SHA256), bundle_id(manifest 내용 SHA256)를 기록합니다.
전처리에는 Welch의 창·겹침·길이·detrend·scaling·평균법, 주파수 범위의 상한 제외,
pooling·로그·dtype·정규화가 명시됩니다. 임계값은 threshold.npz의 해시로 보호합니다.

`runtime_versions`는 **변환 실행 환경**의 NumPy/SciPy/PyTorch 버전입니다.
원래 학습 버전이나 실제 모델 검증 완료를 의미하지 않습니다. 로더는 이 버전과
실행 환경이 일치해야 합니다. 다른 환경에서 변환한다고 기존 모델의 호환성이 입증되지는 않습니다.
`training_provenance` 기본값은 `unverified`입니다.

## legacy 파일 변환

프로젝트 루트에서 같은 Python 가상환경으로 실행합니다.

```python
from vibration.dae_bundle import convert_legacy_bundle

manifest = convert_legacy_bundle(
    "path/to/trusted_legacy_artifacts",
    "path/to/new_dae_bundle",
    model_version="legacy-dae-v1",
)
```

원본에는 다음 세 파일이 있어야 합니다.

- autoencoder_model.pth: ConvAutoencoder의 raw state_dict
- normal_dataset_normalized.npz: mean, std (다른 학습 배열은 복사하지 않음)
- normal_threshold.npz: threshold (train/val/test 점수 배열은 복사하지 않음)

변환은 원본 통계와 가중치를 검증하고, 새 디렉터리에 필요한 파일만 저장한 뒤
생성된 번들을 다시 로드해 검사합니다. 기존 목적지는 덮어쓰지 않습니다.
파일 누락 또는 유효하지 않은 원본이면 성공을 반환하지 않습니다.
가중치는 신뢰하는 출처의 파일만 사용하세요. SHA256은 파일 변경 탐지이며 출처 인증은 아닙니다.

## RPM 정책과 추론

실제 학습된 DAE의 지원 RPM은 현재 확인되지 않았습니다.
변환 기본값은 `supported_rpms: null`, `rpm_evidence: null`이며 이 경우 **추론을 차단**합니다.
파일 이름·MAT의 배열 모양·현재 세 모델의 RPM 목록에서 지원 범위를 추정하지 않습니다.

실제 학습 기록과 검증으로 적용 범위가 확인된 후에만 변환 인수에
`supported_rpms=[확인된_정수_RPM, ...]`, `rpm_evidence="확인 기록의 위치와 내용"`를 지정합니다.
근거 문자열은 호출자의 선언이며 변환기가 과학적으로 검증하는 것은 아닙니다.

```python
from vibration.dae_inference import SavedDAEBundleInference

engine = SavedDAEBundleInference("path/to/new_dae_bundle")
# 확인된 범위를 기록한 번들에서만 실행 가능. RPM 인수는 필수입니다.
result = engine.predict_mat("path/to/input.mat", rpm=measured_rpm)
# 또는 engine.predict_bytes(raw_mat_bytes, rpm=measured_rpm)
```

RPM 미입력·잘못된 형식·미확인·미지원 시 모델 forward 전에 중단합니다.
입력 MAT의 조건은 signals=(6,65536), fs=25600Hz, 유한한 실수 CH3입니다.
MAT 자체로 실제 RPM을 측정하지 않으므로 호출자가 측정 조건을 확인해야 합니다.
기존 SavedDAEInference는 1단계 legacy 대조 API로 유지되며 RPM·번들 검증을 제공하지 않습니다.
새 번들 경로는 이 legacy API로 자동 우회하지 않습니다.

## 검증 범위

`python -m unittest tests.test_dae tests.test_dae_bundle -v`

합성 MAT·임시 가중치로 변환·로드·계산 일치, 누락·해시·규격·통계·RPM 오류를 검사합니다.
실제 가중치로 변환하거나 실제 DAE 성능을 검증한 결과가 아닙니다.
후속 검증에는 실제 가중치·정규화·저장 임계값, 학습/분할/버전 기록,
지원 RPM 근거와 해당 RPM의 실제 MAT가 필요합니다. 기존 legacy 결과와 전처리·복원·
MSE·판정을 대조한 뒤 별도의 분리된 데이터에서 성능을 평가해야 합니다.
