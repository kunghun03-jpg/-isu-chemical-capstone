"""단일 MAT의 저장 모델 추론 전용 API. 학습·통계 재계산·Q95 재설정 없음."""

from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re

import numpy as np
from scipy.io import loadmat

from vibration.inference_bundle import MODEL_NAMES, manifest_id, sha256_file
from vibration.preprocessing import (
    PREPROCESSING_CONFIG,
    apply_normalization,
    ch3_log_spectra,
)


class SavedModelInference:
    def __init__(self, bundle_dir, models=MODEL_NAMES, *, selected_artifacts_only=False):
        self.folder = Path(bundle_dir).resolve()
        self.manifest = json.loads(
            (self.folder / "manifest.json").read_text(encoding="utf-8")
        )
        manifest = self.manifest
        if manifest.get("schema_version") != 1 or manifest.get(
            "bundle_id"
        ) != manifest_id(manifest):
            raise ValueError("추론 manifest 버전/해시 불일치")
        if (
            manifest["preprocessing"] != PREPROCESSING_CONFIG
            or manifest["normalization"]["mode"] != "per_frequency"
        ):
            raise ValueError("지원하는 학습 전처리·주파수별 정규화 패키지가 아닙니다.")
        if (
            manifest["file_aggregation"] != "arithmetic_mean_of_8_segment_scores"
            or manifest["decision_rule"] != "score > threshold"
        ):
            raise ValueError("집계·판정 규칙 불일치")
        self.models = tuple(models)
        if (
            not self.models
            or len(self.models) != len(set(self.models))
            or set(self.models) - set(MODEL_NAMES)
        ):
            raise ValueError("AE·OCSVM·IF만 선택할 수 있습니다.")
        self.rpm = manifest["rpm"]
        required = {
            manifest["normalization"]["artifact"],
            *[manifest["models"][name]["artifact"] for name in self.models],
        }
        if not required.issubset(manifest["artifacts"]):
            raise ValueError("패키지 아티팩트 누락")
        if np.__version__ != manifest["versions"]["numpy"]:
            raise ValueError("검증된 전처리를 위해 패키지와 NumPy 버전을 맞추세요.")
        for name, checksum in manifest["artifacts"].items():
            if selected_artifacts_only and name not in required:
                continue
            if Path(name).name != name or sha256_file(self.folder / name) != checksum:
                raise ValueError("추론 파일 해시 또는 경로 불일치")
        with np.load(
            self.folder / manifest["normalization"]["artifact"], allow_pickle=False
        ) as stats:
            self.mean, self.std = stats["mean"].copy(), stats["std"].copy()
        apply_normalization(np.zeros((8, 1, 3201), np.float32), self.mean, self.std)
        self._estimators = {}
        for name in self.models:
            spec = manifest["models"][name]
            if not np.isfinite(
                [spec["file_threshold"], spec["segment_threshold"]]
            ).all():
                raise ValueError("저장 임계값 불일치")
            if name == "autoencoder":
                import torch
                from vibration.model import Autoencoder

                if str(torch.__version__) != manifest["versions"]["torch"]:
                    raise ValueError(
                        "검증된 AE를 위해 패키지와 PyTorch 버전을 맞추세요."
                    )
                torch.set_num_threads(min(4, torch.get_num_threads()))
                checkpoint = torch.load(
                    self.folder / spec["artifact"],
                    map_location="cpu",
                    weights_only=True,
                )
                if checkpoint["data_fingerprint"] != manifest["data_fingerprint"]:
                    raise ValueError("AE 데이터 식별값 불일치")
                model = Autoencoder()
                model.load_state_dict(checkpoint["model_state_dict"])
                model.requires_grad_(False)
                model.eval()
                if spec["score_method"] != "double_precision_reconstruction_MSE":
                    raise ValueError("AE 점수 방식 불일치")
            else:
                import joblib
                import sklearn

                if sklearn.__version__ != manifest["versions"]["scikit_learn"]:
                    raise ValueError("저장 모델과 scikit-learn 버전을 맞추세요.")
                saved = joblib.load(self.folder / spec["artifact"])
                if saved["config"]["data_fingerprint"] != manifest["data_fingerprint"]:
                    raise ValueError("비교 모델 데이터 식별값 불일치")
                np.testing.assert_allclose(
                    [saved["thresholds"]["segment"], saved["thresholds"]["file"]],
                    [spec["segment_threshold"], spec["file_threshold"]],
                )
                if spec["score_method"] != "negative_score_samples":
                    raise ValueError("비교 모델 점수 방식 불일치")
                model = saved["model"]
            self._estimators[name] = model

    def preprocess(self, raw_bytes):
        try:
            data = loadmat(BytesIO(raw_bytes))
        except Exception as error:
            raise ValueError(
                f"지원하는 MAT 파일을 읽을 수 없습니다: {type(error).__name__}"
            ) from error
        if "signals" not in data or "fs" not in data:
            raise ValueError("MAT에 signals와 fs가 필요합니다.")
        spectra = ch3_log_spectra(data["signals"], data["fs"])
        return spectra, apply_normalization(spectra, self.mean, self.std)

    def predict_bytes(self, raw_bytes, *, rpm, filename="uploaded.mat"):
        if rpm != self.rpm:
            raise ValueError("입력 RPM과 저장 모델 RPM이 다릅니다.")
        if not isinstance(raw_bytes, bytes):
            raise ValueError("MAT 원본 bytes가 필요합니다.")
        _, values = self.preprocess(raw_bytes)
        result = {
            "schema_version": 1,
            "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
            "input": {
                "name": filename,
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "rpm": rpm,
                "rpm_source": "caller-supplied nominal RPM, not measured",
                "channel": 3,
                "fs": 25600.0,
                "raw_shape": [6, 65536],
                "segment_shape": [8, 1, 3201],
            },
            "bundle_id": self.manifest["bundle_id"],
            "source_experiment": self.manifest["source_experiment"],
            "normalization": "per_frequency_train_reference",
            "experiment_mode": self.manifest["experiment_mode"],
            "run_independence_verified": False,
            "limitations": self.manifest["limitations"],
            "models": {},
        }
        for name, model in self._estimators.items():
            spec = self.manifest["models"][name]
            if name == "autoencoder":
                import torch

                with torch.inference_mode():
                    batch = torch.from_numpy(values).float()
                    scores = (
                        (model(batch).double() - batch.double())
                        .square()
                        .mean(dim=(1, 2))
                        .numpy()
                    )
            else:
                features = np.ascontiguousarray(values[:, 0, :], dtype=np.float64)
                scores = -np.asarray(model.score_samples(features), dtype=np.float64)
            if scores.shape != (8,) or not np.isfinite(scores).all():
                raise ValueError("이상점수 계산 실패")
            file_score = float(scores.mean())
            file_threshold = float(spec["file_threshold"])
            alarm = bool(file_score > file_threshold)
            result["models"][name] = {
                "file_score": file_score,
                "file_threshold": file_threshold,
                "predicted_anomaly": int(alarm),
                "decision": "Abnormal" if alarm else "Normal",
                "score_direction": "higher_is_more_anomalous",
                "aggregation": "mean_of_8_segment_scores",
                "segment_threshold": float(spec["segment_threshold"]),
                "segments": [
                    {
                        "segment_id": i,
                        "score": float(score),
                        "predicted_anomaly": int(score > spec["segment_threshold"]),
                    }
                    for i, score in enumerate(scores)
                ],
                "reference_file_metrics": self.manifest["reference_file_metrics"][name],
            }
        return result

    def predict_mat(self, path, *, rpm):
        path = Path(path)
        # 경로에 RPM 정보가 있으면 사용자 입력과도 대조합니다. 새 업로드 bytes에는 RPM을 따로 받아야 합니다.
        encoded = {
            int(m.group(1))
            for part in path.parts
            if (m := re.fullmatch(r"\d{8}_(\d+)RPM", part))
        }
        if encoded and encoded != {rpm}:
            raise ValueError("MAT 경로의 RPM과 요청 RPM이 다릅니다.")
        return self.predict_bytes(path.read_bytes(), rpm=rpm, filename=path.name)
