"""Legacy DAE artifact adapter. No training, threshold fitting, or RPM claims.

The caller supplies a trusted directory containing autoencoder_model.pth,
normal_dataset_normalized.npz (mean/std), and normal_threshold.npz (threshold).
Real trained-weight inference has not yet been validated in this environment.
"""

from pathlib import Path

import numpy as np
import torch

from vibration.dae_model import ConvAutoencoder
from vibration.dae_preprocessing import (
    normalize_spectrum, preprocess_mat_bytes, validate_statistics,
)


class SavedDAEInference:
    def __init__(self, artifact_dir):
        self.folder = Path(artifact_dir).resolve()
        with np.load(self.folder / "normal_dataset_normalized.npz", allow_pickle=False) as data:
            self.mean, self.std = data["mean"].copy(), data["std"].copy()
        validate_statistics(self.mean, self.std)
        with np.load(self.folder / "normal_threshold.npz", allow_pickle=False) as data:
            threshold = data["threshold"]
        self._initialize_model(threshold)

    def _initialize_model(self, threshold):
        if (
            threshold.size != 1 or threshold.dtype.kind not in "fiu"
            or not np.isfinite(threshold).all() or float(threshold.item()) < 0
        ):
            raise ValueError("DAE 임계값은 유한한 음이 아닌 단일 값이어야 합니다.")
        self.threshold = float(threshold.item())
        self.model = ConvAutoencoder()
        state = torch.load(
            self.folder / "autoencoder_model.pth", map_location="cpu", weights_only=True,
        )
        self.model.load_state_dict(state, strict=True)
        if any(not torch.isfinite(value).all() for value in self.model.state_dict().values()):
            raise ValueError("DAE 가중치에 유한하지 않은 값이 있습니다.")
        self.model.eval()
        self.model.requires_grad_(False)

    def predict_bytes(self, raw):
        normalized = normalize_spectrum(preprocess_mat_bytes(raw), self.mean, self.std)
        tensor = torch.tensor(normalized, dtype=torch.float32).reshape(1, 1, 800)
        with torch.no_grad():
            reconstructed = self.model(tensor)
            # Legacy float32 MSE; unlike the current AE, do not promote to double.
            mse = torch.mean((tensor - reconstructed) ** 2, dim=(1, 2)).item()
        if not np.isfinite(mse) or not torch.isfinite(reconstructed).all():
            raise ValueError("DAE 복원 결과 또는 MSE가 유한하지 않습니다.")
        return {
            "prediction": "이상" if mse > self.threshold else "정상",
            "mse": mse,
            "threshold": self.threshold,
            "input_spectrum": normalized,
            "reconstructed_spectrum": reconstructed.squeeze().numpy(),
        }

    def predict_mat(self, path):
        return self.predict_bytes(Path(path).read_bytes())


class SavedDAEBundleInference(SavedDAEInference):
    """Validated DAE bundle; RPM coverage must be confirmed before prediction.

    The legacy adapter remains available for parity checks, without bundle or RPM
    guarantees. Bundled inference never falls back to those unchecked artifacts.
    """

    def __init__(self, bundle_dir):
        from vibration.dae_bundle import load_manifest

        self.folder = Path(bundle_dir).resolve()
        self.manifest = load_manifest(self.folder)
        try:
            with np.load(self.folder / "normalization.npz", allow_pickle=False) as data:
                if set(data.files) != {"mean", "std"}:
                    raise ValueError("DAE 정규화 파일에는 mean/std만 필요합니다.")
                self.mean, self.std = data["mean"].copy(), data["std"].copy()
            validate_statistics(self.mean, self.std)
            with np.load(self.folder / "threshold.npz", allow_pickle=False) as data:
                if set(data.files) != {"threshold"}:
                    raise ValueError("DAE 임계값 파일에는 threshold만 필요합니다.")
                threshold = data["threshold"]
        except (KeyError, OSError) as error:
            raise ValueError("DAE 번들 통계 파일 형식 오류") from error
        self._initialize_model(threshold)

    def predict_bytes(self, raw, *, rpm=None):
        from vibration.dae_bundle import check_rpm

        check_rpm(self.manifest, rpm)
        return super().predict_bytes(raw)

    def predict_mat(self, path, *, rpm=None):
        from vibration.dae_bundle import check_rpm

        check_rpm(self.manifest, rpm)
        return self.predict_bytes(Path(path).read_bytes(), rpm=rpm)
