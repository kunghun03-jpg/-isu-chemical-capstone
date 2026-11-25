"""Experimental band-wise AE reconstruction scoring."""

from pathlib import Path

import numpy as np
import torch

from vibration.inference import SavedModelInference
from vibration.preprocessing import frequency_bins


ROOT = Path(__file__).resolve().parents[1]

BUNDLE_ROOT = (
    ROOT
    / "outputs/inference_bundles/date_prelim_20261106"
)

SUPPORTED_RPMS = (900, 1200, 1800)
FREQ_STEP = 3.125


def build_frequency_bands(rpm):

    if rpm not in SUPPORTED_RPMS:
        raise ValueError(f"Unsupported RPM: {rpm}")

    frequencies = np.asarray(
        frequency_bins(), dtype=np.float64
    )

    expected = np.arange(3201) * FREQ_STEP

    if (
        frequencies.shape != expected.shape
        or not np.allclose(
            frequencies, expected,
            rtol=0, atol=1e-9
        )
    ):
        raise ValueError("Frequency grid mismatch")

    one_x = rpm / 60.0
    two_x = one_x * 2.0

    bands = {
        "1X": (
            np.abs(frequencies - one_x)
            <= FREQ_STEP
        ),
        "2X": (
            np.abs(frequencies - two_x)
            <= FREQ_STEP
        ),
        "0_100Hz": frequencies < 100,
        "100_1000Hz": (
            (frequencies >= 100)
            & (frequencies < 1000)
        ),
        "1000_10000Hz": (
            (frequencies >= 1000)
            & (frequencies <= 10000)
        ),
    }

    for name, mask in bands.items():
        if not mask.any():
            raise ValueError(f"Empty band: {name}")

    return frequencies, bands


class BandwiseAE:

    def __init__(self, rpm, bundle_root=BUNDLE_ROOT):

        self.rpm = rpm

        self.frequencies, self.bands = (
            build_frequency_bands(rpm)
        )

        self.engine = SavedModelInference(
            Path(bundle_root) / f"{rpm}rpm",
            models=("autoencoder",),
            selected_artifacts_only=True,
        )

        self.model = self.engine._estimators[
            "autoencoder"
        ]

        self.model.eval()

    def score_bytes(self, raw):

        _, normalized = self.engine.preprocess(raw)

        normalized = np.asarray(
            normalized, dtype=np.float32
        )

        if normalized.shape != (8, 1, 3201):
            raise ValueError("Unexpected AE input shape")

        if not np.isfinite(normalized).all():
            raise ValueError("Invalid AE input")

        batch = torch.from_numpy(normalized)

        with torch.inference_mode():

            reconstructed = self.model(batch)

            if reconstructed.shape != batch.shape:
                raise ValueError(
                    "Reconstruction shape mismatch"
                )

            error = (
                reconstructed.double()
                - batch.double()
            ).square().cpu().numpy()

        error = error[:, 0, :]

        if not np.isfinite(error).all():
            raise ValueError(
                "Invalid reconstruction error"
            )

        overall_mse = float(error.mean())
        total_error = float(error.sum())

        result = {
            "rpm": self.rpm,
            "bundle_id": self.engine.manifest[
                "bundle_id"
            ],
            "overall_mse": overall_mse,
            "segment_scores": error.mean(
                axis=1
            ).tolist(),
            "bands": {},
            "evaluation_status": "experimental_only",
            "approved_for_production": False,
            "operational_model_changed": False,
        }

        for name, mask in self.bands.items():

            selected = error[:, mask]
            band_sum = float(selected.sum())

            result["bands"][name] = {
                "mse": float(selected.mean()),
                "bin_count": int(mask.sum()),
                "min_hz": float(
                    self.frequencies[mask].min()
                ),
                "max_hz": float(
                    self.frequencies[mask].max()
                ),
                "error_fraction": (
                    band_sum / total_error
                    if total_error > 0
                    else 0.0
                ),
                "global_contribution": (
                    band_sum / error.size
                ),
            }

        broad_bands = (
            "0_100Hz",
            "100_1000Hz",
            "1000_10000Hz",
        )

        combined = sum(
            result["bands"][name][
                "global_contribution"
            ]
            for name in broad_bands
        )

        if not np.isclose(
            combined,
            overall_mse,
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "Band contribution parity failed"
            )

        return result

    def score_file(self, path):

        return self.score_bytes(
            Path(path).read_bytes()
        )
