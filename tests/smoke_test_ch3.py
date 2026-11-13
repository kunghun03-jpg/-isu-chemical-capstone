"""CH3 regression smoke test: temporary artifacts only, no training or evaluation."""

from pathlib import Path
import csv
import hashlib
import tempfile

import numpy as np
import torch
from scipy.io import loadmat, savemat

from vibration.channel import select_ch3, validate_spectra
from vibration.model import Autoencoder
from vibration import spectra, normalize, features


def main():
    torch.set_num_threads(2)
    project = Path(__file__).resolve().parents[1]
    # Verify a real source file without writing to it.
    real = next((project / "data").rglob("*.mat"))
    original_hash = hashlib.sha256(real.read_bytes()).digest()
    raw = loadmat(real)["signals"]
    selected = select_ch3(raw)
    assert selected.shape == (1, 65536)
    np.testing.assert_array_equal(selected[0], raw[2])
    assert not np.shares_memory(selected, raw)
    assert hashlib.sha256(real.read_bytes()).digest() == original_hash

    with tempfile.TemporaryDirectory(prefix="ch3_smoke_") as directory:
        root = Path(directory)
        output = root / "results"
        output.mkdir()
        data_dir = root / "data" / "20260916_1800RPM"
        data_dir.mkdir(parents=True)
        rng = np.random.default_rng(42)
        signal = rng.normal(size=(6, 65536))
        source = signal.copy()
        rows = []
        for split in ("train", "validation", "test_normal", "test_anomaly"):
            condition = "IR" if split == "test_anomaly" else "Normal1"
            folder = data_dir / condition / "Vib"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{split}.mat"
            savemat(path, {"signals": signal, "fs": np.array([[25600.0]])})
            rows.append(
                {
                    "file": path.relative_to(root).as_posix(),
                    "condition": condition,
                    "split": split,
                }
            )
        with (output / "split_1800rpm.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        spectra.run(root, output, 1800)
        # Independently compute the original six-channel FFT and check CH3 equivalence.
        segments = source.reshape(6, 8, 8192).transpose(1, 0, 2)
        segments = segments - segments.mean(axis=2, keepdims=True)
        window = np.hanning(8192)
        amplitude = np.abs(np.fft.rfft(segments * window, axis=2)) / window.sum()
        amplitude[:, :, 1:-1] *= 2
        expected = np.log1p(amplitude[:, 2:3, :3201]).astype(np.float32)
        with np.load(output / "train_spectra.npz") as data:
            assert data["X"].shape == (8, 1, 3201)
            np.testing.assert_array_equal(data["X"], expected)
            assert set(data["segment_id"]) == set(range(8))
        normalize.run(root, output, 1800)
        with np.load(output / "train_normalized.npz") as data:
            values = data["X"].copy()
        validate_spectra(values)
        with np.load(output / "normalization_1800rpm.npz") as ref:
            np.testing.assert_allclose(
                ref["mean"], expected.astype(np.float64).mean(axis=0, keepdims=True)
            )
            standardized = (
                (expected.astype(np.float64) - ref["mean"]) / ref["std"]
            ).astype(np.float32)
            np.testing.assert_array_equal(values, standardized)
        model = Autoencoder().eval()
        batch = torch.from_numpy(values[:2])
        with torch.no_grad():
            reconstructed = model(batch)
            errors = (reconstructed.double() - batch.double()).square().mean(dim=(1, 2))
        assert reconstructed.shape == batch.shape == (2, 1, 3201)
        assert errors.shape == (2,) and torch.isfinite(errors).all()
        np.testing.assert_allclose(
            errors.numpy(),
            np.mean(
                (
                    reconstructed.numpy().astype(np.float64)
                    - values[:2].astype(np.float64)
                )
                ** 2,
                axis=(1, 2),
            ),
        )
        for wrong in (np.zeros((2, 6, 3201)), np.zeros((2, 1, 3200))):
            try:
                validate_spectra(wrong)
            except ValueError:
                pass
            else:
                raise AssertionError("Old or invalid spectra were accepted")
        try:
            model(torch.zeros(2, 6, 3201))
        except ValueError:
            pass
        else:
            raise AssertionError("Six-channel model input was accepted")
        features.run(root, output, 1800)
        with (output / "rotation_features.csv").open(encoding="utf-8-sig") as f:
            fields = csv.DictReader(f).fieldnames
        assert "CH3_peak" in fields and "CH1_peak" not in fields
    print(
        "PASS: real MAT CH3 selection, unchanged source, FFT equivalence, normalization, model (2, 1, 3201), MSE, old-shape rejection, CH3 features"
    )


if __name__ == "__main__":
    main()
