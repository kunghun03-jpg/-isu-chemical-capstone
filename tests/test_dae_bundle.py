"""Synthetic/random-weight bundle tests; no real DAE coverage or accuracy claims."""

import json
import shutil
import unittest
from unittest.mock import patch

import numpy as np

from tests import test_dae as synthetic
from vibration.dae_bundle import convert_legacy_bundle, load_manifest, manifest_id, sha256_file
from vibration.dae_inference import SavedDAEInference, SavedDAEBundleInference


class DAEBundleTests(unittest.TestCase):
    setUp = synthetic.DAEInferenceTests.setUp

    def bundle(self, **kwargs):
        destination = self.folder.parent / (self.folder.name + "_bundle")
        self.addCleanup(lambda: shutil.rmtree(destination, ignore_errors=True))
        convert_legacy_bundle(self.folder, destination, model_version="synthetic-test-v1", **kwargs)
        return destination

    def edit_manifest(self, folder, change):
        path = folder / "manifest.json"
        data = json.loads(path.read_text())
        change(data)
        data["bundle_id"] = manifest_id(data)
        path.write_text(json.dumps(data))

    def test_conversion_extracts_only_statistics_and_preserves_prediction(self):
        np.savez(self.folder / "normal_dataset_normalized.npz", mean=self.mean, std=self.std,
                 X_train=np.ones((2, 800)), X_val=np.zeros((2, 800)))
        np.savez(self.folder / "normal_threshold.npz", threshold=1.0, val_mse=np.zeros(2))
        before = {p.name: sha256_file(p) for p in self.folder.iterdir()}
        folder = self.bundle(supported_rpms=[900], rpm_evidence="synthetic test only")
        self.assertEqual(before, {p.name: sha256_file(p) for p in self.folder.iterdir()})
        with np.load(folder / "normalization.npz") as data:
            self.assertEqual(set(data.files), {"mean", "std"})
            np.testing.assert_array_equal(data["mean"], self.mean)
            self.assertEqual(data["mean"].dtype, self.mean.dtype)
        with np.load(folder / "threshold.npz") as data:
            self.assertEqual(set(data.files), {"threshold"})
        expected = SavedDAEInference(self.folder).predict_bytes(self.raw)
        engine = SavedDAEBundleInference(folder)
        actual = engine.predict_bytes(self.raw, rpm=900)
        for key in expected:
            if isinstance(expected[key], np.ndarray):
                np.testing.assert_array_equal(actual[key], expected[key])
            else:
                self.assertEqual(actual[key], expected[key])
        path = self.folder / "sample.mat"
        path.write_bytes(self.raw)
        self.assertEqual(engine.predict_mat(path, rpm=900)["mse"], expected["mse"])
        with self.assertRaises(FileExistsError):
            convert_legacy_bundle(self.folder, folder, model_version="another")

    def test_unknown_unsupported_and_invalid_rpm_block_prediction(self):
        folder = self.bundle()
        self.assertIsNone(load_manifest(folder)["supported_rpms"])
        engine = SavedDAEBundleInference(folder)
        with patch.object(engine.model, "forward", side_effect=AssertionError("must not infer")):
            for rpm in (900, None, "900", True, 0):
                with self.subTest(rpm=rpm), self.assertRaises(ValueError):
                    engine.predict_bytes(self.raw, rpm=rpm)
        self.edit_manifest(folder, lambda d: d.update(supported_rpms=[900], rpm_evidence="synthetic"))
        engine = SavedDAEBundleInference(folder)
        with self.assertRaisesRegex(ValueError, "지원하지 않는"):
            engine.predict_bytes(self.raw, rpm=1200)
        with self.assertRaises(ValueError):
            engine.predict_bytes(b"invalid mat", rpm=900)

    def test_missing_hash_mismatch_and_symlink(self):
        folder = self.bundle()
        file = folder / "normalization.npz"
        content = file.read_bytes()
        file.unlink()
        with self.assertRaises(FileNotFoundError):
            SavedDAEBundleInference(folder)
        file.write_bytes(content + b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            SavedDAEBundleInference(folder)
        file.unlink()
        file.symlink_to(self.folder / "normal_dataset_normalized.npz")
        with self.assertRaisesRegex(ValueError, "심볼릭"):
            SavedDAEBundleInference(folder)

    def test_manifest_schema_config_versions_and_id_errors(self):
        folder = self.bundle()
        path = folder / "manifest.json"
        original = path.read_text()
        for change in (
            lambda d: d.update(schema_version=True),
            lambda d: d.update(preprocessing={}),
            lambda d: d.update(runtime_versions={}),
            lambda d: d.update(supported_rpms=[900], rpm_evidence=None),
            lambda d: d.update(artifacts={"../escape": "0" * 64}),
            lambda d: d.pop("model_version"),
        ):
            path.write_text(original)
            self.edit_manifest(folder, change)
            with self.assertRaises(ValueError):
                load_manifest(folder)
        path.write_text(original)
        data = json.loads(original); data["model_version"] = "tampered"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "manifest 해시"):
            load_manifest(folder)
        for text in ("[]", "not JSON"):
            path.write_text(text)
            with self.assertRaises(ValueError):
                load_manifest(folder)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            load_manifest(folder)

    def test_bad_statistics_even_with_matching_hash_and_failed_conversion(self):
        folder = self.bundle()
        np.savez(folder / "normalization.npz", mean=self.mean, std=np.zeros(800))
        self.edit_manifest(folder, lambda d: d["artifacts"].update(
            {"normalization.npz": sha256_file(folder / "normalization.npz")}))
        with self.assertRaises(ValueError):
            SavedDAEBundleInference(folder)
        missing_destination = self.folder.parent / (self.folder.name + "_missing")
        with self.assertRaises(FileNotFoundError):
            convert_legacy_bundle(self.folder / "missing", missing_destination, model_version="x")
        self.assertFalse(missing_destination.exists())
        with patch("vibration.dae_inference.SavedDAEBundleInference", side_effect=ValueError("validation failed")):
            with self.assertRaisesRegex(ValueError, "validation failed"):
                convert_legacy_bundle(self.folder, missing_destination, model_version="synthetic")
        self.assertFalse(missing_destination.exists())
        self.assertTrue((self.folder / "autoencoder_model.pth").is_file())
        for rpms, evidence in (([900], None), ([True], "x"), ([], "x"), ([900, 900], "x")):
            with self.assertRaises(ValueError):
                convert_legacy_bundle(self.folder, missing_destination, model_version="x",
                                      supported_rpms=rpms, rpm_evidence=evidence)


if __name__ == "__main__":
    unittest.main()
