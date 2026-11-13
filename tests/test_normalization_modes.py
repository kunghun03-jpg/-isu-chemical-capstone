"""Train 통계만 사용하고 기존 주파수별 정규화를 보존하는지 검사합니다."""

from pathlib import Path
import contextlib
import io
import tempfile
import unittest

import numpy as np
from vibration.normalize import run


class NormalizationModeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="norm_mode_test_")
        self.addCleanup(self.temp.cleanup)
        self.output=Path(self.temp.name)
        rng=np.random.default_rng(42)
        self.train=(rng.normal(size=(16,1,3201))+np.linspace(0,10,3201)).astype(np.float32)
        for split in ("train","validation","test_normal","test_anomaly"):
            values=self.train if split=="train" else self.train+1000
            np.savez_compressed(self.output/f"{split}_spectra.npz",X=values,
                frequency=np.arange(3201),source_file=np.repeat(["f1","f2"],8),
                condition=np.repeat("Normal1",16),segment_id=np.tile(np.arange(8),2),
                date=np.repeat("20260912" if split=="train" else "20260919",16),
                rpm=np.repeat("900",16),run_id=np.repeat("",16),date_group=np.repeat("date:test",16))

    def normalize(self,mode):
        with contextlib.redirect_stdout(io.StringIO()):run(self.output,self.output,900,mode=mode)

    def test_global_train_scalar_statistics_and_metadata(self):
        self.normalize("global_train")
        train=self.train.astype(np.float64)
        with np.load(self.output/"normalization_900rpm.npz") as data:
            self.assertEqual(data["mean"].shape,(1,1,1))
            self.assertEqual(data["std"].shape,(1,1,1))
            np.testing.assert_allclose(data["mean"],train.mean())
            np.testing.assert_allclose(data["std"],train.std())
        with np.load(self.output/"validation_normalized.npz") as data:
            np.testing.assert_allclose(data["X"],((self.train+1000).astype(np.float64)-train.mean())/train.std(),rtol=1e-6)
            self.assertEqual(set(data["date_group"]),{"date:test"})
        with np.load(self.output/"train_normalized.npz") as data:
            self.assertAlmostEqual(float(data["X"].mean()),0,places=5)
            self.assertAlmostEqual(float(data["X"].std()),1,places=5)

    def test_per_frequency_matches_original_formula(self):
        self.normalize("per_frequency")
        train=self.train.astype(np.float64)
        with np.load(self.output/"normalization_900rpm.npz") as data:
            self.assertEqual(data["mean"].shape,(1,1,3201))
            np.testing.assert_allclose(data["mean"],train.mean(axis=0,keepdims=True))
            np.testing.assert_allclose(data["std"],train.std(axis=0,keepdims=True))

    def test_invalid_mode_does_not_create_normalization(self):
        with self.assertRaises(ValueError):self.normalize("invalid")
        self.assertFalse((self.output/"normalization_900rpm.npz").exists())


if __name__=="__main__":unittest.main()
