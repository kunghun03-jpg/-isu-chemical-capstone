"""WSL verifier plumbing tests using synthetic artifacts and reference fixtures."""

import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.validate_dae_wsl import validate
from tests import test_dae as synthetic
from vibration.dae_model import ConvAutoencoder


class DAEWSLTests(unittest.TestCase):
    setUp = synthetic.DAEInferenceTests.setUp

    def prepare(self):
        temp = tempfile.TemporaryDirectory(prefix="dae_wsl_verifier_")
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        code = root / "reference"
        code.mkdir()
        (code / "model.py").write_text("from torch import nn\n" + inspect.getsource(ConvAutoencoder))
        (code / "web_preprocessing.py").write_text('''
from io import BytesIO
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch
def preprocess_uploaded_mat(upload):
    d=loadmat(BytesIO(upload.getvalue()))
    f,p=welch(np.asarray(d['signals'][2],dtype=np.float64),fs=25600.,
              window='hann',nperseg=8192,noverlap=4096,nfft=8192)
    return np.log10(np.sqrt(p)[(f>=0)&(f<10000)].reshape(800,4).mean(axis=1)+1e-12).astype(np.float32)
''')
        (code / "dae_inference.py").write_text('''
from pathlib import Path
import numpy as np
import torch
from model import ConvAutoencoder
from web_preprocessing import preprocess_uploaded_mat
BASE_DIR=Path('.')
def run_dae_inference(upload):
    with np.load(BASE_DIR/'normal_dataset_normalized.npz') as d:
        normalized=(preprocess_uploaded_mat(upload)-d['mean'])/d['std']
    with np.load(BASE_DIR/'normal_threshold.npz') as d:
        threshold=float(d['threshold'])
    x=torch.tensor(normalized,dtype=torch.float32).reshape(1,1,800)
    m=ConvAutoencoder().eval()
    m.load_state_dict(torch.load(BASE_DIR/'autoencoder_model.pth',map_location='cpu',weights_only=True))
    with torch.no_grad():
        y=m(x)
        mse=torch.mean((x-y)**2,dim=(1,2)).item()
    return dict(prediction='이상' if mse>threshold else '정상',mse=mse,threshold=threshold,
                input_spectrum=normalized,reconstructed_spectrum=y.squeeze().numpy())
''')
        return root, code

    def test_synthetic_and_provided_mat_parity_reports(self):
        root, code = self.prepare()
        np.savez(self.folder / "normal_dataset_normalized.npz", mean=self.mean, std=self.std,
                 X_train=np.zeros((2, 800)), train_groups=np.array(["20260912_900RPM"]))
        np.savez(self.folder / "band_statistics.npz", band_names=["low", "mid", "high"],
                 band_means=[1., 2., 3.], band_stds=[1., 1., 1.])
        report = validate(self.folder, code, root / "synthetic")
        self.assertEqual(report["status"], "compatibility_checks_passed_rpm_unconfirmed")
        self.assertTrue(report["source_unchanged"])
        self.assertEqual(report["input_kind"], "synthetic_mat")
        self.assertFalse(report["performance_validated"])
        self.assertEqual(report["rpm_investigation"]["metadata_rpm_observations"], {"train_groups": [900]})
        self.assertIsNone(report["supported_rpms"])
        supplied = root / "input.mat"; supplied.write_bytes(self.raw)
        report = validate(self.folder, code, root / "provided", supplied)
        self.assertEqual(report["input_kind"], "provided_mat")
        self.assertIsNone(json.loads((root / "provided/bundle/manifest.json").read_text())["supported_rpms"])

    def test_failure_report_missing_files_and_no_overwrite(self):
        root, code = self.prepare()
        output = root / "failed"
        with patch("scripts.validate_dae_wsl.subprocess.run", side_effect=RuntimeError("reference failure")):
            with self.assertRaisesRegex(RuntimeError, "reference failure"):
                validate(self.folder, code, output)
        self.assertEqual(json.loads((output / "report.json").read_text())["status"], "failed")
        with self.assertRaises(FileExistsError):
            validate(self.folder, code, output)
        with self.assertRaises(FileNotFoundError):
            validate(root / "missing", code, root / "absent")
        self.assertFalse((root / "absent").exists())


if __name__ == "__main__":
    unittest.main()
