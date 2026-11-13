"""웹 화면 흐름과 기존 CLI의 점수 일치 검사. 실제 학습 없음."""

from io import BytesIO
from pathlib import Path
import json
import os
import subprocess
import sys
import unittest
import tempfile
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from vibration.web_inference import preview_mat

ROOT = Path(__file__).resolve().parents[1]


def sample(rpm=900):
    folder = ROOT / f"data/20260919/20260919_{rpm}RPM/Normal1/Vib"
    return next(folder.glob("*.mat"))


def upload(path):
    data = BytesIO(path.read_bytes())
    data.name = path.name
    return data


class StreamlitTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.environment = patch.dict(
            os.environ,
            {
                "ISU_CHEMICAL_HISTORY_DB": str(
                    Path(self.directory.name) / "history.sqlite3"
                )
            },
        )
        self.environment.start()

    def tearDown(self):
        st.cache_resource.clear()
        self.environment.stop()
        self.directory.cleanup()

    def test_upload_preview_analysis_and_cli_parity(self):
        # AppTest로 실제 화면 버튼까지 실행하고 세 RPM에서 CLI JSON과 비교합니다.
        for rpm in (900, 1200, 1800):
            path = sample(rpm)
            with patch("streamlit.file_uploader", return_value=upload(path)):
                app = AppTest.from_file(
                    str(ROOT / "streamlit_app.py"), default_timeout=60
                ).run()
                app.selectbox[0].select(rpm).run()
                self.assertFalse(app.exception)
                self.assertEqual(app.metric[0].value, "25,600 Hz")
                self.assertEqual(app.metric[1].value, "65,536")
                self.assertNotIn("analysis_result", app.session_state)
                # 웹 경로에서도 학습·역전파·임계값 재계산이 호출되면 실패합니다.
                with patch(
                    "sklearn.svm.OneClassSVM.fit", side_effect=AssertionError("fit")
                ), patch(
                    "sklearn.ensemble.IsolationForest.fit",
                    side_effect=AssertionError("fit"),
                ), patch(
                    "torch.Tensor.backward", side_effect=AssertionError("backward")
                ), patch(
                    "numpy.quantile", side_effect=AssertionError("Q95")
                ):
                    app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                web = app.session_state["analysis_result"]
                env = dict(
                    os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1"
                )
                cli = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-X",
                        "utf8",
                        "-m",
                        "scripts.infer_mat",
                        "--file",
                        str(path),
                        "--rpm",
                        str(rpm),
                        "--model",
                        "all",
                    ],
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=True,
                )
                expected = json.loads(cli.stdout)
                self.assertEqual(web["models"], expected["models"])
                self.assertEqual(web["input"]["sha256"], expected["input"]["sha256"])
                self.assertEqual(len(app.dataframe[0].value), 3)
                for choice, model in (
                    ("AE", "autoencoder"),
                    ("OCSVM", "ocsvm"),
                    ("IF", "iforest"),
                ):
                    app.selectbox[1].select(choice).run()
                    self.assertNotIn("analysis_result", app.session_state)
                    app.button[0].click().run()
                    self.assertFalse(app.exception)
                    actual = app.session_state["analysis_result"]["models"]
                    self.assertEqual(set(actual), {model})
                    self.assertEqual(actual[model], expected["models"][model])

    def test_empty_invalid_upload_and_missing_model(self):
        with patch("streamlit.file_uploader", return_value=None):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
            self.assertTrue(app.button[0].disabled)
        bad = BytesIO(b"not a mat")
        bad.name = "broken.mat"
        with patch("streamlit.file_uploader", return_value=bad):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
            self.assertTrue(app.error)
            self.assertTrue(app.button[0].disabled)
            self.assertFalse(app.exception)
        with patch("streamlit.file_uploader", return_value=upload(sample())), patch(
            "vibration.inference.SavedModelInference",
            side_effect=FileNotFoundError("missing"),
        ):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertIn("저장 모델", app.error[0].value)
            self.assertNotIn("analysis_result", app.session_state)

    def test_preview_matches_ch3_and_filename_does_not_decide(self):
        from scipy.io import loadmat
        from vibration.inference import SavedModelInference
        from vibration.web_inference import upload_name
        import numpy as np

        raw = sample().read_bytes()
        preview = preview_mat(raw)
        np.testing.assert_array_equal(
            preview["signal"], loadmat(BytesIO(raw))["signals"][2]
        )
        self.assertEqual(preview["duration"], 2.56)
        self.assertEqual(upload_name("../../Normal2/IR.mat"), "IR.mat")
        engine = SavedModelInference(
            ROOT / "outputs/inference_bundles/date_prelim_20261106/900rpm"
        )
        self.assertEqual(
            engine.predict_bytes(raw, rpm=900, filename="Normal1.mat")["models"],
            engine.predict_bytes(raw, rpm=900, filename="IR.mat")["models"],
        )


if __name__ == "__main__":
    unittest.main()
