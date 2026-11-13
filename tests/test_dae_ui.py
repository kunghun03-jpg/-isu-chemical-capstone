"""Synthetic UI integration only; test RPM declarations do not describe real models."""

from io import BytesIO
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import streamlit as st
from streamlit.testing.v1 import AppTest

from tests import test_dae as synthetic
from vibration.dae_bundle import convert_legacy_bundle
from vibration.dae_inference import SavedDAEBundleInference

ROOT = Path(__file__).resolve().parents[1]


class DAEUITests(unittest.TestCase):
    def setUp(self):
        synthetic.DAEInferenceTests.setUp(self)
        self.bundle = self.folder.with_name(self.folder.name + "_ui_bundle")
        self.addCleanup(lambda: shutil.rmtree(self.bundle, ignore_errors=True))
        self.env = patch.dict(os.environ, {
            "ISU_CHEMICAL_DAE_BUNDLE": str(self.bundle),
            "ISU_CHEMICAL_HISTORY_DB": str(self.folder / "ui_history.db"),
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(st.cache_resource.clear)
        st.cache_resource.clear()

    def convert(self, known=False):
        kwargs = {"supported_rpms": [900], "rpm_evidence": "synthetic UI fixture only"} if known else {}
        convert_legacy_bundle(self.folder, self.bundle, model_version="synthetic-ui-v1", **kwargs)

    def upload(self):
        stream = BytesIO(self.raw)
        stream.name = "synthetic.mat"
        return patch("streamlit.file_uploader", return_value=stream)

    def app(self):
        return AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()

    def text(self, app):
        return "\n".join(e.value for kind in (app.caption, app.info, app.warning) for e in kind)

    def classic_engine(self, models=("autoencoder", "ocsvm", "iforest")):
        engine = MagicMock()
        engine.models = models
        engine.manifest = {"artifacts": {f"{name}.bin": "a" * 64 for name in models},
                           "models": {name: {"artifact": f"{name}.bin"} for name in models}}
        result = {"models": {name: {
            "file_score": 1., "file_threshold": 2., "predicted_anomaly": 0,
            "segment_threshold": 2.,
            "segments": [{"segment_id": i, "score": 1.} for i in range(8)],
        } for name in models}}
        engine.predict_bytes.return_value = result
        return engine, result

    def test_missing_and_unknown_rpm_disable_only_dae(self):
        with self.upload(), patch("vibration.dae_ui.predict_dae", side_effect=AssertionError("blocked")):
            app = self.app()
            self.assertFalse(app.button[0].disabled)  # Existing default all-three flow.
            app.selectbox[1].select("DAE").run()
            self.assertIn("모델 미준비", self.text(app))
            self.assertTrue(app.button[0].disabled)
            self.assertNotIn("dae_result", app.session_state)
        self.convert()
        with self.upload(), patch("vibration.dae_ui.predict_dae", side_effect=AssertionError("blocked")):
            app = self.app()
            app.selectbox[1].select("DAE").run()
            self.assertIn("RPM 적용 범위 미확인", self.text(app))
            self.assertTrue(app.button[0].disabled)
            self.assertFalse(app.exception)

    def test_ready_dae_exact_result_separate_display_and_no_db_save(self):
        self.convert(known=True)
        expected = SavedDAEBundleInference(self.bundle).predict_bytes(self.raw, rpm=900)
        with self.upload():
            app = self.app()
        with self.upload(), patch("vibration.inference.SavedModelInference", side_effect=AssertionError("classic")), patch(
            "vibration.web_inference.ch3_log_spectra", side_effect=AssertionError("classic FFT")
        ):
            app.selectbox[1].select("DAE").run()
            self.assertFalse(app.exception)
            self.assertFalse(app.button[0].disabled)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            actual = app.session_state["dae_result"]
            self.assertEqual(actual["mse"], expected["mse"])
            np.testing.assert_array_equal(actual["reconstructed_spectrum"], expected["reconstructed_spectrum"])
            self.assertNotIn("analysis_result", app.session_state)
            self.assertEqual(app.dataframe[0].value.iloc[0]["모델"], "DAE")
            self.assertIn("파일 전체 Welch", self.text(app))
            self.assertFalse(any(b.label == "결과 저장" for b in app.button))
            self.assertFalse(any(e.value == "상세 분석" for e in app.subheader))
            app.selectbox[0].select(1200).run()
            self.assertIn("선택 RPM 미지원", self.text(app))
            self.assertTrue(app.button[0].disabled)
            self.assertNotIn("dae_result", app.session_state)

    def test_all_keeps_classic_results_when_dae_missing_unknown_or_fails(self):
        for mode in ("missing", "unknown", "failure"):
            with self.subTest(mode=mode):
                if self.bundle.exists():
                    shutil.rmtree(self.bundle)
                if mode != "missing":
                    self.convert(known=mode == "failure")
                engine, expected = self.classic_engine()
                with self.upload(), patch("vibration.inference.SavedModelInference", return_value=engine), patch(
                    "vibration.dae_ui.predict_dae", side_effect=ValueError("synthetic failure")
                ) as dae_call:
                    st.cache_resource.clear()
                    app = self.app()
                    app.button[0].click().run()
                    self.assertFalse(app.exception)
                    self.assertFalse(app.error)
                    self.assertEqual(app.session_state["analysis_result"], expected)
                    self.assertEqual(len(app.dataframe[0].value), 3)
                    self.assertTrue(any(b.label == "결과 저장" for b in app.button))
                    self.assertEqual(dae_call.call_count, int(mode == "failure"))
                    engine.predict_bytes.assert_called_once_with(self.raw, rpm=900, filename="synthetic.mat")

    def test_individual_classic_selection_does_not_run_dae(self):
        self.convert(known=True)
        for label, name in (("AE", "autoencoder"), ("OCSVM", "ocsvm"), ("IF", "iforest")):
            with self.subTest(label=label):
                engine, expected = self.classic_engine((name,))
                with self.upload(), patch("vibration.inference.SavedModelInference", return_value=engine), patch(
                    "vibration.dae_ui.predict_dae", side_effect=AssertionError("DAE must not run")
                ):
                    st.cache_resource.clear()
                    app = self.app()
                    app.selectbox[1].select(label).run()
                    app.button[0].click().run()
                    self.assertFalse(app.exception)
                    self.assertEqual(app.session_state["analysis_result"], expected)
                    self.assertNotIn("dae_result", app.session_state)

    def test_all_displays_dae_separately_and_selection_clears_result(self):
        self.convert(known=True)
        engine, expected = self.classic_engine()
        with self.upload(), patch("vibration.inference.SavedModelInference", return_value=engine):
            app = self.app()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["analysis_result"], expected)
            self.assertIn("dae_result", app.session_state)
            self.assertEqual(set(app.session_state["model_versions"]), set(engine.models))
            self.assertTrue(any(e.value == "DAE 분석 결과" for e in app.subheader))
            self.assertTrue(any(e.value == "상세 분석" for e in app.subheader))
            self.assertEqual(len(app.dataframe[0].value), 3)
            app.selectbox[1].select("AE").run()
            self.assertNotIn("dae_result", app.session_state)
            self.assertNotIn("analysis_result", app.session_state)

    def test_invalid_dae_upload_does_not_infer(self):
        self.convert(known=True)
        with self.upload():
            app = self.app()
            app.selectbox[1].select("DAE").run()
        bad = BytesIO(b"invalid MAT"); bad.name = "bad.mat"
        with patch("streamlit.file_uploader", return_value=bad), patch(
            "vibration.dae_ui.predict_dae", side_effect=AssertionError("invalid upload")
        ):
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertTrue(app.button[0].disabled)
            self.assertNotIn("dae_result", app.session_state)

    def test_removed_or_tampered_bundle_clears_previous_dae_result(self):
        self.convert(known=True)
        with self.upload():
            app = self.app()
            app.selectbox[1].select("DAE").run()
            app.button[0].click().run()
            self.assertIn("dae_result", app.session_state)
            with (self.bundle / "normalization.npz").open("ab") as stream:
                stream.write(b"tamper")
            app.run()
            self.assertIn("모델 검증 실패", self.text(app))
            self.assertTrue(app.button[0].disabled)
            self.assertNotIn("dae_result", app.session_state)
            shutil.rmtree(self.bundle)
            app.run()
            self.assertIn("모델 미준비", self.text(app))
            self.assertNotIn("dae_result", app.session_state)


if __name__ == "__main__":
    unittest.main()
