"""Synthetic V2 workflow, V1/API regression and existing SQLite compatibility."""

import os
import sqlite3
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import streamlit as st
import numpy as np
from scipy.io import savemat
from streamlit.testing.v1 import AppTest

from tests.test_web_env import synthetic_bundles
from vibration.dashboard_v2 import configuration, analyze
from vibration.history import HistoryStore
from vibration.inference import SavedModelInference
from vibration.inference_bundle import manifest_id

ROOT = Path(__file__).resolve().parents[1]


class DashboardV2Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="dashboard_v2_")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.bundles = self.root / "bundles"
        synthetic_bundles(self.bundles)
        for rpm in (900, 1200, 1800):
            path = self.bundles / f"{rpm}rpm/manifest.json"
            manifest = json.loads(path.read_text())
            manifest.update(source_experiment="synthetic_dashboard", experiment_mode="synthetic",
                            limitations=["synthetic tests only"],
                            reference_file_metrics={m: {} for m in manifest["models"]})
            manifest["bundle_id"] = manifest_id(manifest)
            path.write_text(json.dumps(manifest))
        self.path = self.root / "measurement.mat"
        savemat(self.path, {"signals": np.random.default_rng(8).normal(size=(6, 65536)), "fs": 25600})
        self.raw = self.path.read_bytes()
        self.db = self.root / "existing_history.sqlite3"
        environment = patch.dict(os.environ, {
            "ISU_CHEMICAL_V2_MODEL": "autoencoder",
            "ISU_CHEMICAL_V2_BUNDLE_ROOT": str(self.bundles),
            "ISU_CHEMICAL_HISTORY_DB": str(self.db),
            "ISU_CHEMICAL_DAE_BUNDLE": str(self.root / "missing_dae"),
        })
        environment.start()
        self.addCleanup(environment.stop)
        self.addCleanup(st.cache_resource.clear)
        st.cache_resource.clear()

    def upload(self, raw=None, name="measurement.mat"):
        stream = BytesIO(self.raw if raw is None else raw)
        stream.name = name
        return patch("streamlit.file_uploader", return_value=stream)

    def app(self):
        return AppTest.from_file(str(ROOT / "streamlit_app_v2.py"), default_timeout=30).run()

    def test_initial_dashboard_has_no_model_selector_or_fabricated_result(self):
        with patch("streamlit.file_uploader", return_value=None), patch(
            "vibration.dashboard_v2.SavedModelInference", side_effect=AssertionError("no inference")
        ):
            app = self.app()
            self.assertFalse(app.exception)
            self.assertTrue(app.button(key="v2_run").disabled)
            self.assertEqual([box.label for box in app.selectbox], ["측정 RPM"])
            self.assertNotIn("v2_result", app.session_state)
            self.assertTrue(any(h.value == "분석 대기" for h in app.subheader))

    def test_three_backends_match_api_and_presentation_v2_requires_ae(self):
        for model in ("autoencoder", "ocsvm", "iforest"):
            with self.subTest(model=model), patch.dict(os.environ, {"ISU_CHEMICAL_V2_MODEL": model}):
                config = configuration()
                expected = SavedModelInference(self.bundles / "900rpm", models=(model,)).predict_bytes(
                    self.raw, rpm=900, filename="measurement.mat")
                result, versions = analyze(self.raw, 900, "measurement.mat", config)
                self.assertEqual(result["models"], expected["models"])
                self.assertEqual(set(versions), {model})
                with self.upload(), patch("vibration.dae_inference.SavedDAEBundleInference",
                                          side_effect=AssertionError("DAE forbidden")):
                    app = self.app()
                    if model != 'autoencoder':
                        self.assertTrue(app.error)
                        self.assertNotIn('v2_result', app.session_state)
                        continue
                    app.button(key="v2_run").click().run()
                    self.assertFalse(app.exception)
                    self.assertFalse(app.error)
                    self.assertEqual(app.session_state["v2_result"]["models"], expected["models"])
                    self.assertTrue(any(m.label == "이상 점수" for m in app.metric))

    def test_save_deduplicates_existing_multimodel_history_and_schema_stays_compatible(self):
        engine = SavedModelInference(self.bundles / "900rpm")
        old_result = engine.predict_bytes(self.raw, rpm=900, filename="old_app.mat")
        old_versions = {m: "sha256:" + engine.manifest["artifacts"][engine.manifest["models"][m]["artifact"]]
                        for m in engine.models}
        store = HistoryStore(self.db)
        store.save(old_result, old_versions)
        with sqlite3.connect(self.db) as connection:
            before = connection.execute("SELECT type,name,sql FROM sqlite_master ORDER BY type,name").fetchall()
        with self.upload():
            app = self.app()
            app.button(key="v2_run").click().run()
            app.button(key="v2_save").click().run()
            self.assertFalse(app.exception)
            app.button(key="v2_save").click().run()
            self.assertEqual(len({r["analysis_id"] for r in store.search()}), 2)
            self.assertEqual(len(store.search()), 4)
            app.radio(key="v2_page").set_value("분석 이력").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe[0].value), 2)
            self.assertTrue(any(d.label == "조회 결과 CSV 다운로드" for d in app.get("download_button")))
            self.assertFalse(any("모델" in s.label for s in app.selectbox))
        with sqlite3.connect(self.db) as connection:
            after = connection.execute("SELECT type,name,sql FROM sqlite_master ORDER BY type,name").fetchall()
        before_map = {(r[0], r[1]): r[2] for r in before}
        after_map = {(r[0], r[1]): r[2] for r in after}
        self.assertTrue(all(after_map[k] == sql for k, sql in before_map.items()))
        self.assertEqual({r[1] for r in after if r[0] == 'table'} - {r[1] for r in before if r[0] == 'table'}, {'candidate_reports'})
        self.assertEqual(len(store.detail(1)["models"]), 3)

    def test_input_rpm_config_changes_clear_stale_results(self):
        with self.upload():
            app = self.app()
            app.button(key="v2_run").click().run()
            self.assertIn("v2_result", app.session_state)
            app.selectbox(key="v2_rpm").select(1200).run()
            self.assertNotIn("v2_result", app.session_state)
            app.button(key="v2_run").click().run()
            self.assertEqual(app.session_state["v2_result"]["input"]["rpm"], 1200)
            with patch.dict(os.environ, {"ISU_CHEMICAL_V2_MODEL": "ocsvm"}):
                app.run()
                self.assertNotIn("v2_result", app.session_state)
        with self.upload(raw=b"broken mat"):
            app.run()
            self.assertTrue(app.error)
            self.assertTrue(app.button(key="v2_run").disabled)
            self.assertNotIn("v2_result", app.session_state)

    def test_missing_or_tampered_bundle_and_dae_configuration_never_generate_judgment(self):
        with self.upload():
            app = self.app()
            app.button(key="v2_run").click().run()
            self.assertIn("v2_result", app.session_state)
            with (self.bundles / "900rpm/normalization.npz").open("ab") as stream:
                stream.write(b"tamper")
            app.run()
            self.assertNotIn("v2_result", app.session_state)
            app.button(key="v2_run").click().run()
            self.assertTrue(app.error)
            self.assertNotIn("v2_result", app.session_state)
            (self.bundles / "900rpm/normalization.npz").unlink()
            app.run()
            self.assertTrue(app.button(key="v2_run").disabled)
            self.assertNotIn("v2_result", app.session_state)
        with patch.dict(os.environ, {"ISU_CHEMICAL_V2_MODEL": "dae"}):
            with self.assertRaises(ValueError):
                configuration()
            app = self.app()
            self.assertTrue(app.error)
            self.assertFalse(app.exception)
        # A malformed manifest must not silently select a different model.
        (self.bundles / "900rpm/manifest.json").write_text("not json")
        with self.upload():
            app = self.app()
            self.assertTrue(app.button(key="v2_run").disabled)

    def test_v1_three_model_ui_keeps_same_inference_results(self):
        original = SavedModelInference
        def load(folder, models=("autoencoder", "ocsvm", "iforest")):
            return original(self.bundles / Path(folder).name, models=models)
        expected = original(self.bundles / "900rpm").predict_bytes(self.raw, rpm=900, filename="measurement.mat")
        with self.upload(), patch("vibration.inference.SavedModelInference", side_effect=load):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(app.session_state["analysis_result"]["models"], expected["models"])


if __name__ == "__main__":
    unittest.main()
