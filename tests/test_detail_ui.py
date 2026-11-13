"""실제 900RPM 사례의 상세 그래프·CLI 일치·누락 처리 검사. 재학습 없음."""

from copy import deepcopy
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import streamlit as st
from streamlit.testing.v1 import AppTest

from vibration.detail_ui import spectrum_frame, segment_frame
from vibration.inference import SavedModelInference
from vibration.web_inference import preview_mat
from tests.test_streamlit_app import ROOT, upload

CASES = [
    ("data/20260912/20260912_900RPM/Normal1/Vib/20260912_133556_Vib.mat", (0, 0, 0)),
    (
        "data/20260912/20260912_900RPM/Looseness0.1/Vib/20260912_170916_Vib.mat",
        (0, 0, 0),
    ),
    (
        "data/20260919/20260919_900RPM/Looeness0.1/Vib/20260919_152236_Vib.mat",
        (1, 1, 1),
    ),
]


class DetailTests(unittest.TestCase):
    def test_spectrum_reuses_exact_preprocessing_and_missing_threshold_is_not_replaced(
        self,
    ):
        engine = SavedModelInference(
            ROOT / "outputs/inference_bundles/date_prelim_20261106/900rpm"
        )
        raw = (ROOT / CASES[1][0]).read_bytes()
        preview = preview_mat(raw)
        spectra, _ = engine.preprocess(raw)
        np.testing.assert_array_equal(preview["log_spectra"], spectra)
        frame = spectrum_frame(preview)
        self.assertEqual(
            (frame["주파수 (Hz)"].iloc[0], frame["주파수 (Hz)"].iloc[-1]), (0, 10000)
        )
        for index in range(8):
            np.testing.assert_array_equal(frame[f"구간 {index + 1}"], spectra[index, 0])
        item = engine.predict_bytes(raw, rpm=900)["models"]["iforest"]
        data = segment_frame(item)
        self.assertEqual(int((data["구간 점수"] > item["segment_threshold"]).sum()), 1)
        self.assertEqual(item["predicted_anomaly"], 0)
        self.assertEqual(data["구간 점수"].mean(), item["file_score"])
        missing = deepcopy(item)
        del missing["segment_threshold"]
        with self.assertRaisesRegex(ValueError, "구간 임계값"):
            segment_frame(missing)
        missing = deepcopy(item)
        missing["segments"].pop()
        with self.assertRaisesRegex(ValueError, "8개"):
            segment_frame(missing)

    def test_three_real_cases_screen_and_cli_match(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ, {"ISU_CHEMICAL_HISTORY_DB": str(Path(folder) / "test.sqlite3")}
        ):
            for relative, expected in CASES:
                path = ROOT / relative
                with patch("streamlit.file_uploader", return_value=upload(path)):
                    app = AppTest.from_file(
                        str(ROOT / "streamlit_app.py"), default_timeout=60
                    ).run()
                    app.button[0].click().run()
                    self.assertFalse(app.exception)
                    self.assertFalse(app.error)
                    self.assertEqual(
                        len(app.warning), 1
                    )  # 기존 상단 성능 한계 경고만 표시
                    result = app.session_state["analysis_result"]
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
                            "900",
                            "--model",
                            "all",
                        ],
                        cwd=ROOT,
                        env=dict(
                            os.environ,
                            PYTHONIOENCODING="utf-8",
                            PYTHONDONTWRITEBYTECODE="1",
                        ),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        check=True,
                    )
                    original = json.loads(cli.stdout)
                    self.assertEqual(result["models"], original["models"])
                    self.assertEqual(
                        tuple(
                            result["models"][name]["predicted_anomaly"]
                            for name in ("autoencoder", "ocsvm", "iforest")
                        ),
                        expected,
                    )
                    self.assertTrue(
                        any(
                            "정규화 전 log1p" in element.value
                            for element in app.markdown
                        )
                    )
                    self.assertEqual(len(app.get("vega_lite_chart")), 6)
                    for item in result["models"].values():
                        data = segment_frame(item)
                        self.assertEqual(data["구간 점수"].mean(), item["file_score"])
                        np.testing.assert_array_equal(
                            data["저장 구간 임계값"],
                            np.repeat(item["segment_threshold"], 8),
                        )
        st.cache_resource.clear()


if __name__ == "__main__":
    unittest.main()
