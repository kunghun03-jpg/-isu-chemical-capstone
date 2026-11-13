"""이력 원자성·중복 방지·검색·프로세스 재시작·실제 화면 연결 검사."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
from pathlib import Path
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from vibration.history import HistoryStore, history_csv
from tests.test_streamlit_app import ROOT, sample, upload


def example():
    return {
        "analyzed_at_utc": "2026-11-06T23:00:00+00:00",
        "input": {"name": "=Normal1.mat", "sha256": "a" * 64, "rpm": 900, "channel": 3},
        "bundle_id": "b" * 64,
        "source_experiment": "date_prelim_20261106",
        "models": {
            name: {
                "file_score": score,
                "file_threshold": 0.5,
                "predicted_anomaly": int(score > 0.5),
            }
            for name, score in (("autoencoder", 0.2), ("ocsvm", 0.8), ("iforest", 0.4))
        },
    }


VERSIONS = {
    name: "sha256:" + str(i) * 64
    for i, name in enumerate(("autoencoder", "ocsvm", "iforest"), 1)
}


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "history.sqlite3"
        self.store = HistoryStore(self.path)

    def tearDown(self):
        self.directory.cleanup()

    def test_group_filters_detail_csv_and_repeated_save(self):
        result = example()
        first, created = self.store.save(result, VERSIONS)
        self.assertTrue(created)
        self.assertEqual(self.store.save(deepcopy(result), VERSIONS), (first, False))
        self.assertEqual(len(self.store.detail(first)["models"]), 3)
        self.assertEqual(
            len(self.store.search(start="2026-11-07", end="2026-11-07", rpm=900)), 3
        )
        self.assertFalse(self.store.search(end="2026-11-06"))
        self.assertFalse(self.store.search(rpm=1200))
        self.assertEqual(len(self.store.search(model="ocsvm", decision=1)), 1)
        self.assertFalse(self.store.search(model="autoencoder", decision=1))
        self.assertFalse(self.store.search(model="' OR 1=1 --"))
        csv = history_csv(self.store.search()).decode("utf-8-sig")
        self.assertIn("'=Normal1.mat", csv)
        self.assertIn(VERSIONS["ocsvm"], csv)
        # 같은 파일도 새 분석 시각으로 재분석하면 별도의 분석 건입니다.
        result["analyzed_at_utc"] = "2026-11-07T00:00:00+00:00"
        self.assertTrue(self.store.save(result, VERSIONS)[1])

    def test_atomic_rollback_and_concurrent_duplicate(self):
        self.store.search()
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "CREATE TRIGGER reject_result BEFORE INSERT ON model_results BEGIN SELECT RAISE(ABORT, 'test failure'); END"
            )
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.save(example(), VERSIONS)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0], 0
            )
            connection.execute("DROP TRIGGER reject_result")
        with ThreadPoolExecutor(max_workers=4) as pool:
            saved = list(
                pool.map(
                    lambda _: HistoryStore(self.path).save(example(), VERSIONS),
                    range(8),
                )
            )
        self.assertEqual(sum(created for _, created in saved), 1)
        self.assertEqual(len({key for key, _ in saved}), 1)
        self.assertEqual(len(self.store.search()), 3)

    def test_separate_process_restart_preserves_and_deduplicates(self):
        worker = "from vibration.history import HistoryStore; import json,sys; p=json.load(sys.stdin); s=HistoryStore(p['path']); print(json.dumps([s.save(p['result'],p['versions']),len(s.search())]))"
        payload = json.dumps(
            {"path": str(self.path), "result": example(), "versions": VERSIONS}
        )
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")

        def run():
            return json.loads(
                subprocess.run(
                    [sys.executable, "-B", "-X", "utf8", "-c", worker],
                    input=payload,
                    text=True,
                    encoding="utf-8",
                    capture_output=True,
                    check=True,
                    cwd=ROOT,
                    env=env,
                ).stdout
            )

        first, restarted = run(), run()
        self.assertTrue(first[0][1])
        self.assertFalse(restarted[0][1])
        self.assertEqual(first[0][0], restarted[0][0])
        self.assertEqual(restarted[1], 3)

    def test_web_save_repeat_refresh_restart_and_error(self):
        with patch.dict(os.environ, {"ISU_CHEMICAL_HISTORY_DB": str(self.path)}), patch(
            "streamlit.file_uploader", return_value=upload(sample())
        ):
            app = AppTest.from_file(
                str(ROOT / "streamlit_app.py"), default_timeout=60
            ).run()
            app.button[0].click().run()
            self.assertFalse(self.store.search())
            app.button(key="save_analysis").click().run()
            self.assertTrue(app.success)
            app.button(key="save_analysis").click().run()
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(self.store.search()), 3)
            self.assertEqual(len(self.store.detail(1)["models"]), 3)
            app.selectbox(key="history_model").select("OCSVM").run()
            self.assertEqual(len(app.dataframe[1].value), 1)
            self.assertEqual(len(app.dataframe[2].value), 3)
            with patch(
                "vibration.history.HistoryStore.save",
                side_effect=sqlite3.OperationalError("disk full"),
            ):
                app.button(key="save_analysis").click().run()
                self.assertIn("저장에 실패", app.error[0].value)
                self.assertFalse(app.exception)
        # 새 브라우저/앱 세션에서도 업로드 없이 DB의 이력을 읽습니다.
        with patch.dict(os.environ, {"ISU_CHEMICAL_HISTORY_DB": str(self.path)}), patch(
            "streamlit.file_uploader", return_value=None
        ):
            restarted = AppTest.from_file(
                str(ROOT / "streamlit_app.py"), default_timeout=60
            ).run()
            self.assertFalse(restarted.exception)
            self.assertEqual(len(restarted.dataframe[0].value), 3)
        st.cache_resource.clear()


if __name__ == "__main__":
    unittest.main()
