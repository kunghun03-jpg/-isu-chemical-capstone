"""Synthetic presentation workflow; no real trained model or diagnosis validation."""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import sqlite3
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

from tests import test_dashboard_v2 as fixtures
from vibration.candidate_analysis import UNCLEAR, candidate_analysis, safe_candidate_analysis
from vibration.candidate_history import read_candidate
from vibration.dashboard_v2 import analyze, configuration
from vibration.history import HistoryStore


class PresentationTests(unittest.TestCase):
    # Reuse fixture construction without inheriting/duplicating the older tests.
    setUp = fixtures.DashboardV2Tests.setUp
    app = fixtures.DashboardV2Tests.app
    upload = fixtures.DashboardV2Tests.upload

    def test_same_screen_ae_candidate_features_fft_and_history_round_trip(self):
        expected, _ = analyze(self.raw, 900, 'IR.mat', configuration())
        raw_checksum = hashlib.sha256(self.raw).hexdigest()
        with self.upload(name='IR.mat'):
            app = self.app()
            app.button(key='v2_run').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(app.session_state['v2_result']['models'], expected['models'])
            self.assertEqual(app.session_state['v2_candidate']['candidate'], UNCLEAR)
            self.assertEqual(app.session_state['v2_candidate']['input_sha256'], raw_checksum)
            self.assertTrue(any(m.label == '결함 후보' and m.value == UNCLEAR for m in app.metric))
            self.assertTrue(any(s.value == '1. AE 이상 탐지 결과' for s in app.subheader))
            self.assertTrue(any(s.value == '2. 결함 후보 분석' for s in app.subheader))
            self.assertTrue(any(t.label == 'FFT 진폭 스펙트럼' for t in app.tabs))
            app.radio(key='v2_fft_range').set_value('저주파 0–200Hz').run()
            self.assertFalse(app.exception)
            saved_report = deepcopy(app.session_state['v2_candidate'])
            app.button(key='v2_save').click().run()
            app.button(key='v2_save').click().run()
            store = HistoryStore(self.db)
            self.assertEqual(len(store.search()), 1)
            self.assertEqual(read_candidate(store, 1), saved_report)
            app.radio(key='v2_page').set_value('분석 이력').run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(any(m.label == '결함 후보' and m.value == UNCLEAR for m in app.metric))
        self.assertEqual(hashlib.sha256(self.raw).hexdigest(), raw_checksum)

    def test_filename_labels_do_not_produce_diagnoses(self):
        reports = []
        for filename in ('Normal.mat', 'IR.mat', 'OR.mat', 'Roller.mat', 'Unbalance.mat', 'Looseness.mat'):
            with self.subTest(filename=filename), self.upload(name=filename):
                app = self.app()
                app.button(key='v2_run').click().run()
                self.assertFalse(app.exception)
                reports.append(app.session_state['v2_candidate'])
        self.assertTrue(all(r == reports[0] for r in reports))

    def test_candidate_failure_preserves_ae_and_can_be_saved_as_unavailable(self):
        expected, _ = analyze(self.raw, 900, 'measurement.mat', configuration())
        with self.upload(), patch('vibration.candidate_analysis.analyze_signal', side_effect=ValueError('numeric fixture')):
            app = self.app()
            app.button(key='v2_run').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['v2_result']['models'], expected['models'])
            self.assertEqual(app.session_state['v2_candidate']['status'], 'unavailable')
            app.button(key='v2_save').click().run()
            self.assertEqual(read_candidate(HistoryStore(self.db), 1)['status'], 'unavailable')

    def test_file_and_rpm_changes_clear_both_results_and_fft(self):
        with self.upload():
            app = self.app()
            app.button(key='v2_run').click().run()
            app.selectbox(key='v2_rpm').select(1200).run()
            for key in ('v2_result', 'v2_candidate', 'v2_signal_details'):
                self.assertNotIn(key, app.session_state)
            app.button(key='v2_run').click().run()
            self.assertEqual(app.session_state['v2_candidate']['rpm'], 1200)
        with self.upload(raw=b'bad MAT'):
            app.run()
            self.assertTrue(app.error)
            self.assertNotIn('v2_candidate', app.session_state)

    def test_feature_values_match_existing_eda_and_zero_signal_is_not_fault(self):
        from scripts.analyze_fault_signatures import analyze_signal
        from vibration.web_inference import preview_mat
        result, spectrum = candidate_analysis(self.raw, 900)
        expected = analyze_signal(preview_mat(self.raw)['signal'], 25600., 900)
        for key, value in result['features'].items():
            self.assertEqual(value, expected['features'][key])
        np.testing.assert_array_equal(spectrum['amplitude'], expected['amplitude'])
        stream = BytesIO()
        savemat(stream, {'signals': np.zeros((6, 65536)), 'fs': 25600})
        zero, _ = candidate_analysis(stream.getvalue(), 900)
        self.assertEqual(zero['candidate'], UNCLEAR)
        self.assertIsNone(zero['features']['crest_factor_ac'])
        json.dumps(zero, allow_nan=False)

    def test_existing_history_read_does_not_add_table_or_change_records(self):
        result, versions = analyze(self.raw, 900, 'legacy.mat', configuration())
        store = HistoryStore(self.db)
        store.save(result, versions)
        before = self.db.read_bytes()
        self.assertIsNone(read_candidate(store, 1))
        self.assertEqual(self.db.read_bytes(), before)
        with self.upload():
            app = self.app()
            app.radio(key='v2_page').set_value('분석 이력').run()
            self.assertFalse(app.exception)
            self.assertTrue(any('후보 근거가 없습니다' in i.value for i in app.info))

    def test_candidate_insert_failure_rolls_back_ae_and_legacy_rows_survive(self):
        result, versions = analyze(self.raw, 900, 'transaction.mat', configuration())
        report, _ = candidate_analysis(self.raw, 900)
        store = HistoryStore(self.db)
        with patch('vibration.candidate_history.insert_candidate', side_effect=sqlite3.OperationalError('fixture')):
            with self.assertRaises(sqlite3.OperationalError):
                store.save(result, versions, candidate_report=report)
        self.assertEqual(store.search(), [])
        store.save(result, versions)
        old = store.detail(1)
        store.save(result, versions, candidate_report=report)
        self.assertEqual(store.detail(1), old)
        self.assertEqual(read_candidate(store, 1), report)
        altered = deepcopy(report)
        altered['features']['rms_ac'] += 1
        with self.assertRaises(ValueError):
            store.save(result, versions, candidate_report=altered)
        self.assertEqual(read_candidate(store, 1), report)

    def test_wrong_input_hash_and_nonfinite_features_cannot_be_saved(self):
        result, versions = analyze(self.raw, 900, 'measurement.mat', configuration())
        report, _ = candidate_analysis(self.raw, 900)
        for field in ('hash', 'nonfinite'):
            broken = deepcopy(report)
            if field == 'hash':
                broken['input_sha256'] = '0' * 64
            else:
                broken['features']['rms_ac'] = np.inf
            with self.assertRaises(ValueError):
                HistoryStore(self.db).save(result, versions, candidate_report=broken)


if __name__ == '__main__':
    unittest.main()
