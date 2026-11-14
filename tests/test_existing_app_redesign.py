"""Same-entry-point dashboard regression using synthetic signals/artifacts only."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from streamlit.testing.v1 import AppTest
from tests import test_dashboard_v2 as fixture
from vibration.v3_presentation import overview_values,model_tile_values,display_signals
from vibration.dashboard_v3 import analyze_selected
from vibration.dashboard_v2 import configuration
from scripts.analyze_fault_signatures import analyze_signal,fft_amplitude

ROOT=Path(__file__).resolve().parents[1]

class ExistingAppRedesignTests(unittest.TestCase):
    setUp=fixture.DashboardV2Tests.setUp
    upload=fixture.DashboardV2Tests.upload

    def app(self):
        return AppTest.from_file(str(ROOT/'streamlit_app_v3.py'),default_timeout=40).run()

    def test_same_app_four_real_navigation_destinations_and_no_placeholder_data(self):
        with patch('streamlit.file_uploader',return_value=None):
            app=self.app();self.assertFalse(app.exception)
            self.assertEqual(app.radio(key='v2_page').options,['분석하기','분석 이력','모델 정보','사용 가이드'])
            self.assertTrue(app.button(key='v3_run').disabled)
            self.assertNotIn('v2_result',app.session_state)
            for page in ('분석 이력','모델 정보','사용 가이드','분석하기'):
                app.radio(key='v2_page').set_value(page).run();self.assertFalse(app.exception)
            self.assertFalse(any('0.73' in m.value for m in app.markdown))
            self.assertTrue(any('사용 중지 / 검증 필요' in m.value for m in app.markdown))

    def test_overview_uses_exact_ae_only_and_does_not_mutate_result(self):
        result,_=analyze_selected(self.raw,900,'measurement.mat',configuration())
        before=deepcopy(result);v=overview_values(900,result)
        self.assertEqual(v['score'],result['models']['autoencoder']['file_score'])
        self.assertEqual(v['threshold'],result['models']['autoencoder']['file_threshold'])
        self.assertEqual(v['decision'],'이상 징후')
        normal=deepcopy(result);normal['models']['autoencoder']['predicted_anomaly']=0
        self.assertEqual(overview_values(900,normal)['decision'],'정상')
        self.assertEqual(overview_values(900,None,'failure')['decision'],'판정 불가')
        self.assertIsNone(overview_values(900,None)['score']);self.assertEqual(result,before)

    def test_model_cards_keep_raw_directions_and_blocked_dae_has_no_numbers(self):
        rows=[dict(model='ocsvm',status='완료',decision='이상',score=0.,threshold=-.37),dict(model='dae',status='지원 범위 미확인',decision='미연결',score=None,threshold=None)]
        before=deepcopy(rows);cards={r['model']:r for r in model_tile_values(rows)}
        self.assertEqual(cards['ocsvm']['score'],0.);self.assertEqual(cards['ocsvm']['threshold'],-.37)
        self.assertEqual(cards['dae']['status'],'사용 중지 / 검증 필요');self.assertIsNone(cards['dae']['score'])
        self.assertIsNone(cards['autoencoder']['score']);self.assertEqual(rows,before)

    def test_envelope_display_matches_existing_eda_math_without_input_change(self):
        t=np.arange(65536)/25600
        x=(1+.5*np.sin(2*np.pi*30*t))*np.sin(2*np.pi*3500*t)
        before=x.copy();display=display_signals(x,25600.,1800);existing=analyze_signal(x,25600.,1800)
        np.testing.assert_array_equal(display['envelope_amplitude'],existing['envelope_amplitude'])
        f,a=fft_amplitude(display['envelope_signal'],25600.)
        np.testing.assert_array_equal(a[f<=1000],existing['envelope_amplitude'])
        self.assertAlmostEqual(np.sqrt(np.mean((display['envelope_signal']-display['envelope_signal'].mean())**2)),existing['features']['envelope_rms_ac'])
        self.assertEqual(len(display['envelope_signal']),65536-5120)
        np.testing.assert_array_equal(x,before)
        x32=x.astype(np.float32)
        shown=display_signals(x32,25600.,1800)
        _,spectrum32=fft_amplitude(shown['envelope_signal'],25600.)
        expected32=analyze_signal(x32,25600.,1800)
        np.testing.assert_array_equal(spectrum32[:len(expected32['envelope_amplitude'])],expected32['envelope_amplitude'])

    def test_analysis_exact_before_after_save_and_download_with_tabs(self):
        expected,_=analyze_selected(self.raw,900,'measurement.mat',configuration())
        with self.upload():
            app=self.app();app.button(key='v3_run').click().run();self.assertFalse(app.exception)
            self.assertEqual(app.session_state['v2_result']['models'],expected['models'])
            for label in ('시간 영역 파형','주파수 스펙트럼','포락 분석','회전 성분'):
                self.assertIn(label,[t.label for t in app.tabs])
            app.button(key='v3_save').click().run();self.assertFalse(app.exception)
            self.assertTrue(any(d.label=='분석 결과 JSON' for d in app.get('download_button')))
            app.radio(key='v2_page').set_value('분석 이력').run();self.assertFalse(app.exception)
            self.assertTrue(any(d.label=='조회 결과 CSV 다운로드' for d in app.get('download_button')))

    def test_display_failure_preserves_ae_and_saving(self):
        with self.upload(),patch('vibration.v3_presentation.display_signals',side_effect=ValueError('synthetic chart failure')):
            app=self.app();app.button(key='v3_run').click().run();self.assertFalse(app.exception)
            result=deepcopy(app.session_state['v2_result'])
            self.assertTrue(any('AE 결과는 유지됩니다' in w.value for w in app.warning))
            app.button(key='v3_save').click().run();self.assertFalse(app.exception)
            self.assertEqual(app.session_state['v2_result'],result)

    def test_ae_normal_still_skips_rf_and_input_change_clears_cards(self):
        original=analyze_selected
        def normal(raw,rpm,filename,config):
            result,versions=original(raw,rpm,filename,config)
            item=result['models']['autoencoder'];item['file_threshold']=item['file_score']+1;item['predicted_anomaly']=0
            return result,versions
        with self.upload(),patch('streamlit_app_v3.analyze_selected',side_effect=normal),patch('vibration.candidate_analysis.CandidateEngine',side_effect=AssertionError('RF must be skipped')):
            # AppTest executes the entry point as __main__; patch its imported backend instead.
            with patch('vibration.dashboard_v3.analyze_selected',side_effect=normal):
                app=self.app();app.button(key='v3_run').click().run();self.assertFalse(app.exception)
                self.assertEqual(app.session_state['v2_candidate']['status'],'skipped_normal')
                app.selectbox(key='v2_rpm').set_value(1200).run()
                self.assertNotIn('v2_result',app.session_state);self.assertNotIn('v3_rows',app.session_state)

if __name__=='__main__':unittest.main()
