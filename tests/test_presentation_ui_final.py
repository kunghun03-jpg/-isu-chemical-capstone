"""Presentation rendering regression; no training and no real performance claims."""
from copy import deepcopy
import unittest
from unittest.mock import patch, MagicMock
from tests import test_dashboard_v2 as fixtures
from vibration.dashboard_view import ae_status, app_status


class FinalUITests(unittest.TestCase):
    setUp=fixtures.DashboardV2Tests.setUp
    app=fixtures.DashboardV2Tests.app
    upload=fixtures.DashboardV2Tests.upload

    def test_ring_matches_pending_normal_abnormal_without_mutation(self):
        with self.upload():
            app=self.app()
            self.assertTrue(any('data-ae-status="대기"' in m.value for m in app.markdown))
            app.button(key='v2_run').click().run()
            result=deepcopy(app.session_state['v2_result'])
            for flag, label in ((0,'정상'),(1,'이상')):
                synthetic=deepcopy(result)
                synthetic['models']['autoencoder']['predicted_anomaly']=flag
                before=deepcopy(synthetic)
                self.assertEqual(ae_status(synthetic),label)
                with patch('vibration.dashboard_view.st') as view:
                    from vibration.dashboard_view import ae_card
                    view.columns.return_value=(MagicMock(),MagicMock())
                    ae_card(synthetic)
                    html=view.markdown.call_args.args[0]
                    self.assertIn('data-ae-status="'+label+'"',html)
                    self.assertIn('aria-label="AE 판정: '+label+'"',html)
                self.assertEqual(synthetic,before)
            self.assertEqual(app.session_state['v2_result'],result)
            self.assertTrue(any('분석 완료 · 결과 저장 가능' in m.value for m in app.markdown))
            self.assertTrue(any(s.value=='2. 결함 후보 분석' for s in app.subheader))

    def test_system_status_does_not_claim_loading_before_inference(self):
        self.assertIn('분석 불가', app_status(None,None)[0])
        self.assertIn('로딩 미검증',app_status({'files_present':True},None)[0])
        self.assertFalse(app_status({'files_present':True},None)[1])
        self.assertIn('분석 완료',app_status({'files_present':True},{'models':{}})[0])
        self.assertIn('분석 실패',app_status({'files_present':True},None,'error')[0])

    def test_failure_is_shown_and_cleared_when_input_changes(self):
        with self.upload():
            app=self.app()
            with patch('vibration.dashboard_v2.analyze',side_effect=ValueError('synthetic failure')):
                app.button(key='v2_run').click().run()
            self.assertTrue(any('분석 실패 · 입력·환경 확인 필요' in m.value for m in app.markdown))
            app.selectbox(key='v2_rpm').set_value(1200).run()
            self.assertNotIn('v2_analysis_error', app.session_state)
            self.assertTrue(any('분석 대기' in m.value for m in app.markdown))
