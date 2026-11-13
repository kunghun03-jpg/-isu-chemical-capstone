"""Dashboard layout/pages consume actual results; synthetic fixtures only."""
import unittest
from unittest.mock import patch

import numpy as np

from tests import test_dashboard_v2 as fixtures
from vibration.dashboard_view import envelope_points, feature_rows, rotation_frame


class DashboardRedesignTests(unittest.TestCase):
    setUp=fixtures.DashboardV2Tests.setUp
    app=fixtures.DashboardV2Tests.app
    upload=fixtures.DashboardV2Tests.upload

    def test_all_five_pages_work_and_settings_navigation(self):
        with self.upload():
            app=self.app()
            self.assertFalse(app.exception)
            self.assertEqual(app.radio(key='v2_page').options,
                ['홈 대시보드','데이터 업로드','분석 이력','통계 및 리포트','시스템 설정'])
            app.radio(key='v2_page').set_value('데이터 업로드').run()
            self.assertFalse(app.button(key='v2_run').disabled)
            app.button(key='v2_run').click().run()
            app.button(key='v2_save').click().run()
            app.button(key='v2_go_history').click().run()
            self.assertEqual(app.radio(key='v2_page').value,'분석 이력')
            self.assertFalse(app.exception)
            app.radio(key='v2_page').set_value('통계 및 리포트').run()
            self.assertFalse(app.exception)
            self.assertTrue(any(m.label=='저장된 분석 건' and str(m.value)=='1' for m in app.metric))
            self.assertTrue(any(d.label=='전체 리포트 CSV 다운로드' for d in app.get('download_button')))
            app.button(key='v2_open_settings').click().run()
            self.assertEqual(app.radio(key='v2_page').value,'시스템 설정')
            self.assertFalse(app.exception)
            self.assertTrue(any('읽기 전용' in c.value for c in app.caption))
            app.button(key='v2_refresh_settings').click().run()
            self.assertFalse(app.exception)

    def test_details_retain_full_wave_fft_ae_and_candidate_evidence(self):
        with self.upload():
            app=self.app()
            app.button(key='v2_run').click().run()
            original=app.session_state['v2_result']['models']
            app.toggle(key='v2_show_details').set_value(True).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state['v2_result']['models'],original)
            self.assertTrue(any(t.label=='AE 입력 스펙트럼' for t in app.tabs))
            self.assertTrue(any(e.label=='구간별 분석 근거' for e in app.expander))

    def test_mobile_navigation_callback_opens_actual_page(self):
        with self.upload():
            app=self.app()
            app.button(key='v2_nav_1').click().run()
            self.assertEqual(app.radio(key='v2_page').value,'데이터 업로드')
            self.assertFalse(app.exception)
            self.assertFalse(app.button(key='v2_run').disabled)

    def test_plot_sampling_retains_extrema_actual_samples_and_time(self):
        x=np.zeros(65536)
        x[321]=23
        x[44333]=-19
        time=np.arange(x.size)/25600.
        frame=envelope_points(time,x,bins=100)
        self.assertIn(23,frame['CH3 진폭'].tolist())
        self.assertIn(-19,frame['CH3 진폭'].tolist())
        self.assertLessEqual(len(frame),202)
        self.assertTrue(frame['시간 (초)'].is_monotonic_increasing)
        for t,v in zip(frame['시간 (초)'],frame['CH3 진폭']):
            self.assertEqual(x[round(t*25600)],v)
        self.assertEqual(x[321],23)

    def test_real_feature_values_units_and_correct_rotation_frequencies(self):
        report={'features':{'rms_ac':1.25,'one_x_amplitude':.3,'two_x_amplitude':.1,'energy_high':4}}
        values=feature_rows(report)
        self.assertEqual(values[0][1],1.25)
        self.assertEqual(values[5][1],4)
        self.assertTrue(all('mm/s' not in unit and 'm/s' not in unit for _,_,unit in values))
        rotation=rotation_frame(report,1200)
        self.assertEqual(rotation['회전 성분'].tolist(),['1X · 20Hz','2X · 40Hz'])
        self.assertEqual(rotation['진폭'].tolist(),[.3,.1])
        self.assertTrue(all(value is None for _,value,_ in feature_rows(None)))

    def test_uploaded_name_is_html_escaped_in_summary(self):
        with self.upload(name='<img src=x onerror=alert(1)>.mat'):
            app=self.app()
            app.button(key='v2_run').click().run()
            self.assertFalse(app.exception)
            summary=[m.value for m in app.markdown if 'summary-title' in m.value]
            self.assertTrue(any('&lt;img' in m for m in summary))
            self.assertFalse(any('<img src=x' in m for m in summary))


if __name__=='__main__':
    unittest.main()
