"""Synthetic CSV/UI evidence only. No real Normal1/Normal2 result is fabricated."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from tests import test_dashboard_v2 as fixture
from vibration.normal_comparison_ui import load_comparison, load_exclusions, comparison_long, SUMMARY_NAME

ROOT=Path(__file__).resolve().parents[1]

class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)

    def source(self,frame=None):
        if frame is None:
            frame=pd.DataFrame([dict(rpm=rpm,condition=condition,date=date,rms_ac_median=value,
                one_x_amplitude_median=value/2,two_x_amplitude_median=value/4,three_x_amplitude_median=value/8)
                for date,value in (('20260912',1.),('20260916',9.)) for rpm in (900,1200,1800) for condition in ('Normal1','Normal2')])
        p=self.folder/SUMMARY_NAME;frame.to_csv(p,index=False,encoding='utf-8-sig');return p

    def ui(self):
        return AppTest.from_string('from vibration.normal_comparison_ui import render_normal_comparison\nrender_normal_comparison()',default_timeout=20).run()

    def test_long_schema_exact_values_and_source_bytes_unchanged(self):
        p=self.source();before=hashlib.sha256(p.read_bytes()).hexdigest();frame=load_comparison(p)
        self.assertEqual(len(frame),12);self.assertEqual(set(frame['rms_median']),{1.,9.})
        self.assertEqual(set(comparison_long(frame)['feature']),{'rms_median','one_x_median','two_x_median','three_x_median'})
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),before)

    def test_paired_schema_and_case_insensitive_numeric_aliases(self):
        p=self.source(pd.DataFrame([{'RPM':1200,'Normal1_rms_median':2,'Normal2_rms_median':3,
             'Normal1_1X_median':4,'Normal2_1X_median':5,'Normal1_2X_median':6,'Normal2_2X_median':7,
             'Normal1_3X_median':8,'Normal2_3X_median':9}]))
        frame=load_comparison(p)
        self.assertEqual(frame['condition'].tolist(),['Normal1','Normal2'])
        self.assertEqual(frame['three_x_median'].tolist(),[8,9]);self.assertEqual(len(frame),2)
        self.assertNotIn(900,frame['rpm'].tolist())

    def test_invalid_values_missing_columns_and_duplicate_aggregates_rejected(self):
        original=pd.read_csv(self.source())
        for value in (np.nan,np.inf,-1.,'not-a-number'):
            with self.subTest(value=value):
                frame=original.copy();frame['rms_ac_median']=frame['rms_ac_median'].astype(object);frame.at[0,'rms_ac_median']=value
                with self.assertRaises(ValueError):load_comparison(self.source(frame))
        with self.assertRaises(ValueError):load_comparison(self.source(original.drop(columns='three_x_amplitude_median')))
        with self.assertRaises(ValueError):load_comparison(self.source(pd.concat([original,original.iloc[[0]]])))
        original.loc[0,'rpm']=1500
        with self.assertRaises(ValueError):load_comparison(self.source(original))

    def test_missing_both_files_has_guidance_and_no_fake_results(self):
        with patch('vibration.normal_comparison_ui.DEFAULT_FOLDER',self.folder):app=self.ui()
        self.assertFalse(app.exception);self.assertEqual(len(app.dataframe),0)
        self.assertTrue(any('정상 조건 비교 자료 없음' in i.value for i in app.info))
        self.assertTrue(any('제외 파일 자료 없음' in i.value for i in app.info))
        self.assertTrue(any('Run 독립성' in w.value for w in app.warning))

    def test_date_selection_displays_each_date_without_pooling(self):
        self.source()
        with patch('vibration.normal_comparison_ui.DEFAULT_FOLDER',self.folder):
            app=self.ui();self.assertFalse(app.exception)
            shown=app.dataframe[0].value;self.assertEqual(set(shown['RMS 중앙값']),{1.});self.assertEqual(len(shown),6)
            self.assertEqual(len(app.get('vega_lite_chart')),4)
            app.selectbox(key='normal_compare_date').set_value('20260916').run();self.assertFalse(app.exception)
            self.assertEqual(set(app.dataframe[0].value['RMS 중앙값']),{9.})

    def test_exclusions_visible_even_when_summary_is_broken(self):
        (self.folder/SUMMARY_NAME).write_text('rpm,condition\n900,Normal1\n')
        pd.DataFrame([dict(path=r'C:\data\bad.mat',rpm=1200,condition='Normal2',reason='nonfinite_ch3')]).to_csv(self.folder/'excluded_files.csv',index=False)
        with patch('vibration.normal_comparison_ui.DEFAULT_FOLDER',self.folder):app=self.ui()
        self.assertFalse(app.exception);self.assertTrue(any('표시하지 못했습니다' in w.value for w in app.warning))
        self.assertEqual(app.dataframe[0].value.iloc[0]['파일명'],'bad.mat')
        self.assertEqual(app.dataframe[0].value.iloc[0]['제외 사유'],'nonfinite_ch3')

    def test_exclusion_name_path_and_exact_reason_are_preserved(self):
        p=self.folder/'excluded_files.csv'
        pd.DataFrame([dict(filename='bad.mat',path='/data/bad.mat',rpm='',condition='Normal1',reason='invalid',error='NaN/Inf found')]).to_csv(p,index=False)
        frame=load_exclusions(p);self.assertEqual(frame.iloc[0]['RPM'],'미기록')
        self.assertEqual(frame.iloc[0]['파일명'],'bad.mat');self.assertEqual(frame.iloc[0]['파일'],'/data/bad.mat')
        self.assertEqual(frame.iloc[0]['오류 상세'],'NaN/Inf found')

    def test_empty_comparison_csv_keeps_quality_section_available(self):
        original=pd.read_csv(self.source()).iloc[:0]
        self.source(original)
        with patch('vibration.normal_comparison_ui.DEFAULT_FOLDER',self.folder):app=self.ui()
        self.assertFalse(app.exception)
        self.assertTrue(any('표시할 집계 행이 없습니다' in i.value for i in app.info))
        self.assertTrue(any('데이터 품질' in s.value for s in app.subheader))
        self.assertEqual(len(app.get('vega_lite_chart')),0)

    def test_empty_and_bad_quality_csv_never_crashes(self):
        for text in ('filename,rpm,condition,reason\n','filename,rpm\nbad.mat,900\n',''):
            with self.subTest(text=text),patch('vibration.normal_comparison_ui.DEFAULT_FOLDER',self.folder):
                (self.folder/'excluded_files.csv').write_text(text)
                app=self.ui();self.assertFalse(app.exception)

    def test_missing_condition_not_zero_filled_and_explicit_run_not_pooled(self):
        original=pd.read_csv(self.source()).iloc[:1].copy()
        original['run']='A';other=original.copy();other['run']='B';other['rms_ac_median']=20
        self.source(pd.concat([original,other]))
        with patch('vibration.normal_comparison_ui.DEFAULT_FOLDER',self.folder):
            app=self.ui();self.assertFalse(app.exception)
            self.assertEqual(len(app.dataframe[0].value),1)
            self.assertTrue(any('누락 집계' in i.value for i in app.info))
            app.selectbox(key='normal_compare_run').set_value('B').run();self.assertFalse(app.exception)
            self.assertEqual(app.dataframe[0].value.iloc[0]['RMS 중앙값'],20.)

class ExistingAppComparisonTests(unittest.TestCase):
    setUp=fixture.DashboardV2Tests.setUp
    upload=fixture.DashboardV2Tests.upload

    def test_model_information_integrates_evidence_without_inference_or_db_changes(self):
        with patch('vibration.normal_comparison_ui.DEFAULT_FOLDER',self.root/'missing'),patch('vibration.normal_comparison_ui.load_comparison',side_effect=AssertionError('missing input must not be invented')):
            app=AppTest.from_file(str(ROOT/'streamlit_app_v3.py'),default_timeout=40).run()
            app.radio(key='v2_page').set_value('모델 정보').run();self.assertFalse(app.exception)
            self.assertTrue(any('정상 조건 검증' in s.value for s in app.subheader))
            self.assertTrue(any('Run 독립성' in w.value for w in app.warning))
            app.radio(key='v2_page').set_value('분석하기').run();self.assertFalse(app.exception)
            self.assertNotIn('v2_result',app.session_state)
