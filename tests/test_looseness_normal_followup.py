"""Synthetic cache/provenance and source-error audit; no user-data claims."""
import csv
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scipy.io import savemat
from tests import test_dashboard_v2 as fixture
from scripts.followup_looseness_normal import run
from scripts.analyze_looseness_date_shift import process
from scripts.validate_v2_integration import write_csv
from vibration.inference import SavedModelInference

class FollowupTests(unittest.TestCase):
    setUp=fixture.DashboardV2Tests.setUp

    def mat(self,condition='Normal1',date='20260912',seed=1):
        p=self.root/'data'/date/f'{date}_900RPM'/condition/'Vib'/f'{date}_120000_Vib.mat';p.parent.mkdir(parents=True,exist_ok=True)
        savemat(p,{'signals':np.random.default_rng(seed).normal(size=(6,65536)),'fs':25600})
        return dict(path=str(p),date=date,rpm=900,condition=condition)

    def source(self):
        src=self.root/'old_loose';src.mkdir()
        case=self.mat('Looseness0.05',seed=4)
        engine=SavedModelInference(self.bundles/'900rpm',models=('autoencoder',))
        row=process(case,engine);error={**case,'path':str(Path(case['path']).with_name('20260912_120001_Vib.mat')),'status':'error','sha256':'','read_ok':False,'error':'ValueError: original fixture error'}
        write_csv(src/'file_results.csv',[row,error]);(src/'analysis_report.json').write_text(json.dumps({'completed':True,'conclusions_ready':False,'discovered_looseness':2}))
        return src

    def test_reuse_verified_cached_normal_without_ae_or_db_and_source_errors_retained(self):
        source=self.source();case=self.mat();engine=SavedModelInference(self.bundles/'900rpm',models=('autoencoder',));r=process(case,engine)
        cached=self.root/'cached';cached.mkdir();write_csv(cached/'normal_results.csv',[r])
        original=(source/'file_results.csv').read_bytes()
        with patch('scripts.followup_looseness_normal.process',side_effect=AssertionError('must reuse AE')),patch('sqlite3.connect',side_effect=AssertionError('DB forbidden')),patch('vibration.candidate_analysis.CandidateEngine',side_effect=AssertionError('RF forbidden')):
            report=run(source,self.root/'data',self.bundles,self.root/'followup',[cached],'synthetic',True)
        self.assertEqual(report['normal_ae_modes'],{'reused':1});self.assertEqual(report['normal_ae_inference_calls'],0)
        self.assertEqual(report['recorded_looseness_errors'][0]['recorded_error'],'ValueError: original fixture error')
        self.assertEqual((source/'file_results.csv').read_bytes(),original);self.assertFalse(report['source_conclusions_ready'])
        self.assertFalse(report['independent_test_performance_validated']);self.assertTrue(report['comparison_valid'])
        self.assertEqual(report['normal_by_date_rpm'][0]['success_files'],1)

    def test_changed_model_proof_not_reused_and_new_result_cached_on_next_run(self):
        source=self.source();case=self.mat();engine=SavedModelInference(self.bundles/'900rpm',models=('autoencoder',));r=process(case,engine)
        cached=self.root/'cached';cached.mkdir();bad=deepcopy(r);bad['bundle_id']='wrong';write_csv(cached/'normal_results.csv',[bad])
        report=run(source,self.root/'data',self.bundles,self.root/'new',[cached],'synthetic',True,True)
        self.assertEqual(report['normal_ae_inference_calls'],1)
        self.assertEqual(report['normal_ae_modes'],{'inferred':1})
        with patch('scripts.followup_looseness_normal.process',side_effect=AssertionError('repeat inference')):
            second=run(source,self.root/'data',self.bundles,self.root/'next',[],'synthetic',True)
        self.assertEqual(second['normal_ae_modes'],{'reused':1})
        self.assertEqual(second['normal_ae_inference_calls'],0)

    def test_conflicting_verified_cached_scores_are_not_silently_used(self):
        source=self.source();case=self.mat();engine=SavedModelInference(self.bundles/'900rpm',models=('autoencoder',));r=process(case,engine)
        for i in (1,2):
            folder=self.root/f'cache{i}';folder.mkdir();row=deepcopy(r)
            if i==2:row['ae_score']+=1;row['ae_anomaly']=row['ae_score']>row['ae_threshold']
            write_csv(folder/'normal_results.csv',[row])
        with patch('scripts.followup_looseness_normal.process',side_effect=AssertionError('must not hide conflicts')):
            report=run(source,self.root/'data',self.bundles,self.root/'report',[],'synthetic',True)
        self.assertFalse(report['comparison_valid']);self.assertEqual(report['normal_statuses'],{'error':1})

    def test_integration_cache_requires_manifest_normalization_and_weight_hashes(self):
        from scripts.followup_looseness_normal import ae_proof,coerce
        case=self.mat();engine=SavedModelInference(self.bundles/'900rpm',models=('autoencoder',));r=process(case,engine);r.pop('bundle_id')
        self.assertFalse(ae_proof(r,{},engine,self.bundles/'900rpm'))
        hashes={str(p):__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in (self.bundles/'900rpm').iterdir() if p.is_file()}
        self.assertTrue(ae_proof(r,{'input_artifact_sha256':hashes},engine,self.bundles/'900rpm'))

    def test_original_failed_looseness_only_rechecked_without_rewriting_counts(self):
        source=self.source()
        with (source/'file_results.csv').open(encoding='utf-8-sig') as f:old=list(csv.DictReader(f))
        bad=Path(old[1]['path']);savemat(bad,{'signals':np.ones((6,65536)),'fs':25600})
        self.mat()
        report=run(source,self.root/'data',self.bundles,self.root/'followup',[],'synthetic',True)
        self.assertEqual(report['looseness_statuses'],{'ok':1,'error':1})
        self.assertEqual(report['recorded_looseness_errors'][0]['current_ae_status'],'passed')
        self.assertIsNone(report['recorded_looseness_errors'][0]['same_recorded_bytes'])
        self.assertFalse(report['source_conclusions_ready'])
        self.assertFalse(report['all_requested_normal_validated'])

    def test_unverified_existing_cache_blocks_repeat_ae_by_default(self):
        source=self.source();case=self.mat();engine=SavedModelInference(self.bundles/'900rpm',models=('autoencoder',));r=process(case,engine)
        cached=self.root/'cached';cached.mkdir();r['bundle_id']='unverified';write_csv(cached/'normal_results.csv',[r])
        with patch('scripts.followup_looseness_normal.process',side_effect=AssertionError('do not repeat existing AE')):
            report=run(source,self.root/'data',self.bundles,self.root/'followup',[cached],'synthetic',True)
        self.assertEqual(report['normal_ae_inference_calls'],0)
        self.assertEqual(report['normal_statuses'],{'error':1})
