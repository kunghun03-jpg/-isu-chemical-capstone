"""Synthetic DAE readiness/diagnostics; public RPM blocking must remain intact."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tests import test_dae_ui as fixture
from vibration.dae_readiness import probe
from vibration.dae_bundle import manifest_id,sha256_file
from vibration.dae_inference import SavedDAEBundleInference
from scripts.validate_dae_bundle_mat import validate,provenance


class SafeDAETests(unittest.TestCase):
    setUp=fixture.DAEUITests.setUp
    convert=fixture.DAEUITests.convert

    def test_path_file_hash_and_rpm_states(self):
        self.assertEqual(probe(900,'')['code'],'path_missing')
        self.bundle.mkdir()
        self.assertEqual(probe(900,self.bundle)['code'],'file_missing')
        self.bundle.rmdir();self.convert()
        self.assertEqual(probe(900,self.bundle)['code'],'rpm_unconfirmed')
        self.assertEqual(probe(900,self.bundle)['label'],'지원 범위 미확인')
        with (self.bundle/'autoencoder_model.pth').open('ab') as f:f.write(b'corrupt')
        self.assertEqual(probe(900,self.bundle)['code'],'hash_error')

    def test_valid_hash_wrong_weights_is_load_error(self):
        import torch
        self.convert()
        torch.save({'wrong':torch.zeros(1)},self.bundle/'autoencoder_model.pth')
        p=self.bundle/'manifest.json';m=json.loads(p.read_text());m['artifacts']['autoencoder_model.pth']=sha256_file(self.bundle/'autoencoder_model.pth');m['bundle_id']=manifest_id(m);p.write_text(json.dumps(m))
        self.assertEqual(probe(900,self.bundle)['code'],'model_load_error')

    def test_explicit_offline_diagnostic_does_not_enable_public_prediction(self):
        self.convert();mat=self.folder/'signal.mat';mat.write_bytes(self.raw)
        with tempfile.TemporaryDirectory() as root:
            cases=[{'path':str(mat),'rpm':900}]
            blocked=validate(self.bundle,cases,Path(root)/'blocked','synthetic')
            self.assertEqual(blocked['rows'][0]['status'],'blocked')
            diag=validate(self.bundle,cases,Path(root)/'diag','synthetic',True)
            row=diag['rows'][0]
            self.assertEqual(row['status'],'diagnostic_only_rpm_unconfirmed')
            self.assertEqual(row['prediction'],'');self.assertFalse(row['public_inference_allowed'])
            self.assertIsInstance(row['diagnostic_mse'],float)
            self.assertFalse(diag['performance_validated']);self.assertEqual(diag['changed_artifacts'],[])
        with self.assertRaises(ValueError):SavedDAEBundleInference(self.bundle).predict_bytes(self.raw,rpm=900)

    def test_supported_fixture_only_public_result_matches_existing(self):
        self.convert(known=True);mat=self.folder/'signal.mat';mat.write_bytes(self.raw)
        expected=SavedDAEBundleInference(self.bundle).predict_bytes(self.raw,rpm=900)
        with tempfile.TemporaryDirectory() as root:
            report=validate(self.bundle,[{'path':str(mat),'rpm':900}],Path(root)/'ready','synthetic')
            row=report['rows'][0];self.assertEqual(row['prediction'],expected['prediction']);self.assertEqual(row['diagnostic_mse'],expected['mse'])
        self.assertEqual(probe(1200,self.bundle)['code'],'rpm_unsupported')

    def test_metadata_dates_and_rpms_are_observations_not_support(self):
        with tempfile.TemporaryDirectory() as root:
            np.savez(Path(root)/'normal_dataset_normalized.npz',X_train=np.zeros((2,800)),X_val=np.ones((1,800)),mean=np.zeros(800),std=np.ones(800),train_groups=np.array(['20260912_900RPM_Normal1','20260912_1200RPM_Normal2']),val_groups=np.array(['20260916_900RPM_Normal1']))
            result=provenance(root)
            self.assertIsNone(result['supported_rpms']);self.assertEqual(result['split_independence'],'unconfirmed')
            meta=result['metadata']['normal_dataset_normalized.npz']['observations']
            self.assertEqual(meta['train_groups']['date_counts'],{'20260912':2})
            self.assertEqual(meta['val_groups']['date_counts'],{'20260916':1})

    def test_v3_unknown_rpm_suppresses_score_threshold_and_preserves_ae(self):
        from copy import deepcopy
        from vibration.dashboard_v3 import references
        from vibration.dashboard_v2 import DashboardConfig
        self.convert()
        ae={'models':{'autoencoder':{'file_score':1.,'file_threshold':.5,'predicted_anomaly':1}},'bundle_id':'synthetic'}
        before=deepcopy(ae)
        row=references(self.raw,900,DashboardConfig('autoencoder',self.folder/'missing'),ae)[1]
        self.assertEqual(row['status'],'지원 범위 미확인')
        self.assertIsNone(row['score']);self.assertIsNone(row['threshold']);self.assertNotIn(row['decision'],('정상','이상'))
        self.assertEqual(ae,before)
