"""Synthetic numerical fixtures + mocked version identities; not real cross-version validation."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tests import test_dae_ui as fixture
from scripts.check_dae_runtime_compat import execute
from vibration.dae_bundle import load_manifest,runtime_versions
from vibration.dae_inference import SavedDAEBundleInference
from vibration.dae_runtime_compat import compare_records,read_json


class RuntimeCompatibilityTests(unittest.TestCase):
    setUp=fixture.DAEUITests.setUp
    convert=fixture.DAEUITests.convert

    def prepare(self):
        self.convert();self.mat=self.folder/'test.mat';self.mat.write_bytes(self.raw)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.out=Path(self.tmp.name)
        self.ref=self.out/'reference';execute('export',self.bundle,self.mat,self.ref,'actual')
        # Version identities are mocked only to exercise the policy. All MAT/weights are synthetic.
        self.target={**runtime_versions(),'numpy':'mock-validated-numpy'}
        self.dest=self.out/'target'
        with patch('scripts.check_dae_runtime_compat.runtime_versions',return_value=self.target):
            report=execute('verify',self.bundle,self.mat,self.dest,'actual',self.ref/'reference.json')
        self.cert=self.dest/'runtime_compatibility.json'
        return report

    def test_verified_bundle_bound_pair_allowed_but_rpm_still_blocked(self):
        report=self.prepare()
        self.assertFalse(report['performance_validated']);self.assertFalse(report['rpm_support_validated'])
        self.assertTrue(all(v==0 for v in report['max_abs_errors'].values()))
        with patch.dict(os.environ,{'ISU_CHEMICAL_DAE_RUNTIME_REPORT':str(self.cert)}),patch('vibration.dae_bundle.runtime_versions',return_value=self.target),patch('vibration.dae_runtime_compat.runtime_versions',return_value=self.target):
            self.assertEqual(load_manifest(self.bundle)['supported_rpms'],None)
            with self.assertRaises(ValueError):SavedDAEBundleInference(self.bundle).predict_bytes(self.raw,rpm=900)

    def test_unverified_pair_blocked_and_expected_actual_logged(self):
        self.convert();actual={**runtime_versions(),'torch':'mock-unverified-torch'}
        with patch.dict(os.environ,{'ISU_CHEMICAL_DAE_RUNTIME_REPORT':''}),patch('vibration.dae_bundle.runtime_versions',return_value=actual),self.assertLogs('vibration.dae_bundle',level='WARNING') as logs:
            with self.assertRaises(ValueError):load_manifest(self.bundle)
        self.assertIn('expected=',logs.output[0]);self.assertIn('mock-unverified-torch',logs.output[0])

    def test_reference_requires_origin_and_same_mat(self):
        self.prepare()
        with patch('scripts.check_dae_runtime_compat.runtime_versions',return_value=self.target):
            with self.assertRaises(ValueError):execute('export',self.bundle,self.mat,self.out/'wrong-source','actual')
        changed=self.folder/'other.mat'
        from tests.test_dae import mat_bytes
        changed.write_bytes(mat_bytes(np.ones((6,65536))))
        with self.assertRaises(ValueError):execute('verify',self.bundle,changed,self.out/'wrong-input','actual',self.ref/'reference.json')

    def test_numeric_mismatch_and_tampered_evidence_rejected(self):
        self.prepare();target=read_json(self.dest/'target.json')
        with np.load(self.dest/'target.npz',allow_pickle=False) as data:values={k:data[k].copy() for k in data.files}
        values['reconstructed_spectrum']+=1
        np.savez(self.dest/'target.npz',**values)
        with patch.dict(os.environ,{'ISU_CHEMICAL_DAE_RUNTIME_REPORT':str(self.cert)}),patch('vibration.dae_bundle.runtime_versions',return_value=self.target),patch('vibration.dae_runtime_compat.runtime_versions',return_value=self.target):
            with self.assertRaises(ValueError):load_manifest(self.bundle)

    def test_synthetic_evidence_cannot_enable_live_mismatch(self):
        self.convert();mat=self.folder/'signal.mat';mat.write_bytes(self.raw)
        with tempfile.TemporaryDirectory() as root:
            r=Path(root)/'ref';t=Path(root)/'target'
            execute('export',self.bundle,mat,r,'synthetic');execute('verify',self.bundle,mat,t,'synthetic',r/'reference.json')
            actual={**runtime_versions(),'numpy':'not-verified'}
            with patch.dict(os.environ,{'ISU_CHEMICAL_DAE_RUNTIME_REPORT':str(t/'runtime_compatibility.json')}),patch('vibration.dae_bundle.runtime_versions',return_value=actual):
                with self.assertRaises(ValueError):load_manifest(self.bundle)

    def test_resigned_array_numeric_difference_and_threshold_change_rejected(self):
        from vibration.dae_bundle import sha256_file
        self.prepare();source=read_json(self.ref/'reference.json');target=read_json(self.dest/'target.json')
        with np.load(self.dest/'target.npz',allow_pickle=False) as data:values={k:data[k].copy() for k in data.files}
        values['reconstructed_spectrum']+=.5;np.savez(self.dest/'target.npz',**values)
        target['arrays_sha256']=sha256_file(self.dest/'target.npz')
        with self.assertRaises(AssertionError):compare_records(source,target,self.ref,self.dest)
        with np.load(self.ref/'reference.npz',allow_pickle=False) as data:values={k:data[k].copy() for k in data.files}
        values['threshold']=values['threshold']+1e-12;np.savez(self.dest/'target.npz',**values)
        target['arrays_sha256']=sha256_file(self.dest/'target.npz')
        with self.assertRaisesRegex(ValueError,'threshold'):compare_records(source,target,self.ref,self.dest)

    def test_prior_legacy_report_bound_to_same_input_and_bundle(self):
        from scripts.check_dae_runtime_compat import compare_legacy
        from vibration.dae_bundle import load_integrity_manifest
        self.prepare();r=read_json(self.ref/'reference.json')
        with np.load(self.ref/'reference.npz',allow_pickle=False) as d:values={k:d[k].copy() for k in d.files}
        old={('features' if k=='log_spectrum' else k):v for k,v in values.items()}
        np.savez(self.out/'legacy_reference.npz',**old)
        report={'status':'compatibility_checks_passed_rpm_unconfirmed','mat_sha256':r['input_sha256'],'bundle_id':r['bundle_id'],'runtime_versions':r['runtime']}
        p=self.out/'report.json';p.write_text(json.dumps(report))
        result=compare_legacy(p,load_integrity_manifest(self.bundle),r['input_sha256'],values)
        self.assertTrue(all(v==0 for v in result['max_abs_errors'].values()))
        with self.assertRaises(ValueError):compare_legacy(p,load_integrity_manifest(self.bundle),'0'*64,values)
