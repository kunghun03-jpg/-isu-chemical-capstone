"""Two-environment DAE numeric audit. Does not alter bundle/runtime/RPM metadata."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from vibration.dae_bundle import runtime_versions,sha256_file,load_integrity_manifest
from vibration.dae_runtime_compat import audit_values,code_hashes,compare_records,TOLERANCES,read_json,execution_identity


def compare_legacy(report_path,manifest,input_sha,values):
    path=Path(report_path).expanduser().resolve();prior=read_json(path)
    if (prior.get('status')!='compatibility_checks_passed_rpm_unconfirmed' or prior.get('mat_sha256')!=input_sha or prior.get('bundle_id')!=manifest['bundle_id'] or prior.get('runtime_versions')!=manifest['runtime_versions']):
        raise ValueError('기존 보고서의 상태/MAT/번들/생성 환경이 일치하지 않습니다.')
    errors={}
    with np.load(path.parent/'legacy_reference.npz',allow_pickle=False) as original:
        for key,tol in TOLERANCES.items():
            old=original['features' if key=='log_spectrum' else key]
            a=np.asarray(old).reshape(values[key].shape)
            np.testing.assert_allclose(values[key],a,**tol,err_msg='prior legacy '+key)
            errors[key]=float(np.max(np.abs(values[key]-a)))
        if np.asarray(original['threshold']).item()!=values['threshold'].item():raise ValueError('기존 threshold 불일치')
        if bool(np.asarray(original['mse']).item()>values['threshold'])!=bool(values['mse']>values['threshold']):raise ValueError('기존 진단 경계 불일치')
    return {'report_sha256':sha256_file(path),'legacy_arrays_sha256':sha256_file(path.parent/'legacy_reference.npz'),'max_abs_errors':errors}


def record(bundle,mat,out,data_kind,name,legacy_report=None):
    bundle,mat=Path(bundle).expanduser().resolve(),Path(mat).expanduser().resolve()
    raw=mat.read_bytes();manifest,values=audit_values(bundle,raw)
    arrays_file=name+'.npz';np.savez(out/arrays_file,**values)
    item={'execution_identity':execution_identity(),'runtime':runtime_versions(),'bundle_id':manifest['bundle_id'],'artifacts':manifest['artifacts'],
          'manifest_sha256':sha256_file(bundle/'manifest.json'),'input_sha256':hashlib.sha256(raw).hexdigest(),
          'input_path':str(mat),'code_hashes':code_hashes(),'data_kind':data_kind,
          'arrays_file':arrays_file,'arrays_sha256':sha256_file(out/arrays_file)}
    if legacy_report:item['legacy_comparison']=compare_legacy(legacy_report,manifest,item['input_sha256'],values)
    if sha256_file(mat)!=item['input_sha256'] or load_integrity_manifest(bundle)!=manifest:raise ValueError('입력/번들 변경 감지')
    (out/(name+'.json')).write_text(json.dumps(item,indent=2));return item


def execute(mode,bundle,mat,output,data_kind,reference=None,legacy_report=None):
    out=Path(output).expanduser().resolve();bundle=Path(bundle).expanduser().resolve()
    if out==ROOT or ROOT in out.parents or out==bundle or bundle in out.parents:raise ValueError('출력은 저장소/번들 밖 새 폴더여야 합니다.')
    m=load_integrity_manifest(bundle)
    if mode=='export' and m['runtime_versions']!=runtime_versions():raise ValueError('원본 기준은 manifest와 정확히 같은 생성 환경에서만 내보낼 수 있습니다.')
    out.mkdir(parents=True)
    if mode=='export':return record(bundle,mat,out,data_kind,'reference',legacy_report)
    if not reference:raise ValueError('reference required')
    ref_path=Path(reference).expanduser().resolve();source=read_json(ref_path)
    target=record(bundle,mat,out,data_kind,'target',legacy_report)
    if source['runtime']!=m['runtime_versions']:raise ValueError('기준 환경이 manifest와 다릅니다.')
    errors=compare_records(source,target,ref_path.parent,out)
    report={'schema_version':1,'tolerances':TOLERANCES,'source_record':str(ref_path),
            'source_record_sha256':sha256_file(ref_path),'target_record_sha256':sha256_file(out/'target.json'),
            'max_abs_errors':errors,'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'execution_compatibility_only':True,'performance_validated':False,'rpm_support_validated':False,
            'source_runtime':source['runtime'],'target_runtime':target['runtime'],'input_sha256':target['input_sha256']}
    (out/'runtime_compatibility.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('export','verify'));p.add_argument('--bundle',required=True);p.add_argument('--mat',required=True);p.add_argument('--output',required=True);p.add_argument('--reference');p.add_argument('--legacy-report');p.add_argument('--data-kind',choices=('actual','synthetic'),required=True)
    a=p.parse_args();print(json.dumps(execute(a.mode,a.bundle,a.mat,a.output,a.data_kind,a.reference,a.legacy_report),indent=2))

if __name__=='__main__':main()
