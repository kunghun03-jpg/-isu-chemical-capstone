"""Bundle/input/runtime-bound numerical evidence. No global version allowlist or RPM bypass."""
import json
from pathlib import Path
import platform
import numpy as np
from vibration.dae_bundle import sha256_file,runtime_versions,load_integrity_manifest

ROOT=Path(__file__).resolve().parents[1]
TOLERANCES={'log_spectrum':{'rtol':1e-6,'atol':1e-7},'input_spectrum':{'rtol':1e-6,'atol':1e-7},
            'reconstructed_spectrum':{'rtol':1e-5,'atol':1e-6},'mse':{'rtol':1e-5,'atol':1e-7}}


def code_hashes():
    result={name:sha256_file(ROOT/'vibration'/name) for name in ('dae_model.py','dae_preprocessing.py','dae_inference.py','dae_bundle.py','dae_runtime_compat.py')}
    result['scripts/check_dae_runtime_compat.py']=sha256_file(ROOT/'scripts/check_dae_runtime_compat.py')
    return result


def execution_identity():
    return {'python':platform.python_version(),'system':platform.system(),'machine':platform.machine()}


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def arrays(record,base):
    p=(Path(base)/record['arrays_file']).resolve()
    if p.parent!=Path(base).resolve() or sha256_file(p)!=record['arrays_sha256']:raise ValueError('호환성 수치 파일 경로/해시 오류')
    with np.load(p,allow_pickle=False) as data:
        expected={'log_spectrum','input_spectrum','reconstructed_spectrum','mse','threshold'}
        if set(data.files)!=expected:raise ValueError('호환성 수치 파일 스키마 오류')
        out={k:data[k].copy() for k in data.files}
    for key,value in out.items():
        shape=(800,) if key in ('log_spectrum','input_spectrum','reconstructed_spectrum') else ()
        if value.shape!=shape or not np.isfinite(value).all():raise ValueError('비유한값/입력 크기 오류')
    if out['mse'].item()<0 or out['threshold'].item()<0:raise ValueError('음수 MSE/임계값 오류')
    return out


def compare_records(source,target,source_dir,target_dir):
    for key in ('bundle_id','artifacts','manifest_sha256','input_sha256','code_hashes','data_kind'):
        if source.get(key)!=target.get(key):raise ValueError('호환성 근거 불일치: '+key)
    a,b=arrays(source,source_dir),arrays(target,target_dir)
    if any(a[k].dtype!=b[k].dtype for k in a):raise ValueError('전처리/복원 dtype 불일치')
    if a['threshold'].item()!=b['threshold'].item():raise ValueError('threshold 변경 감지')
    errors={}
    for key,tol in TOLERANCES.items():
        np.testing.assert_allclose(b[key],a[key],**tol,err_msg=key)
        errors[key]=float(np.max(np.abs(b[key]-a[key])))
    if bool(a['mse']>a['threshold'])!=bool(b['mse']>b['threshold']):raise ValueError('임계값 경계 판정 불일치')
    return errors


def audit_values(folder,raw):
    """Offline only: integrity + structure + finite checks, no version or RPM claim."""
    from vibration.dae_inference import SavedDAEInference
    from vibration.dae_preprocessing import preprocess_mat_bytes,validate_statistics
    manifest=load_integrity_manifest(folder)
    engine=SavedDAEInference.__new__(SavedDAEInference);engine.folder=Path(folder)
    with np.load(engine.folder/'normalization.npz',allow_pickle=False) as data:
        if set(data.files)!={'mean','std'}:raise ValueError('normalization schema')
        engine.mean,engine.std=data['mean'].copy(),data['std'].copy()
    validate_statistics(engine.mean,engine.std)
    with np.load(engine.folder/'threshold.npz',allow_pickle=False) as data:
        if set(data.files)!={'threshold'}:raise ValueError('threshold schema')
        threshold=data['threshold'].copy()
    engine._initialize_model(threshold)
    value=SavedDAEInference.predict_bytes(engine,raw)
    return manifest,dict(log_spectrum=preprocess_mat_bytes(raw),input_spectrum=value['input_spectrum'].reshape(800),
                         reconstructed_spectrum=value['reconstructed_spectrum'].reshape(800),mse=np.asarray(value['mse']),threshold=np.asarray(value['threshold']))


def validate_report(path,manifest):
    """An operator-selected local evidence set, never trust a 'passed' flag alone."""
    p=Path(path).expanduser().resolve();report=read_json(p)
    if type(report.get('schema_version')) is not int or report.get('schema_version')!=1 or report.get('tolerances')!=TOLERANCES:raise ValueError('호환성 보고서 규격/허용오차 오류')
    source_path=Path(report['source_record']).resolve();target_path=p.parent/'target.json'
    if sha256_file(source_path)!=report['source_record_sha256'] or sha256_file(target_path)!=report['target_record_sha256']:raise ValueError('호환성 보고서 근거 해시 오류')
    source,target=read_json(source_path),read_json(target_path)
    if source.get('runtime')!=manifest['runtime_versions'] or target.get('runtime')!=runtime_versions():raise ValueError('검증되지 않은 실행 버전 조합')
    if target.get('execution_identity')!=execution_identity():raise ValueError('검증되지 않은 Python/OS/아키텍처 환경')
    if source.get('data_kind')!='actual' or target.get('code_hashes')!=code_hashes():raise ValueError('실제 MAT/현재 코드의 검증 근거가 없습니다.')
    if source.get('bundle_id')!=manifest['bundle_id'] or source.get('artifacts')!=manifest['artifacts']:raise ValueError('다른 번들의 호환성 근거')
    if not isinstance(source.get('input_sha256'),str) or len(source['input_sha256'])!=64:raise ValueError('MAT 지문 누락')
    compare_records(source,target,source_path.parent,p.parent)
    return report
