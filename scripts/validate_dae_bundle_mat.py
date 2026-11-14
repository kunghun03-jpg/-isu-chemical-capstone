"""Offline DAE bundle/MAT audit. Explicit unconfirmed diagnostics never enable public inference."""
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path
import re
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from vibration.dae_readiness import probe
from vibration.dae_inference import SavedDAEBundleInference,SavedDAEInference
from vibration.dae_bundle import FILES,sha256_file
from scripts.validate_v2_integration import write_csv,load_cases
from scripts.validate_dae_wsl import investigate_rpm,array_headers


def provenance(source):
    result={'supported_rpms':None,'split_independence':'unconfirmed','metadata':{},'limitations':['Metadata references do not establish trained RPM coverage or independent splits.']}
    if source is None:return result
    source=Path(source).expanduser().resolve()
    if not (source/'normal_dataset_normalized.npz').is_file():
        result['error']='normal_dataset_normalized.npz missing';return result
    result['investigation']=investigate_rpm(source,ROOT/'dae_legacy')
    for name in ('normal_dataset_normalized.npz','fault_dataset.npz'):
        p=source/name
        if not p.is_file():continue
        meta={'sha256':sha256_file(p),'headers':array_headers(p),'observations':{}}
        with np.load(p,allow_pickle=False) as data:
            for key,h in meta['headers'].items():
                if not re.search('rpm|date|group|file|run',key,re.I) or np.prod(h['shape'])>10000 or 'object' in h['dtype']:continue
                a=data[key]
                if a.dtype.kind in 'US':
                    dates=Counter();rpms=Counter()
                    for v in a.flat:
                        text=str(v)
                        for x in set(re.findall(r'(?<!\d)(20\d{6})(?!\d)',text)):dates[x]+=1
                        for x in set(re.findall(r'(\d+)RPM',text,re.I)):rpms[x]+=1
                    meta['observations'][key]={'date_counts':dict(dates),'rpm_counts':dict(rpms),'split_assignment':'only what metadata key declares; not independently verified'}
                elif 'rpm' in key.lower() and a.dtype.kind in 'iu':meta['observations'][key]={'rpm_values':np.unique(a).tolist()}
        result['metadata'][name]=meta
    return result


def validate(folder,cases,output,data_kind,diagnostic=False,source=None):
    out=Path(output).expanduser().resolve();folder=Path(folder).expanduser().resolve()
    if out==ROOT or ROOT in out.parents:raise ValueError('output must be outside repository')
    if out==folder or folder in out.parents:raise ValueError('output must not be inside bundle')
    if source is not None:
        source_path=Path(source).expanduser().resolve()
        if out==source_path or source_path in out.parents:raise ValueError('output must not be inside source data folder')
    if not cases:raise ValueError('no MAT cases')
    out.mkdir(parents=True)
    snapshot={str(p):sha256_file(p) for p in [folder/'manifest.json',*[folder/f for f in FILES]] if p.is_file()}
    rows=[]
    for case in cases:
        rpm=int(case['rpm']);path=Path(case.get('resolved_path',case['path'])).expanduser().resolve()
        state=probe(rpm,folder)
        row={'path':str(path),'rpm':rpm,'date':case.get('date',''),'condition':case.get('condition',''),'data_kind':data_kind,
             'bundle_state':state['code'],'model_version':state.get('model_version',''),'status':'blocked',
             'training_provenance':state.get('training_provenance','unavailable'),'supported_rpms':json.dumps(state.get('supported_rpms')),
             'public_inference_allowed':state['ready'],'diagnostic_mse':'','threshold':'','prediction':'','reason':state.get('reason','')}
        try:
            if state['ready'] or diagnostic and state['code']=='rpm_unconfirmed':
                raw=path.read_bytes();before=hashlib.sha256(raw).hexdigest();row['input_sha256']=before
                engine=SavedDAEBundleInference(folder)
                if state['ready']:
                    value=engine.predict_bytes(raw,rpm=rpm)
                    row.update(status='public_supported_result',diagnostic_mse=value['mse'],threshold=value['threshold'],prediction=value['prediction'])
                else:
                    # Explicit offline compatibility diagnostic, same validated kernels.
                    # Never return a normal/abnormal public decision for unconfirmed RPM.
                    value=SavedDAEInference.predict_bytes(engine,raw)
                    row.update(status='diagnostic_only_rpm_unconfirmed',diagnostic_mse=value['mse'],threshold=value['threshold'],prediction='')
                if sha256_file(path)!=before:raise ValueError('MAT changed during validation')
        except Exception as error:row.update(status='error',reason=f'{type(error).__name__}: {error}')
        rows.append(row)
    try: training=provenance(source)
    except Exception as error:training={'error':f'{type(error).__name__}: {error}','supported_rpms':None}
    changed=[p for p,h in snapshot.items() if not Path(p).is_file() or sha256_file(p)!=h]
    summary={'data_kind':data_kind,'files':len(rows),'statuses':dict(Counter(r['status'] for r in rows)),
             'bundle_sha256':snapshot,'changed_artifacts':changed,'training_provenance':training,'performance_validated':False,
             'public_policy_unchanged':True,'rows':rows,'limitations':['One-file numerical diagnostics are not RPM support or detection performance. No training or threshold changes.']}
    write_csv(out/'dae_mat_results.csv',rows)
    (out/'dae_validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',required=True);p.add_argument('--mat');p.add_argument('--rpm',type=int,choices=(900,1200,1800));p.add_argument('--cases');p.add_argument('--data-root',default=str(ROOT));p.add_argument('--source');p.add_argument('--diagnostic-unconfirmed',action='store_true');p.add_argument('--output',required=True);p.add_argument('--data-kind',choices=('actual','synthetic'),required=True)
    a=p.parse_args()
    if bool(a.cases)==bool(a.mat):p.error('choose --cases OR --mat')
    if a.mat and a.rpm is None:p.error('--mat requires --rpm')
    cases=load_cases(a.cases,a.data_root,require_reference=False) if a.cases else [{'path':a.mat,'rpm':a.rpm}]
    report=validate(a.bundle,cases,a.output,a.data_kind,a.diagnostic_unconfirmed,a.source)
    print(json.dumps({'statuses':report['statuses'],'output':a.output,'performance_validated':False},ensure_ascii=False))
    return 1 if report['changed_artifacts'] or report['statuses'].get('error') or report['statuses'].get('blocked') else 0

if __name__=='__main__':raise SystemExit(main())
