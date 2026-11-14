"""One MAT diagnostic: reproduce saved OCSVM math and compare classic offline/V3 paths."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from vibration.inference import SavedModelInference
from vibration.dashboard_v2 import DashboardConfig
from vibration.dashboard_v3 import analyze_selected,references
from vibration.v3_presentation import dae_diagnostics
from vibration.one_class import anomaly_scores
from vibration.comparison_metrics import file_scores,predictions


def ocsvm_details(raw,rpm,folder):
    engine=SavedModelInference(folder,models=('ocsvm',),selected_artifacts_only=True)
    _,values=engine.preprocess(raw)
    features=np.ascontiguousarray(values[:,0,:],dtype=np.float64)
    raw_scores=np.asarray(engine._estimators['ocsvm'].score_samples(features),dtype=np.float64)
    anomaly=-raw_scores
    expected=float(anomaly.mean());threshold=float(engine.manifest['models']['ocsvm']['file_threshold'])
    actual=engine.predict_bytes(raw,rpm=rpm)['models']['ocsvm']
    offline_segments=anomaly_scores(engine._estimators['ocsvm'],values)
    offline_files=file_scores([{'split':'test','file':'uploaded.mat','segment_id':i,'score':float(s)} for i,s in enumerate(offline_segments)])
    offline=predictions(offline_files,threshold)[0]
    if offline['score']!=expected or offline['predicted_anomaly']!=actual['predicted_anomaly']:
        raise ValueError('Original offline evaluation helpers mismatch')
    if not np.isfinite(raw_scores).all() or expected!=actual['file_score'] or int(expected>threshold)!=actual['predicted_anomaly']:
        raise ValueError('OCSVM calculation mismatch')
    return dict(score_samples=raw_scores.tolist(),segment_anomaly_scores=anomaly.tolist(),file_score=expected,
                offline_helpers_exact_match=True,offline_helper_score=offline['score'],exact_file_score_repr=repr(expected),threshold=threshold,threshold_repr=repr(threshold),
                predicted_anomaly=int(expected>threshold),rule='mean(-score_samples) > stored file_threshold',
                normalized_input_min=float(values.min()),normalized_input_max=float(values.max()),
                bundle_id=engine.manifest.get('bundle_id'),recorded_versions=engine.manifest.get('versions'),
                all_raw_scores_exactly_zero=bool(np.all(raw_scores==0)),numpy_version=np.__version__,
                explanation='0도 음수 임계값보다 크면 이상입니다. 0은 확률이 아닙니다. raw 값이 0인지 반올림인지 확인하세요. RBF kernel underflow/학습 범위 이탈은 추가 조사 가설이며 이 값만으로 확정하지 않습니다.')


def run(mat,rpm,bundle_root,data_kind="not_declared"):
    path=Path(mat).expanduser().resolve();raw=path.read_bytes();checksum=hashlib.sha256(raw).hexdigest()
    config=DashboardConfig('autoencoder',Path(bundle_root).expanduser().resolve())
    report={'data_kind':data_kind,'bundle_root':str(config.bundle_root),'input':str(path),'input_sha256':checksum,'rpm':rpm,'dae':dae_diagnostics(rpm),'models':{},'limitations':['No training or threshold changes. One-file calculation parity is not test-set performance.']}
    for model in ('autoencoder','ocsvm','iforest'):
        try:
            v3,_=analyze_selected(raw,rpm,'uploaded.mat',config,model)
            item=v3['models'][model];entry={'v3':item}
            try:
                classic=SavedModelInference(config.bundle_root/f'{rpm}rpm',models=(model,)).predict_bytes(raw,rpm=rpm)['models'][model]
                entry.update(classic_offline_path=classic,exact_match=classic==item)
            except Exception as error:
                entry['classic_error']=f'{type(error).__name__}: {error}'
            if model=='ocsvm': entry['calculation']=ocsvm_details(raw,rpm,config.bundle_root/f'{rpm}rpm')
            report['models'][model]=entry
        except Exception as error:report['models'][model]={'error':f'{type(error).__name__}: {error}'}
    ae=report['models']['autoencoder'].get('v3')
    if ae:
        ae_result,_=analyze_selected(raw,rpm,'uploaded.mat',config)
        report['reference_results']=references(raw,rpm,config,ae_result)
    report['input_unchanged']=hashlib.sha256(path.read_bytes()).hexdigest()==checksum
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mat',required=True);p.add_argument('--rpm',type=int,choices=(900,1200,1800),required=True);p.add_argument('--bundle-root',required=True);p.add_argument('--output',required=True);p.add_argument('--data-kind',choices=('actual','synthetic'),required=True)
    a=p.parse_args();out=Path(a.output).expanduser().resolve()
    if out==ROOT or ROOT in out.parents:raise ValueError('output must be outside repository')
    report=run(a.mat,a.rpm,a.bundle_root,a.data_kind)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(out)
    return 1 if not report['input_unchanged'] or any(r.get('error') or r.get('classic_error') or r.get('exact_match') is False for r in report['models'].values()) else 0

if __name__=='__main__':raise SystemExit(main())
