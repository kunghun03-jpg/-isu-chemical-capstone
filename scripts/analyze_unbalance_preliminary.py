"""Fixed 1800RPM AE-only audit. No RF, DB, model fitting or threshold adjustment."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from vibration.dashboard_v2 import DashboardConfig
from vibration.dashboard_v3 import analyze_selected
from scripts.analyze_fault_signatures import metadata
from scripts.validate_v2_integration import digest,write_csv

REFERENCE='20260919_132506_Vib.mat'
REFERENCE_SCORE=0.37572573
REFERENCE_ATOL=5e-9+1e-15 # half the supplied 8-decimal display unit; not a model threshold


def run(data_dir,bundles,output,data_kind):
    folder=Path(data_dir).expanduser().resolve();bundle_root=Path(bundles).expanduser().resolve();out=Path(output).expanduser().resolve()
    if not folder.is_dir():raise FileNotFoundError(f'실제 MAT 폴더 없음: {folder}')
    if out==ROOT or ROOT in out.parents or out==folder or folder in out.parents or out==bundle_root or bundle_root in out.parents:raise ValueError('output must be outside repository/data/models')
    files=sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower()=='.mat')
    if not files:raise ValueError('MAT 파일이 없습니다.')
    model_files={str(p):digest(p) for p in (bundle_root/'1800rpm').glob('*') if p.is_file()}
    out.mkdir(parents=True) # never overwrite a prior run
    rows=[];config=DashboardConfig('autoencoder',bundle_root)
    try:
        for i,path in enumerate(files):
            row=dict(filename=path.name,path=str(path),sha256='',rpm=1800,status='error',ae_score=None,threshold=None,decision='',miss_candidate=None,error='',data_kind=data_kind)
            try:
                raw=path.read_bytes();row['sha256']=hashlib.sha256(raw).hexdigest()
                m=metadata(path)
                if m['date']!='20260919' or m['rpm']!=1800 or m['condition']!='Unbalance_-1':raise ValueError('대상 날짜/RPM/조건 경로가 아닙니다.')
                result,versions=analyze_selected(raw,1800,'uploaded.mat',config)
                item=result['models']['autoencoder']
                if digest(path)!=row['sha256']:raise ValueError('검사 중 MAT 변경')
                row.update(status='ok',ae_score=item['file_score'],threshold=item['file_threshold'],decision='이상' if item['predicted_anomaly'] else '정상',
                    miss_candidate=not bool(item['predicted_anomaly']),bundle_id=result['bundle_id'],model_version=versions['autoencoder'])
            except Exception as error:row['error']=f'{type(error).__name__}: {error}'
            rows.append(row);write_csv(out/'ae_results_50.csv',rows)
            print(f'{i+1}/{len(files)} {path.name}: {row["decision"] or row["status"]}',flush=True)
    finally:
        good=[r for r in rows if r['status']=='ok'];normal=sum(r['miss_candidate'] for r in good);scores=[r['ae_score'] for r in good]
        reference=next((r for r in rows if r['filename']==REFERENCE),None)
        match=bool(reference and reference['status']=='ok' and np.isclose(reference['ae_score'],REFERENCE_SCORE,rtol=0,atol=REFERENCE_ATOL))
        changed=[p for p,h in model_files.items() if not Path(p).is_file() or digest(p)!=h]
        summary={'data_kind':data_kind,'executed_at_utc':datetime.now(timezone.utc).isoformat(),'data_dir':str(folder),'rpm':1800,
            'discovered_count':len(files),'expected_count':50,'expected_count_matches':len(files)==50,'processed_count':len(rows),
            'success_count':len(good),'error_count':sum(r['status']=='error' for r in rows),'unprocessed_count':len(files)-len(rows),
            'normal_miss_candidate_count':normal,'anomaly_count':len(good)-normal,'normal_fraction_total':normal/len(files),
            'normal_fraction_successful':normal/len(good) if good else None,'normal_fraction_total_is_lower_bound':len(good)!=len(files),
            'score_distribution':{'n':len(scores),'min':min(scores) if scores else None,'max':max(scores) if scores else None,
                'mean':float(np.mean(scores)) if scores else None,'median':float(np.median(scores)) if scores else None,
                'q25':float(np.quantile(scores,.25)) if scores else None,'q75':float(np.quantile(scores,.75)) if scores else None},
            'thresholds':sorted({r['threshold'] for r in good}),'reference_filename':REFERENCE,'reference_expected_display_score':REFERENCE_SCORE,
            'reference_actual_score':reference['ae_score'] if reference and reference['status']=='ok' else None,'reference_atol':REFERENCE_ATOL,
            'reference_score_matches':match,'model_artifact_sha256':model_files,'changed_models':changed,
            'independent_test_performance_validated':False,'limitations':['Condition label is supplied by directory; normal predictions are missed-detection candidates, not proven physical fault diagnoses.','Train/Validation/Test or same Run overlap is unconfirmed.','Errors/unprocessed files are not classified as normal.']}
        summary['completed_and_checked']=len(rows)==len(files) and len(good)==len(files) and len(files)==50 and match and not changed
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
        (out/'README.md').write_text(f'''# Unbalance_-1 AE 탐지 누락 예비 분석\n\n데이터 종류: {data_kind}. 실제 입력 폴더: {folder}. RPM=1800.\n\n발견 {len(files)}개 (요청 50개와 일치: {len(files)==50}), 처리 {len(rows)}개, 성공 {len(good)}개, 오류 {summary['error_count']}개, 미처리 {summary['unprocessed_count']}개.\n\n정상 판정/누락 후보 {normal}개, 이상 {len(good)-normal}개. 전체 발견 파일 대비 정상 비율 {normal/len(files):.6%}. 오류/미처리가 있으면 이 비율은 확인된 정상 후보의 하한이며 완전한 50개 판정 비율이 아니다. 성공 파일 기준 비율은 summary.json에 별도 기록한다.\n\n기준 파일 실제 점수: {summary['reference_actual_score']}, 제공 표시값 {REFERENCE_SCORE}, 일치: {match}, atol={REFERENCE_ATOL} (rtol=0, 표시 8자리 기준). 비교값을 모델 점수나 임계값으로 주입하지 않았다.\n\n점수 분포: {json.dumps(summary['score_distribution'],ensure_ascii=False)}\n\n기존 V3 AE 전처리·저장 모델·임계값을 사용했다. RF 및 운영 DB를 호출하지 않았다. 정상으로 판정된 폴더 레이블 Unbalance_-1을 누락 후보로 기록하며 라벨/실제 결함을 자동 확정하지 않는다. 학습/평가 및 같은 날짜·Run 중복 여부가 미확인이므로 독립 Test 성능이나 정확도로 표현하지 않는다.\n''')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',default=str(ROOT/'data/20260919/20260919_1800RPM/Unbalance_-1/Vib'));p.add_argument('--bundles',default=str(ROOT/'outputs/inference_bundles/date_prelim_20261106'));p.add_argument('--output',default=str(Path.home()/'v3-error-analysis/unbalance-preliminary'));p.add_argument('--data-kind',choices=('actual','synthetic'),required=True)
    a=p.parse_args();r=run(a.data_dir,a.bundles,a.output,a.data_kind);print(json.dumps(r,ensure_ascii=False,indent=2));return 0 if r['completed_and_checked'] else 1

if __name__=='__main__':raise SystemExit(main())
