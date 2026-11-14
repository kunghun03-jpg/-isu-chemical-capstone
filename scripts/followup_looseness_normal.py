"""Reuse Looseness CSV/JSON; infer only uncached Normal AE. No RF or DB calls."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import json
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.analyze_looseness_date_shift import discover,process,summaries,control_comparison,comparisons,plots,FEATURES,LOOSE,NORMAL,RPMS
from scripts.validate_v2_integration import digest,write_csv
from vibration.inference import SavedModelInference
from vibration.candidate_analysis import candidate_analysis
from vibration.web_inference import preview_mat


def flag(value):
    if str(value).lower() in ('true','1'):return True
    if str(value).lower() in ('false','0'):return False
    raise ValueError('invalid AE decision')


def number(value):
    result=float(value)
    if not np.isfinite(result):raise ValueError('nonfinite result')
    return result


def coerce(row):
    r=dict(row);r['path']=str(Path(row.get('resolved_path') or row['path']).expanduser().resolve());r['rpm']=int(row['rpm']);r['date']=str(row['date'])
    r['sha256']=row.get('sha256') or row.get('input_sha256','')
    r['read_ok']=flag(row['read_ok']) if row.get('read_ok') not in (None,'') else row.get('status') in ('ok','analyzed')
    if row.get('status') not in ('ok','analyzed'):return r
    r.update(status='ok',ae_score=number(row['ae_score']),ae_threshold=number(row.get('ae_threshold',row.get('threshold'))),ae_anomaly=flag(row['ae_anomaly']))
    if r['ae_anomaly']!=(r['ae_score']>r['ae_threshold']):raise ValueError('cached AE rule mismatch')
    features=json.loads(row['features']) if row.get('features') else row
    r['features_complete']=all(k in features and (features.get(k) not in (None,'','None') or k=='crest_factor_ac') for k in FEATURES)
    for key in FEATURES:r[key]=None if features.get(key) in (None,'','None') else number(features[key])
    r['model_version']=row.get('model_version') or json.loads(row.get('ae_versions') or '{}').get('autoencoder','')
    r['score_threshold_ratio']=r['ae_score']/r['ae_threshold'] if r['ae_threshold']>1e-12 else None
    return r


def cache_metadata(folder):
    for name in ('followup_summary.json','analysis_report.json','integration_summary.json'):
        p=folder/name
        if p.is_file():return json.loads(p.read_text())
    return {}


def cache_index(roots):
    entries={};sources={};bad=[]
    for root in roots:
        root=Path(root).expanduser().resolve()
        if not root.exists():raise FileNotFoundError('cache root missing: '+str(root))
        files=[root] if root.is_file() else [p for p in root.rglob('*.csv') if p.name in ('normal_results.csv','file_results.csv','integration_results.csv')]
        for p in files:
            if str(p) in sources:continue
            sources[str(p)]=digest(p)
            try:
                meta=cache_metadata(p.parent)
                for name in ('followup_summary.json','analysis_report.json','integration_summary.json'):
                    mp=p.parent/name
                    if mp.is_file():sources[str(mp)]=digest(mp)
                with p.open(encoding='utf-8-sig',newline='') as f:
                    for raw in csv.DictReader(f):
                        if raw.get('condition') not in NORMAL or raw.get('status') not in ('ok','analyzed'):continue
                        try:
                            r=coerce(raw);entries.setdefault(r['path'],[]).append((r,meta,str(p)))
                        except Exception as error:bad.append({'source':str(p),'path':raw.get('path',''),'reason':str(error)})
            except Exception as error:bad.append({'source':str(p),'reason':str(error)})
    return entries,sources,bad


def ae_proof(row,meta,engine,folder):
    m=engine.manifest;model=m['models']['autoencoder'];expected='sha256:'+m['artifacts'][model['artifact']]
    if row.get('model_version')!=expected or row['ae_threshold']!=model['file_threshold']:return False
    if row.get('bundle_id')==m['bundle_id']:return True
    hashes=meta.get('input_artifact_sha256',meta.get('artifact_hashes',{}))
    for name in ('manifest.json',m['normalization']['artifact'],model['artifact']):
        values={h for p,h in hashes.items() if Path(p).name==name and Path(p).parent.name==folder.name}
        if values!={digest(folder/name)}:return False
    return True


def features_proven(meta):
    hashes=meta.get('feature_code_hashes',meta.get('implementation_hashes',{}));runtime=meta.get('runtime',{})
    return all(hashes.get(k)==digest(ROOT/k) for k in ('vibration/candidate_analysis.py','scripts/analyze_fault_signatures.py')) and runtime.get('numpy')==np.__version__ and runtime.get('scipy')==scipy.__version__


def inspect_errors(rows,engines):
    result=[]
    for row in rows:
        if row['status']=='ok':continue
        r={'path':row['path'],'date':row['date'],'rpm':row['rpm'],'condition':row['condition'],'recorded_status':row['status'],'recorded_error':row.get('error',''),'recorded_sha256':row.get('sha256','')}
        try:
            p=Path(row['path']);raw=p.read_bytes();r['current_sha256']=hashlib.sha256(raw).hexdigest();r['same_recorded_bytes']=r['current_sha256']==row.get('sha256') if row.get('sha256') else None
            preview_mat(raw);r['current_loader']='passed'
            try:candidate_analysis(raw,row['rpm']);r['current_features']='passed'
            except Exception as e:r['current_features']=f'{type(e).__name__}: {e}'
            if row['status']=='error' and row['rpm'] in engines:
                try:
                    value=engines[row['rpm']].predict_bytes(raw,rpm=row['rpm'],filename='uploaded.mat')['models']['autoencoder']
                    r['current_ae_status']='passed';r['current_ae_score']=value['file_score'];r['current_ae_threshold']=value['file_threshold']
                except Exception as e:r['current_ae_status']=f'{type(e).__name__}: {e}'
        except Exception as e:r['current_loader']=f'{type(e).__name__}: {e}'
        r['limitation']='Only previously failed Looseness may be rechecked. Original CSV counts are never rewritten; current checks are separate evidence.'
        result.append(r)
    return result


def run(looseness_output,data_root,bundles,output,cache_roots=(),data_kind='actual',no_plots=False,allow_recompute=False):
    source=Path(looseness_output).expanduser().resolve();out=Path(output).expanduser().resolve();root=Path(data_root).expanduser().resolve();bundle=Path(bundles).expanduser().resolve()
    if out==ROOT or ROOT in out.parents or any(out==p or p in out.parents for p in (source,root,bundle)):raise ValueError('output must be outside repository/source/data/models')
    if not root.is_dir():raise FileNotFoundError('Normal MAT root unavailable')
    feature_hashes={k:digest(ROOT/k) for k in ('vibration/candidate_analysis.py','scripts/analyze_fault_signatures.py')}
    original=json.loads((source/'analysis_report.json').read_text())
    with (source/'file_results.csv').open(encoding='utf-8-sig',newline='') as f:loose=[coerce(r) for r in csv.DictReader(f) if r.get('condition') in LOOSE]
    if not loose:raise ValueError('No Looseness result rows')
    entries,source_hashes,cache_bad=cache_index([source.parent,*cache_roots])
    invalid_cache_paths={str(Path(b['path']).expanduser().resolve()) for b in cache_bad if b.get('path')}
    source_hashes.update({str(source/n):digest(source/n) for n in ('file_results.csv','analysis_report.json')})
    cases,discovery_bad=discover(root,['20260912','20260916']);cases=[c for c in cases if c['condition'] in NORMAL]
    if not cases:raise ValueError('No Normal MAT files found')
    out.mkdir(parents=True)
    engines={};model_errors={};hashes={};normal=[];reject=[];attempted_ae=0;started=time.monotonic()
    for rpm in RPMS:
        folder=bundle/f'{rpm}rpm'
        hashes.update({str(p):digest(p) for p in folder.glob('*') if p.is_file()})
        try:engines[rpm]=SavedModelInference(folder,models=('autoencoder',))
        except Exception as e:model_errors[rpm]=f'{type(e).__name__}: {e}'
    errors=inspect_errors(loose,engines);write_csv(out/'looseness_errors.csv',errors)
    try:
        for i,case in enumerate(cases):
            r={**case,'status':'error','read_ok':False,'sha256':'','error':'','ae_mode':'error'}
            try:
                raw=Path(case['path']).read_bytes();r['sha256']=hashlib.sha256(raw).hexdigest();preview_mat(raw);r['read_ok']=True
                if case['rpm'] in model_errors:raise ValueError(model_errors[case['rpm']])
                engine=engines[case['rpm']];valid=[]
                for old,meta,location in entries.get(case['path'],[]):
                    reason=''
                    if old['sha256']!=r['sha256']:reason='MAT SHA mismatch'
                    elif any(old[k]!=case[k] for k in ('date','rpm','condition')):reason='metadata mismatch'
                    elif not ae_proof(old,meta,engine,bundle/f'{case["rpm"]}rpm'):reason='model/threshold/normalization provenance not verified'
                    if reason:reject.append({'path':case['path'],'source':location,'reason':reason})
                    else:valid.append((old,meta,location))
                if valid:
                    first,meta,location=valid[0]
                    if any(v[0]['ae_anomaly']!=first['ae_anomaly'] or not np.isclose(v[0]['ae_score'],first['ae_score'],rtol=1e-10,atol=1e-12) for v in valid):raise ValueError('conflicting verified cached AE results; no automatic re-inference')
                    r={**first,**case,'read_ok':True,'status':'ok','ae_mode':'reused','reuse_source':location}
                    if features_proven(meta) and first.get('features_complete'):r['features_mode']='reused'
                    else:
                        features,_=candidate_analysis(raw,case['rpm']);r.update(features['features']);r['features_mode']='recomputed_without_AE'
                else:
                    if (entries.get(case['path']) or case['path'] in invalid_cache_paths) and not allow_recompute:raise ValueError('existing cache is unverified; automatic repeat AE disabled (see cache_rejections.csv)')
                    attempted_ae+=1
                    r=process(case,engine,raw);r.update(ae_mode='inferred',features_mode='inferred',reuse_source='')
                if digest(case['path'])!=r['sha256']:raise ValueError('Normal MAT changed during processing')
            except Exception as e:r.update(status='error',error=f'{type(e).__name__}: {e}')
            r['data_kind']=data_kind;normal.append(r);write_csv(out/'normal_results.csv',normal)
            print(f'{i+1}/{len(cases)} {r["status"]} AE={r["ae_mode"]} {case["path"]}',flush=True)
    finally:
        combined=loose+normal;groups=summaries(combined,combined)
        write_csv(out/'group_summary.csv',groups);write_csv(out/'normal_control_comparison.csv',control_comparison(groups,['20260912','20260916']))
        by=[]
        for date in ('20260912','20260916'):
            for rpm in RPMS:
                good=[r for r in normal if r['date']==date and r['rpm']==rpm and r['status']=='ok']
                by.append(dict(date=date,rpm=rpm,success_files=len(good),ae_anomaly_n=sum(r['ae_anomaly'] for r in good),observed_normal_alarm_fraction=sum(r['ae_anomaly'] for r in good)/len(good) if good else None))
        write_csv(out/'normal_by_date_rpm.csv',by);write_csv(out/'cache_rejections.csv',reject+cache_bad)
        changed=[p for p,h in {**hashes,**source_hashes,**{str(ROOT/k):v for k,v in feature_hashes.items()}}.items() if not Path(p).is_file() or digest(p)!=h]
        mismatch=[r['path'] for r in loose if r['status']=='ok' and (r['rpm'] not in engines or r.get('bundle_id')!=engines[r['rpm']].manifest['bundle_id'])]
        distributions=[]
        for date in ('20260912','20260916'):
            for rpm in RPMS:
                for family,conditions in (('Looseness',LOOSE),('Normal',NORMAL)):
                    part=[r for r in combined if r['date']==date and r['rpm']==rpm and r['condition'] in conditions and r['status']=='ok']
                    distributions.append(dict(date=date,rpm=rpm,family=family,n=len(part),alarms=sum(r['ae_anomaly'] for r in part),
                        alarm_fraction=sum(r['ae_anomaly'] for r in part)/len(part) if part else None,
                        ae_score_median=float(np.median([r['ae_score'] for r in part])) if part else None,
                        rms_median=float(np.median([r['rms_ac'] for r in part])) if part else None))
        write_csv(out/'date_rpm_comparison.csv',distributions)
        shifts=comparisons(groups,['20260912','20260916'])
        write_csv(out/'date_shift_by_condition.csv',shifts)
        loose_pairs=[s for s in shifts if s['condition'] in LOOSE and 'ae_score_median_difference' in s]
        normal_pairs=[s for s in shifts if s['condition'] in NORMAL and 'ae_score_median_difference' in s]
        direction={name:{'paired_groups':len(part),'higher_score_0916':sum(s['ae_score_median_difference']>0 for s in part),'higher_rms_0916':sum(s['rms_ac_median_difference']>0 for s in part)} for name,part in (('Looseness',loose_pairs),('Normal_by_condition',normal_pairs))}
        summary={'data_kind':data_kind,'source_looseness_report':str(source),'source_conclusions_ready':original.get('conclusions_ready'),'source_reference_matched':original.get('reference_matched'),'source_reference_expected':original.get('reference_expected'),
            'source_json_count_matches':original.get('discovered_looseness')==len(loose),
            'looseness_rows':len(loose),'looseness_non_ok_blocks_original_conclusion':any(r['status']!='ok' for r in loose),'looseness_statuses':dict(Counter(r['status'] for r in loose)),'recorded_looseness_errors':errors,
            'declared_discovered_looseness':original.get('discovered_looseness'),'normal_discovered':len(cases),'normal_expected_from_user':614,'normal_count_matches_expected':len(cases)==614,
            'normal_processed':len(normal),'normal_ae_inference_calls':attempted_ae,'normal_statuses':dict(Counter(r['status'] for r in normal)),'normal_ae_modes':dict(Counter(r['ae_mode'] for r in normal if r['status']=='ok')),
            'normal_by_date_rpm':by,'date_rpm_comparison':distributions,'paired_direction_observations':direction,'input_csv_sha256':source_hashes,'artifact_hashes':hashes,'changed_sources':changed,'model_mismatch_paths':mismatch,
            'normal_discovery_exclusions':[e for e in discovery_bad if any(p in NORMAL for p in Path(e['path']).parts)],
            'feature_code_hashes':feature_hashes,
            'runtime':{'numpy':np.__version__,'scipy':scipy.__version__},'independent_test_performance_validated':False,
            'comparison_valid':not changed and not mismatch and len(normal)==len(cases) and all(r['status']=='ok' for r in normal),'elapsed_seconds':time.monotonic()-started}
        summary['normal_coverage_missing']=[{'date':b['date'],'rpm':b['rpm']} for b in by if b['success_files']==0]
        summary['comparison_valid']=summary['comparison_valid'] and summary['source_json_count_matches']
        summary['all_requested_normal_validated']=summary['comparison_valid'] and summary['normal_count_matches_expected'] and not summary['normal_coverage_missing'] and not summary['normal_discovery_exclusions']
        (out/'followup_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
        lines=['# Looseness / Normal 날짜 후속 검증',f'\n데이터 종류: {data_kind}. Looseness 성공 결과는 기존 CSV/JSON 재사용이다. 기존 오류 파일만 독립 재점검하며 원본 통계를 바꾸지 않는다.',
            f"\n기존 Looseness 상태: {summary['looseness_statuses']}. 원래 conclusions_ready={original.get('conclusions_ready')}를 정상군 추가만으로 true로 바꾸지 않는다.",
            f"\nNormal 발견 {len(cases)} (사용자 예상 614와 일치: {len(cases)==614}), 처리 {len(normal)}, 상태 {summary['normal_statuses']}, AE 재사용/신규 {summary['normal_ae_modes']}.",
            '\n## 기록된 오류와 현재 점검 (별도 증거)']
        for e in errors:lines.append(f"\n- {e['path']}\n  - 기존: {e['recorded_error']}\n  - 현재 로더: {e.get('current_loader')} / 특징: {e.get('current_features','미실행')} / AE: {e.get('current_ae_status','미실행')}")
        lines+=['\n## 날짜 × RPM별 Normal 관측 이상 비율','\n|날짜|RPM|성공 n|AE 이상|비율|','|---|---:|---:|---:|---:|']
        for b in by:lines.append(f"|{b['date']}|{b['rpm']}|{b['success_files']}|{b['ae_anomaly_n']}|{b['observed_normal_alarm_fraction']}|")
        lines+=['\n## 날짜/RPM별 Looseness와 Normal 분포','\n|날짜|RPM|구분|n|AE 이상 비율|AE 중앙값|RMS 중앙값|','|---|---:|---|---:|---:|---:|---:|']
        for d in distributions:lines.append(f"|{d['date']}|{d['rpm']}|{d['family']}|{d['n']}|{d['alarm_fraction']}|{d['ae_score_median']}|{d['rms_median']}|")
        lines+=['\n## 날짜 변화의 실제 방향 집계',json.dumps(direction,ensure_ascii=False),'같은 RPM/원래 조건의 두 날짜에 유효 데이터가 있는 조합만 집계한다. Normal1/2를 분리하며 원인/정확도 개선으로 해석하지 않는다.']
        lines+=['\n## 해석과 불확실성','\n기존 분석의 conclusions_ready는 탐지율 지표가 아니다. status!=ok인 파일이 하나라도 있으면 false이며, 오류와 AE 정상 판정(누락 후보)을 구분한다.',
            '\nNormal과 Looseness의 점수·RMS 분포는 날짜/RPM/원래 조건별 그림과 group_summary.csv에서 비교한다. Normal에서도 날짜 차이가 있으면 Looseness만의 변화라고 단정할 수 없다. 반대로 변화 크기/방향이 다르더라도 결함·센서·부하 원인이 확정되는 것은 아니다.',
            '\nLooseness의 AE 정상은 폴더 라벨 기준 탐지 누락 후보다. 이는 정상군의 날짜 분포 변화와 구분한다. 오류 파일은 정상/이상 분모에서 제외한다.',
            '\nNormal은 AE Train/Validation/threshold 데이터와 겹칠 수 있다. 관측 이상 판정 비율은 독립 정상 FPR/정확도가 아니다. 날짜·Run·라벨·센서·게인·부하 및 학습 파일 매핑이 미확인이다.',
            f"\n비교 계산 상태 comparison_valid={summary['comparison_valid']}. 모델/입력 불일치나 누락이 있으면 비교 결론을 보류한다. 원본 모델/threshold/운영 DB/Streamlit을 변경하지 않았다."]
        (out/'FOLLOWUP_REPORT.md').write_text('\n'.join(lines))
    if not no_plots:plots(combined,out,['20260912','20260916'])
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--looseness-output',required=True);p.add_argument('--data-root',required=True);p.add_argument('--bundles',default=str(ROOT/'outputs/inference_bundles/date_prelim_20261106'));p.add_argument('--output',required=True);p.add_argument('--cache-root',action='append',default=[]);p.add_argument('--data-kind',choices=('actual','synthetic'),required=True);p.add_argument('--no-plots',action='store_true');p.add_argument('--allow-new-analysis-for-rejected-cache',action='store_true')
    a=p.parse_args();r=run(a.looseness_output,a.data_root,a.bundles,a.output,a.cache_root,a.data_kind,a.no_plots,a.allow_new_analysis_for_rejected_cache);print(json.dumps(r,ensure_ascii=False,indent=2));return 0 if r['all_requested_normal_validated'] else 1

if __name__=='__main__':raise SystemExit(main())
