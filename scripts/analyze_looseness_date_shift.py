"""Full Looseness date-shift audit. AE/feature inference only; no RF, DB or tuning."""
import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime,timezone
import hashlib
import importlib.metadata
from io import StringIO
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import scipy
from scripts.analyze_fault_signatures import metadata,RPMS,NORMAL,BANDS
from vibration.inference import SavedModelInference
from vibration.web_inference import preview_mat
from vibration.candidate_analysis import candidate_analysis,FEATURE_LABELS
from scripts.validate_v2_integration import write_csv,digest

LOOSE=('Looseness0.05','Looseness0.1','LoosenessFull')
FEATURES=tuple(FEATURE_LABELS)
RTOL=1e-10
ATOL=1e-12
RATIO_FLOOR=1e-12


def discover(root,dates):
    root=Path(root).expanduser().resolve();rows=[];excluded=[];seen=set()
    for date in sorted(dates):
        folder=root/date
        if not folder.is_dir():excluded.append({'path':str(folder),'reason':'date_directory_missing'});continue
        for p in sorted(folder.rglob('*')):
            if p.suffix.lower()!='.mat' or not p.is_file():continue
            if not any(part in LOOSE+NORMAL for part in p.parts):continue
            try:
                if p.is_symlink() or not p.resolve().is_relative_to(root):raise ValueError('symlink_or_outside_root_excluded')
                m=metadata(p)
                if m['date']!=date or m['condition'] not in LOOSE+NORMAL:raise ValueError('out_of_scope')
                key=str(p.resolve())
                if key in seen:raise ValueError('duplicate_physical_path')
                seen.add(key);rows.append({'path':key,**{k:m[k] for k in ('date','rpm','condition')}})
            except ValueError as error:excluded.append({'path':str(p),'reason':str(error)})
    return rows,excluded


def inventory(rows,dates):
    counts=Counter((r['date'],r['rpm'],r['condition']) for r in rows)
    return [dict(date=d,rpm=r,condition=c,discovered_files=counts[d,r,c],readable_files='not_tested',read_failed_files='not_tested') for d in dates for r in RPMS for c in LOOSE+NORMAL]


def stats(values):
    a=np.asarray([v for v in values if v is not None],dtype=float)
    if not len(a):return dict(n=0,mean=None,median=None,min=None,max=None,q25=None,q75=None,q05=None,q95=None)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()),
                q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)),q05=float(np.quantile(a,.05)),q95=float(np.quantile(a,.95)))


def summaries(results,cases):
    groups=defaultdict(list)
    for r in results:groups[(r['date'],r['rpm'],r['condition'])].append(r)
    counts=Counter((r['date'],r['rpm'],r['condition']) for r in cases);out=[]
    for (date,rpm,condition),total in sorted(counts.items()):
        rows=groups[date,rpm,condition];good=[r for r in rows if r['status']=='ok']
        row=dict(date=date,rpm=rpm,condition=condition,total_files=total,processed_files=len(rows),success_files=len(good),
                 failed_files=sum(r['status']=='error' for r in rows),duplicate_files=sum(r['status']=='duplicate_content' for r in rows),
                 readable_files=sum(r.get('read_ok',False) for r in rows),read_failed_files=sum(not r.get('read_ok',False) for r in rows),
                 ae_threshold_values=json.dumps(sorted({r['ae_threshold'] for r in good})),ae_anomaly_n=sum(r['ae_anomaly'] for r in good),ae_anomaly_rate=sum(r['ae_anomaly'] for r in good)/len(good) if good else None)
        for key in ('ae_score','score_threshold_ratio',*FEATURES):
            for k,v in stats([r.get(key) for r in good]).items():row[key+'_'+k]=v
        out.append(row)
    return out


def comparisons(groups,dates):
    lookup={(r['date'],r['rpm'],r['condition']):r for r in groups};out=[]
    if len(dates)!=2:return out
    for rpm in RPMS:
        for condition in LOOSE+NORMAL:
            a=lookup.get((dates[0],rpm,condition));b=lookup.get((dates[1],rpm,condition))
            row=dict(rpm=rpm,condition=condition,date_a=dates[0],date_b=dates[1],n_a=a['success_files'] if a else 0,n_b=b['success_files'] if b else 0)
            if a and b and a['success_files'] and b['success_files']:
                row['ae_anomaly_rate_difference']=b['ae_anomaly_rate']-a['ae_anomaly_rate']
                for feature in ('ae_score',*FEATURES):
                    for aggregation in ('mean','median'):
                        key=feature+'_'+aggregation;x,y=a[key],b[key]
                        row[key+'_a'],row[key+'_b']=x,y
                        row[key+'_difference']=None if x is None or y is None else y-x
                        row[key+'_ratio_b_over_a']=y/x if x is not None and y is not None and abs(x)>RATIO_FLOOR else None
                        row[key+'_ratio_note']='denominator_small_or_missing' if x is None or abs(x)<=RATIO_FLOOR else ''
            out.append(row)
    return out


def segment_summary(rows):
    grouped=defaultdict(list)
    for row in rows:
        if row['status']!='ok':continue
        scores=json.loads(row['ae_segment_scores'])
        for i,value in enumerate(scores):grouped[(row['date'],row['rpm'],row['condition'],i)].append(value)
    return [dict(date=k[0],rpm=k[1],condition=k[2],segment_id=k[3],start_seconds=k[3]*8192/25600,end_seconds=(k[3]+1)*8192/25600,**stats(v)) for k,v in sorted(grouped.items())]


def control_comparison(groups,dates):
    lookup={(g['date'],g['rpm'],g['condition']):g for g in groups};out=[]
    for rpm in RPMS:
        for loose in LOOSE:
            for normal in NORMAL:
                parts=[lookup.get((d,rpm,c)) for d in dates for c in (loose,normal)]
                if any(p is None or not p['success_files'] for p in parts):continue
                a,n,b,m=parts
                row=dict(rpm=rpm,looseness=loose,normal_control=normal,date_a=dates[0],date_b=dates[1],n_loose_a=a['success_files'],n_normal_a=n['success_files'],n_loose_b=b['success_files'],n_normal_b=m['success_files'])
                for f in ('ae_score',*FEATURES):
                    key=f+'_median';values=[p[key] for p in parts]
                    if any(v is None for v in values):continue
                    row[f+'_loose_minus_normal_a']=a[key]-n[key]
                    row[f+'_loose_minus_normal_b']=b[key]-m[key]
                    row[f+'_date_difference_minus_normal_difference']=(b[key]-a[key])-(m[key]-n[key])
                out.append(row)
    return out


def reference_rows(paths,include_normal):
    lookup={};hashes={}
    for path in paths:
        path=Path(path).expanduser().resolve();hashes[str(path)]=digest(path)
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                if row.get('condition') in NORMAL and not include_normal:continue
                key=str(Path(row.get('resolved_path') or row['path']).expanduser().resolve())
                if key in lookup:raise ValueError('reference CSV duplicate path')
                lookup[key]=row
    return lookup,hashes


def parity(row,reference):
    if reference is None:return {'status':'not_reference'}
    try:
        if int(reference['rpm'])!=row['rpm'] or reference['date']!=row['date'] or reference['condition']!=row['condition']:raise ValueError('reference metadata mismatch')
        if not reference.get('input_sha256'):raise ValueError('reference MAT SHA missing')
        if reference.get('input_sha256') and reference['input_sha256']!=row['sha256']:raise ValueError('reference MAT SHA mismatch')
        if reference.get('ae_versions') and json.loads(reference['ae_versions']).get('autoencoder')!=row.get('model_version'):raise ValueError('reference AE artifact version mismatch')
        if reference.get('status')!='analyzed' or row['status']!='ok':raise ValueError('reference/current analysis not successful')
        expected={'ae_score':float(reference['ae_score']),'ae_threshold':float(reference['ae_threshold']),**json.loads(reference['features'])}
        differences={}
        for key in ('ae_score','ae_threshold',*FEATURES):
            a,b=row[key],expected[key]
            if a is None or b is None:
                if a is not None or b is not None:raise ValueError(key+' missing mismatch')
            else:
                if not np.isfinite([a,b]).all() or not np.isclose(a,b,rtol=RTOL,atol=ATOL):raise ValueError(key+' numerical mismatch')
                differences[key]=a-b
        flag=str(reference['ae_anomaly']).lower()
        if flag not in ('true','false','0','1'):raise ValueError('reference decision format')
        expected_flag=flag in ('true','1')
        if row['ae_anomaly']!=expected_flag:raise ValueError('AE decision mismatch')
        return {'status':'passed','differences':differences}
    except Exception as error:return {'status':'mismatch','reason':f'{type(error).__name__}: {error}'}


def process(case,engine,raw=None):
    row={**case,'status':'error','read_ok':False,'error':'','sha256':''}
    raw=Path(case['path']).read_bytes() if raw is None else raw
    preview_mat(raw);row['read_ok']=True;row['sha256']=hashlib.sha256(raw).hexdigest()
    result=engine.predict_bytes(raw,rpm=case['rpm'],filename='uploaded.mat');item=result['models']['autoencoder']
    evidence,_=candidate_analysis(raw,case['rpm'])  # feature extraction only; NEVER analyze_with_engine.
    row.update(status='ok',ae_score=item['file_score'],ae_threshold=item['file_threshold'],ae_anomaly=bool(item['predicted_anomaly']),
               score_threshold_ratio=item['file_score']/item['file_threshold'] if item['file_threshold']>RATIO_FLOOR else None,
               ae_segment_scores=json.dumps([s['score'] for s in item['segments']]),ae_segment_threshold=item['segment_threshold'],model_version='sha256:'+engine.manifest['artifacts'][engine.manifest['models']['autoencoder']['artifact']],bundle_id=result['bundle_id'],**evidence['features'])
    if not np.isfinite([row[k] for k in ('ae_score','ae_threshold',*FEATURES) if row[k] is not None]).all():raise ValueError('nonfinite result')
    return row


def plots(rows,out,dates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    good=[r for r in rows if r['status']=='ok']
    for rpm in RPMS:
        data=[r for r in good if r['rpm']==rpm]
        if not data:continue
        conditions=sorted({r['condition'] for r in data})
        for feature in ('score_threshold_ratio','rms_ac','peak_ac','one_x_amplitude','two_x_amplitude','energy_mid','energy_high'):
            fig,ax=plt.subplots(figsize=(max(8,len(conditions)*2),5));values=[];labels=[]
            for c in conditions:
                for d in dates:
                    v=[r[feature] for r in data if r['condition']==c and r['date']==d and r.get(feature) is not None]
                    if v:values.append(v);labels.append(f'{d}/{c}\nn={len(v)}')
            if values:
                ax.boxplot(values,tick_labels=labels);ax.tick_params(axis='x',rotation=45)
                ax.set_yscale('symlog',linthresh=RATIO_FLOOR);ax.set_ylabel(feature+' (symlog; linear near 0)')
                if feature=='score_threshold_ratio':ax.axhline(1,color='red',linestyle='--')
                ax.set_title(f'{rpm}RPM date/condition; all successful files');fig.tight_layout();fig.savefig(out/f'{feature}_{rpm}.png')
            plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5));labels=[];rates=[]
        for c in conditions:
            for d in dates:
                part=[r for r in data if r['condition']==c and r['date']==d]
                if part:labels.append(f'{d}/{c}\nn={len(part)}');rates.append(sum(r['ae_anomaly'] for r in part)/len(part))
        ax.bar(labels,rates);ax.set_ylim(0,1);ax.tick_params(axis='x',rotation=45);ax.set_ylabel('AE alarm fraction (not accuracy)');fig.tight_layout();fig.savefig(out/f'ae_alarm_rate_{rpm}.png');plt.close(fig)
        # Chronological filename order is measurement-time proxy, not confirmed Run identity.
        fig,ax=plt.subplots(figsize=(10,5))
        for d in dates:
            for c in conditions:
                part=sorted([r for r in data if r['date']==d and r['condition']==c],key=lambda r:r['path'])
                ax.plot([int(Path(r['path']).stem[9:11])*3600+int(Path(r['path']).stem[11:13])*60+int(Path(r['path']).stem[13:15]) for r in part],[r['ae_score'] for r in part],'.-',label=f'{d}/{c}')
        ax.set_xlabel('Filename seconds since midnight; not verified Run');ax.set_ylabel('AE file score');ax.tick_params(axis='x',rotation=90);ax.legend(fontsize=6);fig.tight_layout();fig.savefig(out/f'measurement_sequence_{rpm}.png');plt.close(fig)


def run(args):
    root=Path(args.data_root).expanduser().resolve();out=Path(args.output).expanduser().resolve();bundles=Path(args.bundles).expanduser().resolve()
    if out==ROOT or ROOT in out.parents or out==root or root in out.parents or out==bundles or bundles in out.parents:raise ValueError('output must be outside repository/data/bundles')
    if not root.is_dir():raise ValueError('data root missing')
    dates=sorted(args.dates)
    for d in dates:datetime.strptime(d,'%Y%m%d')
    if len(set(dates))!=2:raise ValueError('exactly two distinct dates required')
    all_cases,excluded=discover(root,dates);cases=[c for c in all_cases if args.include_normal or c['condition'] in LOOSE]
    if not cases and not args.inventory_only:raise ValueError('no selected MAT files; inspect inventory/discovery exclusions first')
    print(json.dumps({'looseness_files':sum(c['condition'] in LOOSE for c in all_cases),'normal_files':sum(c['condition'] in NORMAL for c in all_cases),'normal_analysis_enabled':args.include_normal,'normal_seconds_estimate':'after measured Looseness time; not yet known'},ensure_ascii=False),flush=True)
    config={'root':str(root),'bundles':str(bundles),'dates':dates,'include_normal':args.include_normal,'data_kind':args.data_kind,'script_sha256':digest(__file__),'cases':cases}
    refs,ref_hashes=reference_rows(args.reference,args.include_normal) if not args.inventory_only else ({},{})
    try:torch_version=importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError:torch_version='not_installed'
    config['runtime']={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch_version}
    config['reference_hashes']=ref_hashes
    config['implementation_hashes']={name:digest(ROOT/name) for name in ('vibration/inference.py','vibration/preprocessing.py','vibration/model.py','vibration/candidate_analysis.py','scripts/analyze_fault_signatures.py')}
    artifact_hashes={str(p):digest(p) for rpm in RPMS for p in (bundles/f'{rpm}rpm').glob('*') if p.is_file()}
    config['artifact_hashes']=artifact_hashes
    results=[]
    if args.resume:
        if json.loads((out/'run_config.json').read_text())!=config:raise ValueError('resume input/config/artifact/code differs')
        with (out/'progress.jsonl').open() as f:
            for line in f:results.append(json.loads(line)) # truncated/corrupt journal fails closed, never delete it
        for row in results:
            if row['sha256'] and digest(row['path'])!=row['sha256']:raise ValueError('processed MAT changed; use a new output')
    else:
        out.mkdir(parents=True);(out/'run_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    write_csv(out/'inventory.csv',inventory(all_cases,dates));write_csv(out/'discovery_exclusions.csv',excluded)
    if args.inventory_only:return {'inventory_only':True,'discovered':len(all_cases),'normal_files':sum(c['condition'] in NORMAL for c in all_cases)}
    if len({r['path'] for r in results})!=len(results):raise ValueError('duplicate paths in progress journal; preserve and investigate')
    processed={r['path'] for r in results};seen_hash={r['sha256']:r['path'] for r in results if r['sha256']};engines={};started=time.monotonic();elapsed=[];completed=False
    try:
        with (out/'progress.jsonl').open('a') as journal:
            for index,case in enumerate(sorted(cases,key=lambda c:(c['condition'] in NORMAL,c['path']))):
                if case['path'] in processed:continue
                if case['condition'] in NORMAL and elapsed:
                    if not any(r['condition'] in NORMAL for r in results):print(f'Normal preflight: {sum(c["condition"] in NORMAL for c in cases)} files; estimate {np.mean(elapsed)*sum(c["condition"] in NORMAL for c in cases):.1f}s from Looseness CPU timings',flush=True)
                begin=time.monotonic();row={**case,'status':'error','read_ok':False,'sha256':'','error':''}
                try:
                    raw=Path(case['path']).read_bytes();row['bytes_read']=True;row['sha256']=hashlib.sha256(raw).hexdigest()
                    preview_mat(raw);row['read_ok']=True
                    if row['sha256'] in seen_hash:row.update(status='duplicate_content',error='same SHA256 as '+seen_hash[row['sha256']])
                    else:
                        seen_hash[row['sha256']]=case['path']
                        if case['rpm'] not in engines:
                            try:engines[case['rpm']]=SavedModelInference(bundles/f'{case["rpm"]}rpm',models=('autoencoder',))
                            except Exception as error:engines[case['rpm']]=error
                        if isinstance(engines[case['rpm']],Exception):raise ValueError('AE bundle error: '+str(engines[case['rpm']]))
                        row=process(case,engines[case['rpm']],raw)
                    if digest(case['path'])!=row['sha256']:raise ValueError('MAT changed during processing')
                except Exception as error:row.update(status='error',error=f'{type(error).__name__}: {error}')
                row['reference_parity']=parity(row,refs.get(case['path']))
                row['elapsed_seconds']=time.monotonic()-begin;elapsed.append(row['elapsed_seconds']);results.append(row)
                journal.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');journal.flush();os.fsync(journal.fileno())
                print(f'{len(results)}/{len(cases)} {row["status"]} {case["path"]}',flush=True)
            completed=True
    finally:
        groups=summaries(results,cases);write_csv(out/'file_results.csv',[{**r,'reference_parity':json.dumps(r['reference_parity'],ensure_ascii=False)} for r in results])
        write_csv(out/'group_summary.csv',groups);write_csv(out/'date_comparison.csv',comparisons(groups,dates))
        write_csv(out/'segment_summary.csv',segment_summary(results))
        write_csv(out/'normal_control_comparison.csv',control_comparison(groups,dates))
        observed={r['path'] for r in results};missing_refs=sorted(set(refs)-observed);mismatches=[r['path'] for r in results if r['reference_parity']['status']=='mismatch']
        changed=[p for p,h in {**artifact_hashes,**ref_hashes,**{str(ROOT/name):h for name,h in config['implementation_hashes'].items()},str(Path(__file__).resolve()):config['script_sha256']}.items() if not Path(p).is_file() or digest(p)!=h]
        try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        except Exception:commit='unknown'
        summary={'completed':completed,'data_kind':args.data_kind,'executed_at_utc':datetime.now(timezone.utc).isoformat(),'git_commit':commit,
                 'runtime':config['runtime'],'data_root':str(root),'bundle_root':str(bundles),
                 'artifact_hashes':artifact_hashes,'feature_settings':{'channel':3,'fs':25600,'samples':65536,'bands_hz':BANDS,'order_tolerance_hz':1.,'envelope_carrier_hz':[2000,8000],'envelope_edge_trim_seconds':.1,'units':'raw, raw squared, dimensionless crest factor'},
                 'discovered_looseness':sum(c['condition'] in LOOSE for c in all_cases),'discovered_normal':sum(c['condition'] in NORMAL for c in all_cases),'selected':len(cases),'processed':len(results),'discovery_excluded':len(excluded),
                 'statuses':dict(Counter(r['status'] for r in results)),'reference_expected':len(refs),'reference_matched':sum(r['reference_parity']['status']=='passed' for r in results),
                 'reference_missing':missing_refs,'reference_mismatches':mismatches,'reference_tolerance':{'rtol':RTOL,'atol':ATOL},'changed_sources':changed,'elapsed_seconds':time.monotonic()-started,'implementation_hashes':config['implementation_hashes'],'groups':groups,
                 'limitations':['RF features are not direct AE inputs or causal explanations.','Run independence and labels/measurement conditions are unverified.','Normal may overlap Train/Validation; observed alarm fraction is not independent FPR.','Duplicate SHA256 files are excluded from successful denominators.']}
        summary['parity_verified']=bool(refs) and not missing_refs and not mismatches
        covered={(g['date'],g['rpm'],g['condition']) for g in groups if g['success_files']}
        summary['coverage_missing']=[dict(date=d,rpm=r,condition=c) for d in dates for r in RPMS for c in LOOSE if (d,r,c) not in covered]
        timings=[r['elapsed_seconds'] for r in results if r['status']=='ok' and r['condition'] in LOOSE]
        summary['normal_seconds_estimate']=float(np.mean(timings)*summary['discovered_normal']) if timings else None
        summary['scoped_discovery_exclusions']=[e for e in excluded if args.include_normal or e['reason']=='date_directory_missing' or any(p in LOOSE for p in Path(e['path']).parts)]
        summary['conclusions_ready']=not summary['scoped_discovery_exclusions'] and completed and summary['parity_verified'] and not summary['coverage_missing'] and not changed and all(r['status']=='ok' for r in results)
        (out/'analysis_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
        (out/'analysis_summary.md').write_text(f'''# Looseness 전체 날짜 조사\n\n데이터 종류: {args.data_kind}. 완료: {completed}. 성공/오류/중복: {summary['statuses']}. 발견 Looseness: {summary['discovered_looseness']}, Normal: {summary['discovered_normal']}.\n\n기존 표본 일치: {summary['reference_matched']}/{len(refs)}, 누락 {len(missing_refs)}, 불일치 {len(mismatches)}. 전체 결론 사용 가능: {summary['conclusions_ready']}. false이면 기존 가설을 전체 데이터 결론으로 제시하지 않는다.\n\n## 확인 자료\n\n날짜×RPM×강도별 실제 값은 group_summary.csv, 평균/중앙값 절대값·차이·안전 비율은 date_comparison.csv를 확인한다. ratio 분모 ≤1e-12는 공란이며 절대값/차이를 함께 본다. 그래프 symlog는 0 주변을 선형 처리하며 음수/0을 버리지 않는다. 파일명 시간 순서는 측정 구간의 단서이지 Run 독립성 근거가 아니다.\n\n## 원인 조사\n\n날짜별 진동 특징/점수 차이가 관찰돼도 센서 위치·방향·감도/교정, 장비·체결 토크/실제 간극, 부하·회전수, 수집 게인/단위, 실험 순서와 라벨·Run 로그가 없으면 원인을 확정하지 않는다. AE가 날짜를 학습했다거나 라벨이 틀렸다고 단정하지 않는다.\n\n## 다음 우선순위\n\n데이터 품질/라벨/중복 검토 → AE 학습 split 및 정상 기준 출처 검토 → 독립 Run/날짜 일반화 평가 → 필요 시 별도 승인된 임계값 정책 검토. 본 실행은 모델·임계값을 변경하지 않았다. Normal은 학습/검증 중복 가능성이 있어 독립 오탐률로 주장하지 않는다.\n''')
    date_totals=[]
    for date in dates:
        for kind,conditions in (('Looseness',LOOSE),('Normal1',('Normal1',)),('Normal2',('Normal2',))):
            part=[r for r in results if r['date']==date and r['condition'] in conditions and r['status']=='ok']
            if part:date_totals.append(dict(date=date,group=kind,n=len(part),alarms=sum(r['ae_anomaly'] for r in part),rate=sum(r['ae_anomaly'] for r in part)/len(part)))
    summary['date_totals']=date_totals
    paired=[r for r in comparisons(groups,dates) if r['condition'] in LOOSE and 'ae_score_median_difference' in r]
    summary['paired_direction_counts']={'paired_groups':len(paired),'higher_score_median_b':sum(r['ae_score_median_difference']>0 for r in paired),'higher_rms_median_b':sum(r['rms_ac_median_difference']>0 for r in paired)}
    (out/'analysis_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    with (out/'analysis_summary.md').open('a') as f:
        f.write('\n## 날짜별 실제 관측 (성공 파일만, 강도별 불균형 가능)\n\n|날짜|구분|n|AE 이상수|비율|\n|---|---|---:|---:|---:|\n')
        for r in date_totals:f.write(f"|{r['date']}|{r['group']}|{r['n']}|{r['alarms']}|{r['rate']:.6g}|\n")
        f.write('\n동일 RPM/강도별 방향 집계: '+json.dumps(summary['paired_direction_counts'],ensure_ascii=False)+'\n')
        f.write('\n미검증 Looseness 날짜/RPM/강도 조합: '+json.dumps(summary['coverage_missing'],ensure_ascii=False)+'\n')
    if completed and not args.no_plots:plots(results,out,dates)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True);p.add_argument('--dates',nargs=2,default=['20260912','20260916']);p.add_argument('--bundles',default=str(ROOT/'outputs/inference_bundles/date_prelim_20261106'));p.add_argument('--output',required=True);p.add_argument('--data-kind',choices=('actual','synthetic'),required=True);p.add_argument('--reference',action='append',default=[]);p.add_argument('--include-normal',action='store_true');p.add_argument('--inventory-only',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--no-plots',action='store_true')
    a=p.parse_args();r=run(a);print(json.dumps({k:v for k,v in r.items() if k not in ('groups','artifact_hashes')},ensure_ascii=False,indent=2));return 0 if a.inventory_only or r['conclusions_ready'] else 1

if __name__=='__main__':raise SystemExit(main())
