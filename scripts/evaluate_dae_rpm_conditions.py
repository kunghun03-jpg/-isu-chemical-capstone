"""Read-only, stratified DAE fixed-threshold evaluation; never authorizes V3 RPMs."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.analyze_fault_signatures import discover,CONDITIONS,RPMS
from scripts.validate_v2_integration import write_csv
from scripts.validate_dae_bundle_mat import provenance
from vibration.dae_bundle import FILES,sha256_file,runtime_versions
from vibration.dae_inference import SavedDAEBundleInference,SavedDAEInference


def interval(alarms,n):
    if not n:return None,None
    p=alarms/n;z=1.959963984540054;den=1+z*z/n
    center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-half),min(1.,center+half)


def distribution(values):
    if not values:return {'n':0,'mean':None,'median':None,'q25':None,'q75':None,'min':None,'max':None}
    a=np.asarray(values,dtype=float)
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),'q25':float(np.quantile(a,.25)),
            'q75':float(np.quantile(a,.75)),'min':float(a.min()),'max':float(a.max())}


def aggregate(rows):
    rpms=[];conditions=[]
    for rpm in RPMS:
        selected=[r for r in rows if r['rpm']==rpm];good=[r for r in selected if r['status']=='ok']
        normal=[r for r in good if r['family']=='Normal'];fault=[r for r in good if r['family']!='Normal']
        item={'rpm':rpm,'selected_files':len(selected),'success_unique_files':len(good),'error_files':sum(r['status']=='error' for r in selected),
              'duplicate_files':sum(r['status']=='duplicate_content' for r in selected),'normal_n':len(normal),'fault_n':len(fault)}
        for name,part in (('normal',normal),('fault',fault)):
            alarms=sum(r['threshold_alarm'] for r in part);lo,hi=interval(alarms,len(part))
            item.update({name+'_alarm_n':alarms,name+'_alarm_fraction':alarms/len(part) if part else None,
                         name+'_alarm_wilson95_low':lo,name+'_alarm_wilson95_high':hi})
            item.update({name+'_mse_'+k:v for k,v in distribution([r['mse'] for r in part]).items()})
        item['observed_auroc']=float(roc_auc_score([r['family']!='Normal' for r in good],[r['mse'] for r in good])) if normal and fault else None
        item['observed_auprc']=float(average_precision_score([r['family']!='Normal' for r in good],[r['mse'] for r in good])) if normal and fault else None
        item['normal_alarms_observed']=item['normal_alarm_n']
        item['fault_below_threshold_observed']=item['fault_n']-item['fault_alarm_n']
        item['metric_scope']='독립성 미확인 예비 평가; 폴더 레이블 기준'
        rpms.append(item)
    groups=defaultdict(list)
    for r in rows:groups[(r['rpm'],r['date'],r['condition'])].append(r)
    for (rpm,date,condition),part in sorted(groups.items()):
        good=[r for r in part if r['status']=='ok'];n=len(good);alarm=sum(r['threshold_alarm'] for r in good)
        conditions.append(dict(rpm=rpm,date=date,condition=condition,selected_n=len(part),valid_unique_n=n,
            error_n=sum(r['status']=='error' for r in part),duplicate_n=sum(r['status']=='duplicate_content' for r in part),
            alarm_n=alarm,alarm_fraction=alarm/n if n else None,**{'mse_'+k:v for k,v in distribution([r['mse'] for r in good]).items()}))
    return rpms,conditions


def overlap_evidence(source,selected):
    try:result=provenance(source)
    except Exception as e:result={'error':f'{type(e).__name__}: {e}','supported_rpms':None}
    dates={r['date'] for r in selected};observed=[]
    for filename,metadata in result.get('metadata',{}).items():
        for key,values in metadata.get('observations',{}).items():
            match=sorted(dates.intersection(values.get('date_counts',{})))
            if match:observed.append({'dataset':filename,'metadata_key':key,'evaluation_dates_in_metadata':match,'interpretation':'potential date overlap, not proof of exact file/Run or training membership'})
    result['potential_date_overlap']=observed
    result['exact_train_test_file_mapping_verified']=False
    result['run_independence_verified']=False
    return result


def plots(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for rpm in RPMS:
        good=[r for r in rows if r['rpm']==rpm and r['status']=='ok']
        if not good:continue
        keys=sorted({(r['date'],r['condition']) for r in good});values=[];labels=[]
        for date,condition in keys:
            part=[r['mse'] for r in good if (r['date'],r['condition'])==(date,condition)]
            values.append(part);labels.append(f'{date}/{condition}\nn={len(part)}')
        fig,ax=plt.subplots(figsize=(max(10,len(labels)*.45),5));ax.boxplot(values,tick_labels=labels)
        ax.axhline(good[0]['threshold'],color='red',linestyle='--',label='stored fixed threshold')
        ax.set_yscale('symlog',linthresh=1e-8);ax.set_ylabel('DAE whole-file MSE (symlog; linear near zero)');ax.tick_params(axis='x',rotation=90)
        ax.set_title(f'{rpm} RPM / date / original condition — exploratory');ax.legend();fig.tight_layout();fig.savefig(out/f'dae_mse_{rpm}.png');plt.close(fig)


def evaluate(data_root,bundle,output,*,count=10,all_files=False,dates=None,source=None,data_kind='actual',no_plots=False):
    root=Path(data_root).expanduser().resolve();folder=Path(bundle).expanduser().resolve();out=Path(output).expanduser().resolve()
    if out==ROOT or ROOT in out.parents or any(out==p or p in out.parents for p in (root,folder)):raise ValueError('output must be outside repository/data/bundle')
    if source and (out==Path(source).expanduser().resolve() or Path(source).expanduser().resolve() in out.parents):raise ValueError('output must be outside legacy source')
    records=discover([root],count=sys.maxsize if all_files else count)
    if dates:
        for date in dates:datetime.strptime(date,'%Y%m%d')
        records=[r for r in records if r.get('date') in dates or r['status']=='excluded']
    selected=[r for r in records if r['status']=='selected']
    if not selected:raise ValueError('No valid MAT cases selected')
    out.mkdir(parents=True);write_csv(out/'discovery_inventory.csv',records)
    snapshot={str(p):sha256_file(p) for p in [folder/'manifest.json',*[folder/name for name in FILES]] if p.is_file()}
    runtime_report=os.environ.get('ISU_CHEMICAL_DAE_RUNTIME_REPORT','')
    if runtime_report and Path(runtime_report).expanduser().is_file():snapshot[str(Path(runtime_report).expanduser().resolve())]=sha256_file(Path(runtime_report).expanduser())
    engine=None;bundle_error='';manifest={};rows=[];duplicates={};started=time.monotonic();completed=False
    try:
        engine=SavedDAEBundleInference(folder) # full integrity, runtime evidence and model/statistics checks
        manifest=engine.manifest
    except Exception as e:bundle_error=f'{type(e).__name__}: {e}'
    try:
        for i,record in enumerate(selected):
            r={k:record[k] for k in ('path','date','rpm','condition','family')}
            r.update(sha256='',status='error',error='',mse=None,threshold=None,threshold_alarm=None,offline_prediction='',data_kind=data_kind,
                inference_scope='offline_fixed_threshold_only',evaluation_authorizes_v3=False,model_version=manifest.get('model_version','unavailable'))
            try:
                if engine is None:raise ValueError('bundle_preflight_failed: '+bundle_error)
                path=Path(r['path']);raw=path.read_bytes();r['sha256']=hashlib.sha256(raw).hexdigest()
                if r['sha256'] in duplicates:
                    prior=duplicates[r['sha256']];r.update(status='duplicate_content',duplicate_of=prior['path'],mse=prior['mse'],threshold=prior['threshold'],threshold_alarm=prior['threshold_alarm'],offline_prediction=prior['offline_prediction'])
                else:
                    # Existing standalone diagnostic kernel; no public RPM check is removed or changed.
                    value=SavedDAEInference.predict_bytes(engine,raw)
                    if not np.isfinite([value['mse'],value['threshold']]).all():raise ValueError('nonfinite MSE/threshold')
                    r.update(status='ok',mse=value['mse'],threshold=value['threshold'],threshold_alarm=bool(value['mse']>value['threshold']),offline_prediction=value['prediction'])
                    duplicates[r['sha256']]=dict(r)
                if sha256_file(path)!=r['sha256']:raise ValueError('MAT changed during evaluation')
            except Exception as e:r.update(status='error',error=f'{type(e).__name__}: {e}',mse=None,threshold_alarm=None,offline_prediction='')
            rows.append(r);write_csv(out/'dae_results.csv',rows)
            print(f'{i+1}/{len(selected)} {r["status"]} {r["rpm"]}/{r["date"]}/{r["condition"]} {r["path"]}',flush=True)
        completed=True
    finally:
        rpm_summary,condition_summary=aggregate(rows);write_csv(out/'dae_rpm_summary.csv',rpm_summary);write_csv(out/'dae_condition_summary.csv',condition_summary)
        training=overlap_evidence(source,selected)
        changed=[p for p,h in snapshot.items() if not Path(p).is_file() or sha256_file(p)!=h]
        evaluation_dates=sorted(set(dates) if dates else {r['date'] for r in selected})
        coverage=[]
        for rpm in RPMS:
            for date in evaluation_dates:
                for condition in CONDITIONS:
                    part=[r for r in rows if (r['rpm'],r['date'],r['condition'])==(rpm,date,condition)]
                    coverage.append(dict(rpm=rpm,date=date,condition=condition,selected_n=sum((r['rpm'],r.get('date'),r.get('condition'))==(rpm,date,condition) for r in selected),valid_unique_n=sum(r['status']=='ok' for r in part)))
        plan=[]
        for item in rpm_summary:
            missing=[c for c in coverage if c['rpm']==item['rpm'] and c['valid_unique_n']==0]
            reasons=['Training/file split and Run independence remain unverified.','Predeclare target normal alarm limit and per-condition detection targets; do not optimize the saved threshold on this Test.','Review observed rates and uncertainty; no automatic acceptance threshold has been invented.']
            if missing:reasons.append('Missing valid date/condition strata: '+str(len(missing)))
            if item['error_files']:reasons.append('Investigate excluded inference errors: '+str(item['error_files']))
            if item['duplicate_files']:reasons.append('Resolve identical-content duplicates; they are excluded from rate denominators.')
            plan.append({'rpm':item['rpm'],'calculation_succeeded_on_selected_unique_files':item['success_unique_files']>0,
                'activation_status':'blocked_pending_independent_review','registration_performed':False,'additional_verification':reasons})
        report={'completed':completed,'data_kind':data_kind,'status':'offline_preliminary_evaluation' if engine else 'blocked_bundle_validation',
            'data_root':str(root),'bundle_folder':str(folder),'fixed_threshold':engine.threshold if engine else None,'bundle_preflight_error':bundle_error,'bundle_id':manifest.get('bundle_id'),'model_version':manifest.get('model_version'),
            'declared_supported_rpms':manifest.get('supported_rpms'),'rpm_evidence':manifest.get('rpm_evidence'),'training_provenance':manifest.get('training_provenance'),
            'evaluation_script_sha256':sha256_file(__file__),'runtime_validation_mode':('exact_manifest_versions' if manifest.get('runtime_versions')==runtime_versions() else 'validated_external_evidence') if engine else 'blocked','runtime':runtime_versions(),'runtime_report':runtime_report,'artifact_sha256':snapshot,'changed_artifacts':changed,
            'selection':{'method':'all valid files' if all_files else 'equally spaced indices in lexically sorted paths per RPM/date/original-condition; outcome-independent',
                'files_per_stratum':None if all_files else count,'dates_filter':dates,'discovered_records':len(records),'selected_files':len(selected)},
            'sampling_shortfalls':[c for c in coverage if not all_files and c['valid_unique_n']<count],'result_statuses':dict(Counter(r['status'] for r in rows)),'unprocessed_files':len(selected)-len(rows),'coverage':coverage,
            'rpm_summary':rpm_summary,'training_audit':training,'activation_plan':plan,'public_rpm_policy_unchanged':True,
            'independent_test_performance_validated':False,'threshold_policy':'original bundle threshold; never tuned',
            'limitations':['독립성 미확인 예비 평가','Folder labels are evaluation annotations, not proof of physical faults.','Normal may overlap training/validation. Error/duplicate exclusions can bias observed rates.','One runtime compatibility input does not validate all operating conditions.','No AE/RF/DB/Streamlit changes or score-scale comparison.'],
            'elapsed_seconds':time.monotonic()-started,'created_at_utc':datetime.now(timezone.utc).isoformat()}
        (out/'dae_validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
        lines=['# DAE RPM·조건 오프라인 검증',f'\n데이터 종류: {data_kind}. 상태: {report["status"]}, 완료: {completed}. 선택 {len(selected)}, 처리 {len(rows)}, 상태 {report["result_statuses"]}.',
            '\n**독립성 미확인 예비 평가**다. V3 RPM 등록/활성화는 수행하지 않았다. 실제 점수는 기존 float32 재구성 MSE와 저장 threshold의 비교이며 정확도/신뢰도가 아니다.',
            '\n|RPM|계산 성공 고유 파일|Normal 이상/유효|결함 이상/유효|Normal 이상 비율|결함 이상 비율|',
            '|---:|---:|---|---|---:|---:|']
        for r in rpm_summary:lines.append(f'|{r["rpm"]}|{r["success_unique_files"]}|{r["normal_alarm_n"]}/{r["normal_n"]}|{r["fault_alarm_n"]}/{r["fault_n"]}|{r["normal_alarm_fraction"]}|{r["fault_alarm_fraction"]}|')
        lines+=['\n## MSE 분포 관측 요약','\n|RPM|Normal 중앙값|결함 중앙값|Normal IQR|결함 IQR|관측 AUROC|','|---:|---:|---:|---|---|---:|']
        for r in rpm_summary:lines.append(f"|{r['rpm']}|{r['normal_mse_median']}|{r['fault_mse_median']}|{r['normal_mse_q25']}–{r['normal_mse_q75']}|{r['fault_mse_q25']}–{r['fault_mse_q75']}|{r['observed_auroc']}|")
        lines+=['\n고정 threshold의 관측 오류 후보 (폴더 레이블 기준):']
        for r in rpm_summary:lines.append(f"- {r['rpm']}RPM: 정상 이상 {r['normal_alarms_observed']}, 결함 threshold 이하 {r['fault_below_threshold_observed']}. 오류/중복은 분모에서 제외됨.")
        lines+=['\n## 필수 질문과 해석']
        for p in plan:lines.append(f'\n- {p["rpm"]}RPM 실제 계산: 선택 고유 파일 성공 여부={p["calculation_succeeded_on_selected_unique_files"]} (data_kind를 확인). V3 활성화={p["activation_status"]}.')
        lines+=['\n정상/결함 MSE 분리는 RPM별 중앙값·IQR·AUROC/AUPRC와 날짜/원래 조건 분포에서 관측 수준으로 확인한다. 클래스가 없으면 지표는 공란이다. 적은 표본의 Wilson 구간도 파일 독립성을 보장하지 않는다.',
            '\n기존 threshold를 계속 운영에 써도 되는지는 아직 확정하지 않는다. 고정 threshold의 정상 이상 비율과 결함 검출 비율을 함께 검토하고 사전에 정한 운영 목표 및 독립 날짜/Run 시험이 필요하다. 이 Test에서 threshold를 재선택하지 않는다.',
            '\ntraining_audit의 날짜/RPM 관측은 학습 범위의 증거가 아니다. 원본 정상 NPZ의 split/파일/Run 매핑과 평가 파일 중복을 추가 확인해야 한다. supported_rpms=null과 unverified를 임의 수정하지 않는다.',
            '\n## 별도 활성화 변경 계획',
            '\n1. 실제 계산 오류·중복·범위 누락 해결 → 2. 학습/정상 기준 출처 및 독립 split 확정 → 3. 사전 운영 목표와 날짜/RPM/약한 결함별 고정 threshold 결과·불확실성 검토 → 4. 성능이 낮거나 자료 부족한 RPM은 차단 유지 → 5. 충분한 RPM만 근거/해시를 첨부한 별도 PR로 검토한다. 이 도구는 어떤 manifest도 수정하지 않는다.']
        if bundle_error:lines.append('\n번들/환경 검증 오류: '+bundle_error)
        (out/'README.md').write_text('\n'.join(lines))
    if not no_plots:plots(rows,out)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True);p.add_argument('--bundle',required=True);p.add_argument('--runtime-report');p.add_argument('--source');p.add_argument('--output',default=str(Path.home()/'dae-evaluation'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')));p.add_argument('--files-per-stratum',type=int,default=10);p.add_argument('--all-files',action='store_true');p.add_argument('--dates',nargs='+');p.add_argument('--data-kind',choices=('actual','synthetic'),required=True);p.add_argument('--no-plots',action='store_true')
    a=p.parse_args();previous=os.environ.get('ISU_CHEMICAL_DAE_RUNTIME_REPORT')
    try:
        if a.runtime_report:os.environ['ISU_CHEMICAL_DAE_RUNTIME_REPORT']=str(Path(a.runtime_report).expanduser().resolve())
        report=evaluate(a.data_root,a.bundle,a.output,count=a.files_per_stratum,all_files=a.all_files,dates=a.dates,source=a.source,data_kind=a.data_kind,no_plots=a.no_plots)
    finally:
        if previous is None:os.environ.pop('ISU_CHEMICAL_DAE_RUNTIME_REPORT',None)
        else:os.environ['ISU_CHEMICAL_DAE_RUNTIME_REPORT']=previous
    print(json.dumps({'output':str(Path(a.output).expanduser().resolve()),'status':report['status'],'result_statuses':report['result_statuses'],'rpm_summary':report['rpm_summary'],'activation_status':{str(p['rpm']):p['activation_status'] for p in report['activation_plan']}},ensure_ascii=False,indent=2,allow_nan=False))
    return 0 if report['completed'] and not report['bundle_preflight_error'] and not report['changed_artifacts'] and not report['result_statuses'].get('error') else 1

if __name__=='__main__':raise SystemExit(main())
