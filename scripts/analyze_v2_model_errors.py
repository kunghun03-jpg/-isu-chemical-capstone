"""CSV-only, source/date/RPM-stratified errors; no threshold optimization or training."""
import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime
from io import StringIO
from itertools import groupby
import json
import math
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.analyze_fault_signatures import CONDITIONS
from scripts.validate_v2_integration import write_csv,digest

CLASSES=('Normal','Bearing','Looseness','Unbalance')


def family(condition):
    if condition in ('Normal1','Normal2'): return 'Normal'
    if condition in ('IR','OR','Roller'): return 'Bearing'
    if condition in ('Looseness0.05','Looseness0.1','LoosenessFull'): return 'Looseness'
    if condition in ('Unbalance_-1','Unbalance_-2'): return 'Unbalance'
    raise ValueError('unknown condition')


def finite(value):
    value=float(value)
    if not math.isfinite(value): raise ValueError('nonfinite score/feature')
    return value


def boolean(value):
    if str(value).lower() in ('true','1'): return True
    if str(value).lower() in ('false','0'): return False
    raise ValueError('invalid decision')


def load(path,kind):
    source=str(Path(path).expanduser().resolve()); rows=[];excluded=[];seen=set()
    for r in csv.DictReader(StringIO(Path(source).read_text(encoding='utf-8-sig'))):
        try:
            condition=r['condition']; truth=family(condition);rpm=int(r['rpm']);date=r['date']
            datetime.strptime(date,'%Y%m%d')
            if rpm not in (900,1200,1800): raise ValueError('unsupported RPM')
            key=(r['path'],rpm)
            if key in seen: raise ValueError('duplicate path/RPM in source')
            seen.add(key)
            if r.get('truth') and r['truth']!=truth: raise ValueError('truth/condition mismatch')
            row={'source':source,'evidence':'existing_csv_'+kind,'data_kind':r.get('data_kind','not_recorded'),
                 'path':r['path'],'rpm':rpm,'date':date,'condition':condition,'truth':truth}
            if not row['path']: raise ValueError('missing path')
            if kind=='integration':
                if r.get('status')!='analyzed': raise ValueError('original analysis error: '+r.get('error','unknown'))
                score,threshold=finite(r['ae_score']),finite(r['ae_threshold']); abnormal=boolean(r['ae_anomaly'])
                if abnormal != (score>threshold): raise ValueError('stored AE rule inconsistent')
                raw=r.get('rf_raw_prediction',''); executed=boolean(r['rf_executed'])
                if bool(raw)!=executed or (not abnormal and executed): raise ValueError('RF gate inconsistent')
                row.update(ae_score=score,ae_threshold=threshold,ae_anomaly=abnormal,
                           false_positive=abnormal if truth=='Normal' else None,
                           missed_fault=not abnormal if truth!='Normal' else None,
                           rf_raw_prediction=raw,rf_executed=executed,final_candidate=r.get('final_candidate',''),
                           abstained=boolean(r['abstained']),reasons=r.get('reasons',''),
                           ae_versions=r.get('ae_versions',''),rf_model_version=r.get('rf_model_version',''))
                features=json.loads(r.get('features') or '{}')
                if not isinstance(features,dict): raise ValueError('features must be an object')
                for value in features.values():
                    if value is not None: finite(value)
                for f in ('rms_ac','one_x_amplitude','two_x_amplitude','energy_high','crest_factor_ac','envelope_rms_ac'):
                    if features.get(f) is not None: row[f]=finite(features[f])
            else:
                raw=r['raw_prediction'];row.update(rf_raw_prediction=raw,rf_executed=True,
                    final_candidate='',abstained=boolean(r['accepted']) is False if r.get('accepted') else None,
                    reasons=r.get('reasons',''),final_state='not_recorded_in_offline_csv')
            if raw and raw not in CLASSES: raise ValueError('invalid RF class')
            row['rf_correct']=raw==truth if raw else None
            rows.append(row)
        except (ValueError,KeyError,TypeError) as error:
            excluded.append({'source':source,'path':r.get('path',''),'reason':str(error)})
    return rows,excluded


def summarize(rows):
    groups=defaultdict(list)
    for r in rows: groups[(r['source'],r['rpm'],r['date'],r['condition'])].append(r)
    result=[]
    for (source,rpm,date,condition),group in sorted(groups.items()):
        ae=[r for r in group if 'ae_score' in r];rf=[r for r in group if r['rf_executed']]
        normal=family(condition)=='Normal'
        result.append(dict(source=source,rpm=rpm,date=date,condition=condition,files=len(group),ae_n=len(ae),
            false_positive_rate=sum(r['ae_anomaly'] for r in ae)/len(ae) if ae and normal else None,
            missed_fault_rate=sum(not r['ae_anomaly'] for r in ae)/len(ae) if ae and not normal else None,
            ae_score_median=float(np.median([r['ae_score'] for r in ae])) if ae else None,
            ae_score_q25=float(np.quantile([r['ae_score'] for r in ae],.25)) if ae else None,
            ae_score_q75=float(np.quantile([r['ae_score'] for r in ae],.75)) if ae else None,
            stored_thresholds=json.dumps(sorted({r['ae_threshold'] for r in ae})),
            rf_n=len(rf),rf_raw_accuracy=sum(r['rf_correct'] for r in rf)/len(rf) if rf else None,
            rf_predictions=json.dumps(dict(Counter(r['rf_raw_prediction'] for r in rf))),
            final_abstained_n=sum(r.get('abstained') is True for r in group),
            rf_scope='AE-gated subset' if ae else 'offline full CSV; final labels not recorded'))
    return result


def confusion(rows):
    counts=Counter()
    for r in rows:
        if r['rf_executed']:
            counts[(r['source'],r['rpm'],r['date'],r['truth'],r['rf_raw_prediction'])]+=1
    return [dict(source=k[0],rpm=k[1],date=k[2],truth=k[3],raw_prediction=k[4],count=v) for k,v in sorted(counts.items())]


def plots(rows,out):
    for i,source in enumerate(sorted({r['source'] for r in rows})):
        data=[r for r in rows if r['source']==source and 'ae_score' in r]
        for rpm in (900,1200,1800):
            group=[r for r in data if r['rpm']==rpm]
            if not group: continue
            labels=sorted({(r['date'],r['condition']) for r in group})
            fig,ax=plt.subplots(figsize=(max(8,len(labels)*.5),5))
            values=[[r['ae_score'] for r in group if (r['date'],r['condition'])==k] for k in labels]
            ax.boxplot(values,tick_labels=['/'.join(k) for k in labels]);ax.tick_params(axis='x',rotation=90)
            for threshold in sorted({r['ae_threshold'] for r in group}): ax.axhline(threshold,color='red',linestyle='--',label=f'stored threshold {threshold:.6g}')
            ax.set_ylabel('AE file score');ax.set_title(f'{rpm} RPM: date/condition (source {i})');ax.legend();fig.tight_layout();fig.savefig(out/f'ae_distribution_{i}_{rpm}.png');plt.close(fig)
            for feature in ('rms_ac','one_x_amplitude','two_x_amplitude'):
                fig,ax=plt.subplots(figsize=(7,5))
                for date,condition in labels:
                    part=[r for r in group if (r['date'],r['condition'])==(date,condition) and feature in r]
                    ax.scatter([r[feature] for r in part],[r['ae_score'] for r in part],label=f'{date}/{condition}',s=18)
                ax.set_xlabel(feature+' (raw)');ax.set_ylabel('AE score');ax.legend(fontsize=6);fig.tight_layout();fig.savefig(out/f'ae_feature_{feature}_{i}_{rpm}.png');plt.close(fig)
            fig,ax=plt.subplots(figsize=(max(10,len(labels)*.9),5))
            for j,key in enumerate(labels):
                for feature in ('rms_ac','one_x_amplitude','two_x_amplitude'):
                    vals=[r[feature] for r in group if (r['date'],r['condition'])==key and feature in r]
                    ax.scatter([f'{key[0]}/{key[1]}/{feature}']*len(vals),vals,s=10)
            ax.tick_params(axis='x',rotation=90);ax.set_ylabel('Feature value (raw; different features not directly comparable)');fig.tight_layout();fig.savefig(out/f'features_{i}_{rpm}.png');plt.close(fig)
    for i,((source,rpm,date),group) in enumerate(groupby(sorted([r for r in rows if r['rf_executed']],key=lambda r:(r['source'],r['rpm'],r['date'])),key=lambda r:(r['source'],r['rpm'],r['date']))):
        matrix=np.zeros((4,4),dtype=int)
        for r in group: matrix[CLASSES.index(r['truth']),CLASSES.index(r['rf_raw_prediction'])]+=1
        fig,ax=plt.subplots(figsize=(6,5));ax.imshow(matrix,cmap='Blues');ax.set_xticks(range(4),CLASSES,rotation=30);ax.set_yticks(range(4),CLASSES);ax.set_xlabel('Raw RF prediction');ax.set_ylabel('Truth');ax.set_title(f'{rpm}/{date} source={Path(source).parent.name}')
        for y in range(4):
            for x in range(4): ax.text(x,y,str(matrix[y,x]),ha='center',va='center')
        fig.tight_layout();fig.savefig(out/f'rf_confusion_{i}.png');plt.close(fig)


def run(integration,offline,output):
    out=Path(output).expanduser().resolve()
    if out==ROOT or ROOT in out.parents: raise ValueError('output must be outside repository')
    rows=[];excluded=[];hashes={}
    for kind,paths in (('integration',integration),('offline',offline)):
        for path in paths:
            resolved=str(Path(path).expanduser().resolve())
            if resolved in hashes: raise ValueError('duplicate source CSV')
            hashes[resolved]=digest(resolved);r,b=load(resolved,kind);rows.extend(r);excluded.extend(b)
    if not hashes: raise ValueError('input CSV required')
    out.mkdir(parents=True)
    coverage=[]
    for source in sorted(hashes):
        dates=sorted({r['date'] for r in rows if r['source']==source})
        for date in dates:
            for rpm in (900,1200,1800):
                for condition in CONDITIONS:
                    group=[r for r in rows if (r['source'],r['date'],r['rpm'],r['condition'])==(source,date,rpm,condition)]
                    coverage.append(dict(source=source,date=date,rpm=rpm,condition=condition,files=len(group),coverage='recorded' if group else 'not_recorded'))
    write_csv(out/'condition_coverage.csv',coverage)
    write_csv(out/'ae_error_analysis.csv',[r for r in rows if 'ae_score' in r])
    write_csv(out/'rf_error_analysis.csv',rows);write_csv(out/'rpm_condition_summary.csv',summarize(rows))
    write_csv(out/'rf_confusion.csv',confusion(rows));write_csv(out/'excluded_rows.csv',excluded);plots(rows,out)
    target=[r for r in rows if Path(r['path']).name=='20260919_132506_Vib.mat' and r['rpm']==1800 and r['condition']=='Unbalance_-1']
    changed=[p for p,h in hashes.items() if digest(p)!=h]
    report={'input_sha256':hashes,'changed_inputs':changed,'valid_rows':len(rows),'excluded_rows':len(excluded),'target_observed_rows':target,
        'evidence':'CSV-only analysis; no MAT inference performed by this command','model_changes':False}
    (out/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    (out/'MODEL_ERROR_ANALYSIS.md').write_text(f'''# AE/RF 오류 원인 조사\n\n입력 CSV 기반 분석: 유효 {len(rows)}, 제외 {len(excluded)}. 이 명령은 실제 MAT를 실행하지 않았다. source/data_kind 열에서 실제 통합 검증 기록과 합성 테스트를 구분한다. source가 다른 결과를 합쳐 성능을 계산하지 않는다.\n\n## 확인된 기록\n\n지정 사례 일치 행: {len(target)}. 실제 점수/임계값은 analysis.json 및 ae_error_analysis.csv에서 확인한다. 사용자가 제공한 0.375726 < 0.445470 사례는 정상 판정/RF 생략과 일관되지만, 이 값은 새 실행 결과로 생성하지 않는다.\n\n## 해석 한계\n\n- RF 원시 예측과 최종 후보/보류는 별개다. 웹 RF 정확도는 AE 이상 부분집합이며 오프라인 전체 Test 정확도와 직접 비교하지 않는다.\n- Normal1/2, 결함별, 날짜/RPM을 분리한다. 표본 0이면 성능 미확인이다.\n- 미탐지는 기존 점수가 기존 임계값 이하였다는 사실이다. 신호 강도·특징 중첩·날짜/부하 차이·모델 표현력은 가설이며 상관만으로 원인을 확정하지 않는다.\n- RF Unbalance→Looseness 또는 IR/OR/Roller→Bearing 패턴은 confusion CSV/PNG에 있는 실제 행만 근거로 삼는다.\n- 특징 차이는 인과관계가 아니며 작은 표본/Run 독립성 미확인, Train 중복 가능성이 남아 있다. 확장 Looseness는 기존 Train/Validation 파일과 겹칠 수 있어 독립 Test 성능으로 주장하지 않는다.\n- 날짜별 클래스 누락과 오류 제외를 확인한다. 보류를 오분류와 혼합하지 않는다.\n- 임계값 최적화, 모델 재학습, 정확도 개선 검증은 수행하지 않았다.\n''')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--integration',action='append',default=[]);p.add_argument('--offline-rf',action='append',default=[]);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(a.integration,a.offline_rf,a.output),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
