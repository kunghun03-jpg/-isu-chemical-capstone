"""Read-only MAT/V2 parity checks; only a fresh external report DB is written."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vibration.dashboard_v2 import DashboardConfig, analyze
from vibration.candidate_analysis import analyze_with_engine
from vibration.candidate_engine import FEATURES
from vibration.candidate_history import read_candidate
from vibration.history import HistoryStore

PRIORITY = 'data/20260919/20260919_1800RPM/Unbalance_-1/Vib/20260919_132506_Vib.mat'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases(path, data_root):
    """Condition/path are audit metadata, never inference features."""
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Test 목록이 비었습니다.')
    seen = set()
    for row in rows:
        if not all(row.get(k) for k in ('path', 'rpm', 'raw_prediction')):
            raise ValueError('path/rpm/raw_prediction 필수 열 오류')
        row['rpm'] = int(row['rpm'])
        if row['rpm'] not in (900,1200,1800) or row['raw_prediction'] not in ('Normal','Bearing','Looseness','Unbalance'):
            raise ValueError('RPM 또는 오프라인 클래스 오류')
        path = Path(row['path']).expanduser()
        row['resolved_path'] = str((path if path.is_absolute() else Path(data_root)/path).resolve())
        key = (row['resolved_path'],row['rpm'])
        if key in seen:
            raise ValueError('중복 Test 파일/RPM')
        seen.add(key)
    return sorted(rows,key=lambda r: (not r['path'].replace('\\','/').endswith(PRIORITY),r['rpm'],r['path']))


def compare(ae_abnormal, raw_prediction, offline):
    if not ae_abnormal:
        return 'skipped_ae_normal'
    if raw_prediction is None:
        return 'rf_unavailable'
    return 'match' if raw_prediction == offline else 'mismatch'


def check_db(db, result, versions, report):
    store=HistoryStore(db)
    identity, created=store.save(result,versions,candidate_report=report)
    duplicate, second_created=store.save(result,versions,candidate_report=report)
    if not created or second_created or duplicate != identity:
        raise ValueError('중복 저장 계약 불일치')
    expected=store.detail(identity)
    if read_candidate(store,identity) != report:
        raise ValueError('후보 저장/재조회 불일치')
    # A new Python process verifies persistent state after all writer connections closed.
    code="""import json,sys
from vibration.history import HistoryStore
from vibration.candidate_history import read_candidate
s=HistoryStore(sys.argv[1]); i=int(sys.argv[2])
print(json.dumps({'detail':s.detail(i),'candidate':read_candidate(s,i)},ensure_ascii=False))"""
    env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT))
    restarted=json.loads(subprocess.check_output([sys.executable,'-B','-c',code,str(db),str(identity)],env=env,text=True,cwd=ROOT))
    if restarted != {'detail':expected,'candidate':report}:
        raise ValueError('프로세스 재시작 후 이력 불일치')
    # AE-only compatibility uses a second isolated DB, never adds a fake old production row.
    legacy=HistoryStore(Path(db).with_name('ae_only.sqlite3'))
    old,_=legacy.save(result,versions)
    if read_candidate(legacy,old) is not None or legacy.detail(old)['models'] != expected['models']:
        raise ValueError('AE-only 이력 호환 오류')
    return {'analysis_id':identity,'duplicate_id':duplicate,'restarted':True,'ae_only_compatible':True}


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def run(args):
    output=Path(args.output).expanduser().resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError('출력은 Git 저장소 밖이어야 합니다.')
    cases=load_cases(args.test_predictions,args.data_root)
    # mkdir without exist_ok forbids overwriting existing results/DBs.
    output.mkdir(parents=True)
    ae_root=Path(args.ae_bundles).expanduser().resolve()
    rf=Path(args.rf_bundle).expanduser().resolve()
    snapshot={str(p):digest(p) for p in [Path(args.test_predictions),rf/'manifest.json',rf/'training_report.json',rf/'model.joblib'] if p.is_file()}
    try:
        rf_manifest=json.loads((rf/'manifest.json').read_text())
    except (OSError, ValueError):
        rf_manifest={}
    for rpm in sorted({c['rpm'] for c in cases}):
        folder=ae_root/f'{rpm}rpm'
        for artifact in folder.iterdir() if folder.is_dir() else []:
            if artifact.is_file(): snapshot[str(artifact)]=digest(artifact)
    results=[]; comparisons=[]
    previous={k:os.environ.get(k) for k in ('ISU_CHEMICAL_CANDIDATE_BUNDLE','ISU_CHEMICAL_CANDIDATE_LABELS')}
    os.environ['ISU_CHEMICAL_CANDIDATE_BUNDLE']=str(rf)
    # Preserve declared existing label setting; no policy relaxation/default enable.
    os.environ['ISU_CHEMICAL_CANDIDATE_LABELS']=previous['ISU_CHEMICAL_CANDIDATE_LABELS'] or '0'
    try:
        for index,case in enumerate(cases):
            row={k:case.get(k,'') for k in ('path','resolved_path','rpm','condition','date')}
            row.update(data_kind=args.data_kind,status='error',rf_executed=False,rf_raw_prediction='',final_candidate='',reasons='',db_status='not_run')
            comparison={**row,'offline_raw_prediction':case['raw_prediction'],'comparison':'analysis_error'}
            try:
                path=Path(case['resolved_path']); raw=path.read_bytes(); before=hashlib.sha256(raw).hexdigest()
                row['input_sha256']=before
                # Neutral display name prevents labels or actual filenames entering inference.
                result,versions=analyze(raw,case['rpm'],'uploaded.mat',DashboardConfig('autoencoder',ae_root))
                report,_=analyze_with_engine(raw,case['rpm'],result)
                item=result['models']['autoencoder'];engine=report.get('engine') or {}
                row.update(status='analyzed',ae_score=item['file_score'],ae_threshold=item['file_threshold'],ae_anomaly=item['predicted_anomaly'],
                           ae_versions=json.dumps(versions,sort_keys=True),rf_executed=bool(engine),
                           rf_execution_status='skipped_ae_normal' if not item['predicted_anomaly'] else 'completed' if engine else 'failed_or_unavailable',
                           rf_model_version=engine.get('model_version',rf_manifest.get('model_version','unavailable')),
                           rf_model_sha256=engine.get('model_sha256',rf_manifest.get('model_sha256','unavailable')),
                           rf_raw_prediction=engine.get('raw_prediction',''),final_candidate=report['candidate'],abstained=report['abstained'],
                           reasons=json.dumps(report['reasons'],ensure_ascii=False),features=json.dumps(report['features'],sort_keys=True,allow_nan=False),
                           feature_order=json.dumps(list(FEATURES)),measurement=json.dumps(report.get('measurement',{}),sort_keys=True),
                           feature_config=json.dumps(rf_manifest.get('feature_config'),sort_keys=True),
                           feature_schema=engine.get('feature_schema',rf_manifest.get('feature_schema')))
                comparison.update(row,comparison=compare(bool(item['predicted_anomaly']),engine.get('raw_prediction'),case['raw_prediction']))
                if digest(path) != before:
                    raise ValueError('검증 중 MAT 파일 변경 감지')
                # One DB per input also makes repeated MAT entries independent audit cases.
                db=output/f'case_{index:04d}'/'integration.sqlite3';db.parent.mkdir()
                row['db_verification']=json.dumps(check_db(db,result,versions,report),sort_keys=True)
                row['db_status']='passed'
            except Exception as error:
                row['error']=f'{type(error).__name__}: {error}'
                if row['status']=='analyzed':
                    row['db_status']='failed'
                comparison.setdefault('error',row['error'])
            comparison['db_status']=row['db_status']
            results.append(row);comparisons.append(comparison)
    finally:
        for k,v in previous.items():
            if v is None: os.environ.pop(k,None)
            else: os.environ[k]=v
    changed=[p for p,h in snapshot.items() if not Path(p).is_file() or digest(p)!=h]
    counts=Counter(r['comparison'] for r in comparisons)
    summary={'data_kind':args.data_kind,'executed_at_utc':datetime.now(timezone.utc).isoformat(),'files':len(results),
             'comparison_counts':dict(counts),'conditions':dict(Counter(r.get('condition','unknown') for r in results)),
             'comparison_by_rpm':{str(rpm):dict(Counter(r['comparison'] for r in comparisons if r['rpm']==rpm)) for rpm in (900,1200,1800)},
             'reference_csv':str(Path(args.test_predictions).resolve()),
             'reference_is_bundle_test_csv':Path(args.test_predictions).resolve()==rf/'test_predictions.csv',
             'ae_bundle_root':str(ae_root),'rf_bundle':str(rf),
             'unrepresented_conditions':[c for c in ('Normal1','Normal2','IR','OR','Roller','Unbalance_-1','Unbalance_-2','Looseness0.05','Looseness0.1','LoosenessFull') if not any(r.get('condition')==c for r in results)],
             'labels_enabled_setting':previous['ISU_CHEMICAL_CANDIDATE_LABELS'] or '0',
             'rpm_counts':dict(Counter(r['rpm'] for r in results)),'ae_anomaly_counts':dict(Counter(str(r.get('ae_anomaly','error')) for r in results)),
             'candidate_counts':dict(Counter(r.get('final_candidate','') for r in results)),
             'abstained_count':sum(r.get('abstained') is True for r in results),
             'db_failures':sum(r['db_status']=='failed' for r in results),'input_artifact_sha256':snapshot,'changed_sources':changed,
             'priority_file_included':any(c['path'].replace('\\','/').endswith(PRIORITY) for c in cases),
             'limitations':['No retraining or policy adjustment. No new performance claim.',
                 'CSV-only reference cannot establish original feature/model identity; inspect diagnostics on mismatch.',
                 'RF executed means a prediction was returned; failed attempts are separately marked.',
                 'Production DB is never opened. Fresh report DBs only.','Run independence is not established.'],
             'diagnostic_steps':['Check RPM, RF model_version/model_sha256 and original training report.',
                 'Compare feature_order, feature_schema, feature_config, measurement and features with original EDA CSV.',
                 'Do not count skipped_ae_normal as RF mismatch. Never tune policy on these test files.']}
    summary['passed']=not changed and not counts.get('mismatch',0) and not counts.get('analysis_error',0) and not counts.get('rf_unavailable',0) and not summary['db_failures']
    write_csv(output/'integration_results.csv',results);write_csv(output/'offline_web_comparison.csv',comparisons)
    (output/'integration_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    (output/'PRESENTATION_CHECKLIST.md').write_text('''# 발표 전 확인\n\n- [ ] data_kind 및 실제 모델 경로/해시 확인 (합성 결과는 성능 검증 아님)\n- [ ] integration_summary.json의 오류/불일치/누락과 조건별 표본 확인\n- [ ] AE 정상 RF 생략과 AE 이상 RF 원시/최종 보류 구분\n- [ ] Normal1/2, IR/OR/Roller, Unbalance -1/-2, Looseness 포함 확인\n- [ ] 실제 운영 DB를 사용할 시 먼저 앱 종료 및 SQLite backup으로 외부 백업\n- [ ] 웹 업로드 후 파일/RPM 변경 시 과거 결과 초기화 확인\n- [ ] 분석 중 재클릭하지 말고 spinner 종료 후 조작\n- [ ] 실제 저장/앱 재시작 후 이력 재조회 (이 도구는 임시 DB만 검증)\n- [ ] Chrome 자동 번역을 사용하지 않음: 번역 아이콘 → 이 사이트 번역 안 함, 원문 표시\n- [ ] 1920×1080, 1366×768에서 스크롤·저장 버튼 확인\n- [ ] 오류 파일을 성공 시연으로 보고하지 않음\n''')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test-predictions',required=True);p.add_argument('--data-root',default=str(ROOT))
    p.add_argument('--ae-bundles',required=True);p.add_argument('--rf-bundle',required=True)
    p.add_argument('--output',required=True);p.add_argument('--data-kind',choices=['actual','synthetic'],required=True)
    args=p.parse_args(); summary=run(args);print(json.dumps(summary,ensure_ascii=False,indent=2));return 0 if summary['passed'] else 1

if __name__=='__main__':
    raise SystemExit(main())
