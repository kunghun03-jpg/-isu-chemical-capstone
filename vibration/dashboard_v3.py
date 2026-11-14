"""Additional reference inference; AE operational result/policy are never mutated."""
from pathlib import Path
import hashlib
import json
import sqlite3
from contextlib import closing
import numpy as np
import pandas as pd
from vibration.inference import SavedModelInference
from vibration.dae_ui import predict_dae
from vibration.dae_readiness import probe as bundle_status
from scripts.analyze_fault_signatures import peak_component

MODELS={'autoencoder':'AE (기준)','dae':'DAE','ocsvm':'One-Class SVM','iforest':'Isolation Forest'}


def analyze_selected(raw,rpm,filename,config,model='autoencoder'):
    engine=SavedModelInference(config.bundle_root/f'{rpm}rpm',models=(model,),selected_artifacts_only=True)
    result=engine.predict_bytes(raw,rpm=rpm,filename=filename)
    versions={model:'sha256:'+engine.manifest['artifacts'][engine.manifest['models'][model]['artifact']]}
    return result,versions


def references(raw,rpm,config,ae):
    item=ae['models']['autoencoder']
    rows=[dict(model='autoencoder',label=MODELS['autoencoder'],status='완료',score=item['file_score'],threshold=item['file_threshold'],
               decision='이상' if item['predicted_anomaly'] else '정상',rule='저장 점수 > 저장 임계값',version=ae['bundle_id'])]
    for model in ('dae','ocsvm','iforest'):
        row=dict(model=model,label=MODELS[model],status='모델 미연결',score=None,threshold=None,decision='미연결',rule='개별 저장 기준',version=None)
        try:
            if model=='dae':
                state=bundle_status(rpm)
                if not state['ready']:
                    row.update(status='실행 오류' if state.get('code') in ('hash_error','manifest_error','model_load_error') or state['label']=='모델 검증 실패' else state['label'],reason=state.get('reason',state['label']))
                    rows.append(row);continue
                result=predict_dae(raw,rpm,state['id'])
                row.update(score=result['mse'],threshold=result['threshold'],decision=result['prediction'],version=state['id'],rule='파일 전체 Welch 복원 MSE > 저장 임계값')
            else:
                result,versions=analyze_selected(raw,rpm,'uploaded.mat',config,model)
                item=result['models'][model]
                row.update(score=item['file_score'],threshold=item['file_threshold'],decision='이상' if item['predicted_anomaly'] else '정상',version=versions[model],rule='-score_samples 8구간 평균 > 저장 임계값')
            if not np.isfinite([row['score'],row['threshold']]).all(): raise ValueError('비유한 모델 결과')
            row['status']='완료'
        except FileNotFoundError as error:
            row['reason']=str(error)
        except Exception as error:
            row.update(status='실행 오류',decision='실행 오류',reason=f'{type(error).__name__}: {error}')
        rows.append(row)
    return rows


def rotation_orders(details,rpm):
    if rpm not in (900,1200,1800): raise ValueError('unsupported RPM')
    values=[]
    for order in range(1,6):
        frequency,amplitude=peak_component(np.asarray(details['frequency']),np.asarray(details['amplitude']),order*rpm/60.,1.)
        values.append({'회전 성분':f'{order}X','명목 주파수 (Hz)':order*rpm/60.,'피크 주파수 (Hz)':float(frequency),'진폭 (raw)':float(amplitude)})
    return pd.DataFrame(values)


def save_references(store,result,versions,candidate,rows):
    identity,created=store.save(result,versions,candidate_report=candidate)
    payload=json.dumps(rows,ensure_ascii=False,sort_keys=True,allow_nan=False)
    checksum=hashlib.sha256(payload.encode()).hexdigest()
    with closing(sqlite3.connect(store.path)) as c,c:
        c.execute('PRAGMA foreign_keys=ON')
        c.execute('CREATE TABLE IF NOT EXISTS v3_reference_results (analysis_id INTEGER PRIMARY KEY REFERENCES analyses(id), payload_json TEXT NOT NULL, sha256 TEXT NOT NULL)')
        old=c.execute('SELECT sha256 FROM v3_reference_results WHERE analysis_id=?',(identity,)).fetchone()
        if old and old[0]!=checksum: raise ValueError('기존 참고 결과를 덮어쓰지 않습니다.')
        c.execute('INSERT OR IGNORE INTO v3_reference_results VALUES (?,?,?)',(identity,payload,checksum))
    return identity,created


def read_references(store,identity):
    if not store.path.is_file(): return None
    with closing(sqlite3.connect(store.path.resolve().as_uri()+'?mode=ro',uri=True)) as c:
        if not c.execute("SELECT 1 FROM sqlite_master WHERE name='v3_reference_results'").fetchone(): return None
        row=c.execute('SELECT payload_json,sha256 FROM v3_reference_results WHERE analysis_id=?',(int(identity),)).fetchone()
    if not row: return None
    if hashlib.sha256(row[0].encode()).hexdigest()!=row[1]: raise ValueError('참고 결과 해시 오류')
    return json.loads(row[0])
