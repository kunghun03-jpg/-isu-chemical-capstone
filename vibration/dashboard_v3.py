"""Additional reference inference; AE operational result/policy are never mutated."""
from pathlib import Path
import hashlib
import json
import sqlite3
from contextlib import closing
import numpy as np
import pandas as pd
from vibration.inference import SavedModelInference
from vibration.dae_experimental_bridge import predict_experimental_dae
from vibration.dae_experimental_history import model_version
from scripts.analyze_fault_signatures import peak_component

MODELS={'autoencoder':'AE (기준)','dae':'DAE (실험용)','ocsvm':'One-Class SVM','iforest':'Isolation Forest'}


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
                result = predict_experimental_dae(raw, rpm)
                row.update(
                    label='DAE (실험용)',
                    score=result['score'],
                    threshold=result['threshold'],
                    decision=result['prediction'],
                    version=model_version(),
                    rule='Welch 복원 MSE > Normal Validation Q95',
                    evaluation_scope='date_based_exploratory',
                    validated_for_operation=False,
                    reason='날짜 기준 탐색적 분석 · 운영 검증 미완료',
                )
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


def save_references(store,result,versions,candidate,rows,*,raw=None):
    # V3 참고 결과의 직렬화 오류를 DB 저장 전에 확인
    payload=json.dumps(rows,ensure_ascii=False,sort_keys=True,allow_nan=False)
    checksum=hashlib.sha256(payload.encode()).hexdigest()

    # DAE가 성공적으로 실행된 경우에만 실험용 이력 연결
    dae_rows=[r for r in rows if r.get('model')=='dae']
    if len(dae_rows)>1:
        raise ValueError('DAE 결과가 중복되어 있습니다.')

    dae_experimental=None
    if dae_rows and dae_rows[0].get('status')=='완료':
        row=dae_rows[0]
        if type(raw) is not bytes or not raw:
            raise ValueError('DAE 저장에 원본 MAT bytes가 필요합니다.')

        rpm=int(result['input']['rpm'])
        if hashlib.sha256(raw).hexdigest()!=result['input']['sha256']:
            raise ValueError('업로드 파일과 분석 결과 SHA256이 다릅니다.')
        if row.get('version')!=model_version():
            raise ValueError('DAE 모델 버전이 분석 시점과 다릅니다.')
        if row.get('validated_for_operation') is not False:
            raise ValueError('DAE 실험용 검증 상태가 올바르지 않습니다.')

        score=float(row['score'])
        threshold=float(row['threshold'])
        predicted=int(score>threshold)

        if not np.isfinite([score,threshold]).all():
            raise ValueError('DAE 점수 또는 임계값 오류')
        if row['decision']!=('이상' if predicted else '정상'):
            raise ValueError('DAE 점수와 판정이 일치하지 않습니다.')

        dae_result={
            'model':'DAE',
            'rpm':rpm,
            'score':score,
            'threshold':threshold,
            'predicted_anomaly':predicted,
            'prediction':row['decision'],
            'evaluation_scope':'date_based_exploratory',
            'validated_for_operation':False,
        }
        dae_experimental=(raw,rpm,dae_result)

    # AE·RF·DAE는 HistoryStore의 동일 트랜잭션에서 저장
    identity,created=store.save(
        result,
        versions,
        candidate_report=candidate,
        dae_experimental=dae_experimental,
    )
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
