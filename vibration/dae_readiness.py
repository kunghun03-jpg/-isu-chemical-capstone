"""Strict DAE readiness audit for V3; no RPM bypass or model mutation."""
import os
from pathlib import Path
from vibration.dae_bundle import load_manifest,check_rpm
from vibration.dae_inference import SavedDAEBundleInference
from vibration.dae_ui import DEFAULT_BUNDLE


def probe(rpm,folder=None):
    configured=str(folder) if folder is not None else os.environ.get('ISU_CHEMICAL_DAE_BUNDLE',str(DEFAULT_BUNDLE))
    state={'ready':False,'id':None,'folder':str(Path(configured).expanduser().resolve()) if configured.strip() else '',
           'label':'모델 미준비','code':'path_missing'}
    if not configured.strip(): return dict(state,reason='DAE 번들 경로가 비어 있습니다.')
    path=Path(state['folder'])
    if not path.is_dir(): return dict(state,reason='DAE 번들 폴더를 찾을 수 없습니다.')
    try: manifest=load_manifest(path)
    except FileNotFoundError as error: return dict(state,code='file_missing',reason=str(error))
    except Exception as error:
        code='hash_error' if any(k in str(error) for k in ('해시','SHA256')) else 'manifest_error'
        return dict(state,code=code,label='모델 검증 오류',reason=f'{type(error).__name__}: {error}')
    state.update(id=manifest['bundle_id'],model_version=manifest['model_version'],training_provenance=manifest['training_provenance'],supported_rpms=manifest['supported_rpms'])
    try: SavedDAEBundleInference(path)
    except Exception as error: return dict(state,code='model_load_error',label='모델 로드 오류',reason=f'{type(error).__name__}: {error}')
    if manifest['supported_rpms'] is None:
        return dict(state,code='rpm_unconfirmed',label='지원 범위 미확인',reason='계산 호환성은 RPM별 탐지 성능 또는 지원 범위를 증명하지 않습니다.')
    try: check_rpm(manifest,rpm)
    except Exception as error: return dict(state,code='rpm_unsupported',label='선택 RPM 미지원',reason=str(error))
    return dict(state,code='ready',label='사용 가능',ready=True)
