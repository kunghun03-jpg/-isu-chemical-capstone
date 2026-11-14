"""V3-only layout and artifact diagnostics; no model/policy changes."""
from pathlib import Path
from vibration.dae_readiness import probe as bundle_status

CSS='''<style>
.st-key-v2_results_row [data-testid="stColumn"]{min-width:0!important}
[class*="st-key-panel_"]{box-sizing:border-box;min-width:0;max-width:100%;overflow-wrap:anywhere}
.st-key-panel_topbar{background:#101f3c!important;color:white!important;border-radius:10px}
[data-testid="stSidebar"]{background:linear-gradient(160deg,#142849,#0b1930)!important}
@media(min-width:1051px) and (max-width:1500px){
.st-key-v2_results_row [data-testid="stHorizontalBlock"]:has(.st-key-panel_ae){flex-wrap:wrap!important}
.st-key-v2_results_row [data-testid="stHorizontalBlock"]:has(.st-key-panel_ae)>[data-testid="stColumn"]:nth-child(1){flex:1 1 37%!important;width:37%!important}
.st-key-v2_results_row [data-testid="stHorizontalBlock"]:has(.st-key-panel_ae)>[data-testid="stColumn"]:nth-child(2){flex:1 1 57%!important;width:57%!important}
.st-key-v2_results_row [data-testid="stHorizontalBlock"]:has(.st-key-panel_ae)>[data-testid="stColumn"]:nth-child(3){flex:1 1 100%!important;width:100%!important}
}
@media(max-width:1050px){
.st-key-panel_summary [data-testid="stHorizontalBlock"]{flex-wrap:wrap}
.st-key-panel_summary [data-testid="stColumn"]{flex:1 1 100%!important;width:100%!important}
}
</style>'''


def dae_diagnostics(rpm):
    state=bundle_status(rpm)
    folder=Path(state['folder']) if state.get('folder') else None
    files={name:bool(folder and (folder/name).is_file()) for name in ('manifest.json','autoencoder_model.pth','normalization.npz','threshold.npz')}
    return {**state,'configured_folder':str(folder) if folder else '','files_present':files,
        'explanation':'모델 미준비는 bundle manifest 또는 필수 파일을 찾지 못한 상태입니다. legacy NPZ/PTH만 있는 폴더는 추론 번들이 아닙니다.' if state['label']=='모델 미준비' else state.get('reason',state['label'])}
