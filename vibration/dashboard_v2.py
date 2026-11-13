"""V2 presentation adapter over the unchanged inference and history APIs."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

from vibration.history import HistoryStore
from vibration.inference import SavedModelInference

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_MODELS = {"autoencoder", "ocsvm", "iforest"}


@dataclass(frozen=True)
class DashboardConfig:
    model: str
    bundle_root: Path


def configuration():
    model = os.environ.get("ISU_CHEMICAL_V2_MODEL", "autoencoder").strip()
    if model not in ALLOWED_MODELS:
        raise ValueError("대표 분석기 설정은 autoencoder, ocsvm, iforest 중 하나여야 합니다. DAE는 사용할 수 없습니다.")
    root = Path(os.environ.get("ISU_CHEMICAL_V2_BUNDLE_ROOT", str(
        ROOT / "outputs/inference_bundles/date_prelim_20261106"))).expanduser().resolve()
    return DashboardConfig(model, root)


def bundle_state(config, rpm):
    folder = config.bundle_root / f"{rpm}rpm"
    try:
        raw = (folder / "manifest.json").read_bytes()
        manifest = json.loads(raw)
        artifacts = manifest["artifacts"]
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError("번들 파일 목록 오류")
        stamps = []
        for name in artifacts:
            if Path(name).name != name:
                raise ValueError("번들 파일 경로 오류")
            stat = (folder / name).stat()
            stamps.append((name, stat.st_size, stat.st_mtime_ns))
        stamp = hashlib.sha256(raw).hexdigest(), tuple(sorted(stamps))
        return {"files_present": True, "stamp": stamp, "message": "분석 대기"}
    except Exception as error:
        return {"files_present": False, "stamp": None,
                "message": "분석기 준비 필요", "detail": f"{type(error).__name__}: {error}"}


def analyze(raw, rpm, filename, config):
    if config.model not in ALLOWED_MODELS:
        raise ValueError("이 대시보드에서 지원하지 않는 분석기입니다.")
    engine = SavedModelInference(config.bundle_root / f"{rpm}rpm", models=(config.model,))
    result = engine.predict_bytes(raw, rpm=rpm, filename=filename)
    versions = {name: "sha256:" + engine.manifest["artifacts"][engine.manifest["models"][name]["artifact"]]
                for name in engine.models}
    return result, versions


def history_store():
    return HistoryStore(os.environ.get("ISU_CHEMICAL_HISTORY_DB", str(ROOT / "outputs/history/analyses.sqlite3")))


def group_history(rows):
    """Old multi-model records remain one inspection; any alarm is explicit."""
    groups = {}
    for row in rows:
        item = groups.setdefault(row["analysis_id"], {
            "analysis_id": row["analysis_id"], "analyzed_at": row["analyzed_at"],
            "filename": row["filename"], "rpm": row["rpm"], "results": 0, "alarms": 0,
        })
        item["results"] += 1
        item["alarms"] += row["predicted_anomaly"]
    return list(groups.values())
