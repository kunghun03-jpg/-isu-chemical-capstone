"""Check exact web runtime versions and load all three real inference bundles."""

import argparse
import json
from pathlib import Path
import sys

PINNED = {"numpy": "2.4.6", "torch": "2.14.1+cpu", "scikit_learn": "1.9.0",
          "scipy": "1.18.1", "streamlit": "1.67.0"}
ROOT = Path(__file__).resolve().parents[1]


def installed_versions():
    import numpy
    import scipy
    import sklearn
    import streamlit
    import torch

    return {"numpy": numpy.__version__, "torch": str(torch.__version__),
            "scikit_learn": sklearn.__version__, "scipy": scipy.__version__,
            "streamlit": streamlit.__version__}


def verify(bundle_root):
    report = {"python": sys.version.split()[0], "executable": sys.executable,
              "required_versions": PINNED.copy(), "bundles": [], "ready": False,
              "actual_mat_inference_validated": False}
    try:
        report["installed_versions"] = installed_versions()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        return report
    if report["installed_versions"] != PINNED:
        report["error"] = "웹 전용 환경의 고정 패키지 버전이 일치하지 않습니다."
        return report
    from vibration.inference import SavedModelInference

    for rpm in (900, 1200, 1800):
        folder = Path(bundle_root).expanduser().resolve() / f"{rpm}rpm"
        item = {"rpm": rpm, "folder": str(folder), "loaded": False}
        try:
            engine = SavedModelInference(folder)
            if engine.rpm != rpm:
                raise ValueError("번들 폴더와 manifest의 RPM이 다릅니다.")
            item.update(loaded=True, models=list(engine.models),
                        bundle_id=engine.manifest["bundle_id"])
            del engine
        except Exception as error:
            item["error"] = f"{type(error).__name__}: {error}"
        report["bundles"].append(item)
    report["ready"] = all(item["loaded"] for item in report["bundles"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path,
                        default=ROOT / "outputs/inference_bundles/date_prelim_20261106")
    args = parser.parse_args()
    report = verify(args.bundle_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
