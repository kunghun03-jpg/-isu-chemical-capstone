"""새 MAT 데이터의 읽기 전용 일괄 추론. 학습 및 임계값 변경 없음."""

import argparse
import csv
import re
from pathlib import Path

from vibration.inference import SavedModelInference


RPMS = (900, 1200, 1800)
BUNDLE_ROOT = Path(
    "outputs/inference_bundles/date_prelim_20261106"
)

FIELDS = [
    "file", "rpm", "model", "score",
    "threshold", "predicted_anomaly", "decision",
    "bundle_id", "error",
]


def detect_rpm(path, input_root, default_rpm):
    found = set()

    for part in path.relative_to(input_root).parts[:-1]:
        match = re.fullmatch(
            r"(?:\d{8}_)?(900|1200|1800)RPM",
            part, re.IGNORECASE
        )
        if match:
            found.add(int(match.group(1)))

    if len(found) > 1:
        raise ValueError("폴더의 RPM 정보가 서로 다릅니다.")

    if found:
        rpm = found.pop()
        if default_rpm is not None and rpm != default_rpm:
            raise ValueError("폴더 RPM과 지정 RPM이 다릅니다.")
        return rpm

    if default_rpm is not None:
        return default_rpm

    raise ValueError("RPM 확인 불가: --rpm을 지정하세요.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rpm", type=int, choices=RPMS)
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_root.is_dir():
        raise ValueError("입력 폴더가 존재하지 않습니다.")

    if output_path.exists():
        raise FileExistsError("기존 결과 파일은 덮어쓰지 않습니다.")

    files = sorted(input_root.rglob("*.mat"))

    if not files:
        raise ValueError("입력 폴더에 MAT 파일이 없습니다.")

    if output_path.is_relative_to(input_root):
        raise ValueError("결과 파일은 입력 폴더 밖에 저장하세요.")

    print("MAT 파일:", len(files))

    models = {}
    results = []
    success = 0
    failed = 0

    for i, path in enumerate(files, 1):
        relative = path.relative_to(input_root).as_posix()
        rpm = ""

        try:
            rpm = detect_rpm(path, input_root, args.rpm)

            if rpm not in models:
                bundle = BUNDLE_ROOT / f"{rpm}rpm"
                models[rpm] = SavedModelInference(
                    bundle,
                    selected_artifacts_only=True
                )

            prediction = models[rpm].predict_mat(
                path, rpm=rpm
            )

            for name, info in prediction["models"].items():
                results.append({
                    "file": relative,
                    "rpm": rpm,
                    "model": name,
                    "score": info["file_score"],
                    "threshold": info["file_threshold"],
                    "predicted_anomaly": info["predicted_anomaly"],
                    "decision": info["decision"],
                    "bundle_id": prediction["bundle_id"],
                    "error": "",
                })

            success += 1

        except Exception as error:
            failed += 1
            results.append({
                "file": relative,
                "rpm": rpm,
                "model": "",
                "score": "",
                "threshold": "",
                "predicted_anomaly": "",
                "decision": "",
                "bundle_id": "",
                "error": f"{type(error).__name__}: {error}",
            })

        if i % 20 == 0 or i == len(files):
            print(f"진행: {i}/{len(files)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print("\n===== 일괄 분석 결과 =====")
    print("전체 파일:", len(files))
    print("분석 성공:", success)
    print("분석 실패:", failed)
    print("결과 CSV:", output_path)
    print("운영 모델 및 임계값 변경 없음")


if __name__ == "__main__":
    main()
