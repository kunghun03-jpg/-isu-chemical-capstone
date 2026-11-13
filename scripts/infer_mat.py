"""MAT 파일 하나를 저장 모델로만 분석합니다. 모델 학습을 호출하지 않습니다."""

from pathlib import Path
import argparse
import json

from vibration.inference import SavedModelInference


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--rpm", type=int, choices=(900, 1200, 1800), required=True)
    parser.add_argument(
        "--model",
        choices=("all", "ae", "autoencoder", "ocsvm", "iforest"),
        default="all",
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=root / "outputs/inference_bundles/date_prelim_20261106",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="선택: outputs/single_file_inference 아래의 새 JSON 경로",
    )
    args = parser.parse_args()
    destination = args.output.resolve() if args.output else None
    if destination is not None and (
        destination.exists()
        or not destination.is_relative_to(root / "outputs/single_file_inference")
    ):
        parser.error(
            "결과는 outputs/single_file_inference 아래 새 파일에만 저장할 수 있습니다."
        )
    models = (
        ("autoencoder", "ocsvm", "iforest")
        if args.model == "all"
        else ("autoencoder" if args.model == "ae" else args.model,)
    )
    engine = SavedModelInference(args.bundle_root / f"{args.rpm}rpm", models=models)
    result = engine.predict_mat(args.file, rpm=args.rpm)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
