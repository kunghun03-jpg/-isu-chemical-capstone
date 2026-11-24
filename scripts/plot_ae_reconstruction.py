"""저장된 운영 AE로 입력/복원 Spectrum을 비교하는 읽기 전용 진단 도구."""

import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vibration.inference import SavedModelInference


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_ROOT = (
    ROOT / "outputs/inference_bundles/date_prelim_20261106"
)


def find_test_file(rpm, split, condition):
    """저장된 파일 단위 평가표에서 실제 평가 파일을 선택한다."""
    report = (
        ROOT / f"outputs/{rpm}rpm/date_prelim_20261106"
        / f"evaluation_file_{rpm}rpm.csv"
    )

    with report.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    candidates = [
        row for row in rows
        if row["split"] == split
        and row["condition"] == condition
    ]

    if not candidates:
        raise ValueError(
            f"평가 파일 없음: {rpm}RPM / {split} / {condition}"
        )

    # 평가표에 기록된 실제 파일 경로를 사용한다.
    for row in candidates:
        path = Path(row["file"])
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            return path, row

    raise FileNotFoundError(
        f"평가표의 파일 경로를 찾지 못했습니다: {candidates[0]['file']}"
    )


def analyze_file(engine, path, rpm):
    raw = path.read_bytes()

    # 실제 운영 추론과 동일한 전처리 및 정규화
    _, normalized = engine.preprocess(raw)

    model = engine._estimators["autoencoder"]

    with torch.inference_mode():
        x = torch.from_numpy(normalized).float()
        reconstructed = model(x).cpu().numpy()

    inputs = normalized[:, 0, :]
    outputs = reconstructed[:, 0, :]

    squared_error = (outputs.astype(np.float64)
                     - inputs.astype(np.float64)) ** 2

    # 실제 웹 추론 결과와 점수 일치 여부 확인
    result = engine.predict_bytes(raw, rpm=rpm, filename=path.name)
    ae = result["models"]["autoencoder"]

    calculated_score = float(
        squared_error.mean(axis=1).mean()
    )

    np.testing.assert_allclose(
        calculated_score,
        ae["file_score"],
        rtol=1e-6,
        atol=1e-9,
    )

    return inputs, outputs, squared_error, ae


def plot_case(axs, title, inputs, outputs, error, ae):
    # 8개 구간 중 복원오차가 가장 큰 구간을 대표 사례로 표시
    segment_id = int(np.argmax(error.mean(axis=1)))
    frequencies = np.arange(inputs.shape[1]) * 3.125

    axs[0].plot(
        frequencies, inputs[segment_id],
        label="Input Spectrum", linewidth=0.9
    )
    axs[0].plot(
        frequencies, outputs[segment_id],
        label="AE Reconstruction", linewidth=0.9,
        alpha=0.85
    )
    axs[0].set_ylabel("Train Z-score")
    axs[0].legend(fontsize=8)
    axs[0].set_title(
        f"{title} | segment {segment_id} (highest segment MSE)"
    )

    axs[1].plot(
        frequencies, error[segment_id],
        linewidth=0.8
    )
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Squared error")

    decision = "Abnormal" if ae["predicted_anomaly"] else "Normal"

    axs[1].set_title(
        f"File score={ae['file_score']:.5f} | "
        f"Q95 threshold={ae['file_threshold']:.5f} | "
        f"Decision={decision}",
        fontsize=10,
    )

    for ax in axs:
        ax.grid(alpha=0.2)
        ax.set_xlim(0, 10000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpm", type=int, default=900)
    parser.add_argument(
        "--bundle-root",
        default=os.environ.get(
            "ISU_CHEMICAL_V2_BUNDLE_ROOT",
            str(DEFAULT_BUNDLE_ROOT),
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/ae_reconstruction_visualization",
    )
    args = parser.parse_args()

    if args.rpm not in (900, 1200, 1800):
        raise ValueError("지원 RPM: 900, 1200, 1800")

    bundle_dir = Path(args.bundle_root) / f"{args.rpm}rpm"
    engine = SavedModelInference(
        bundle_dir,
        models=("autoencoder",),
        selected_artifacts_only=True,
    )

    cases = [
        ("Normal Test", "test_normal", "Normal1"),
        ("Bearing IR Test", "test_anomaly", "IR"),
    ]

    fig, axes = plt.subplots(
        4, 1, figsize=(13, 12), constrained_layout=True
    )

    for i, (label, split, condition) in enumerate(cases):
        path, archived = find_test_file(
            args.rpm, split, condition
        )
        inputs, outputs, error, ae = analyze_file(
            engine, path, args.rpm
        )

        # 선택한 MAT와 저장 평가 결과가 같은 점수인지 검증
        np.testing.assert_allclose(
            ae["file_score"],
            float(archived["score"]),
            rtol=1e-5,
            atol=1e-7,
        )

        plot_case(
            axes[2 * i:2 * i + 2],
            label, inputs, outputs, error, ae
        )

        print(
            f"{label}: {path.name} | "
            f"score={ae['file_score']:.6f} | "
            f"threshold={ae['file_threshold']:.6f} | "
            f"decision={'이상' if ae['predicted_anomaly'] else '정상'}"
        )

    fig.suptitle(
        f"{args.rpm}RPM | Operational AE Input vs Reconstruction\n"
        "Date-preliminary experiment; Run independence not verified",
        fontsize=14,
    )

    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    target = output / f"ae_input_vs_reconstruction_{args.rpm}rpm.png"
    fig.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\n그래프 저장 완료: {target}")


if __name__ == "__main__":
    main()
