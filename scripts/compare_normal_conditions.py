
from pathlib import Path
import csv

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs" / "normal_condition_comparison"
OUTPUT.mkdir(parents=True, exist_ok=True)

N_PER_GROUP = 30

# 탐색 분석용 임시 수치 안전 기준
# 물리적인 진동 이상 판정 임계값이 아님
MAX_ABS_SIGNAL = 1e100

METRICS = (
    "rms",
    "peak",
    "crest_factor",
    "1X",
    "2X",
    "3X",
)


def load_ch3(path):
    mat = loadmat(path)

    if "signals" not in mat or "fs" not in mat:
        raise ValueError(f"signals/fs 없음: {path}")

    signals = np.asarray(mat["signals"])
    fs = float(np.asarray(mat["fs"]).squeeze())

    if signals.ndim != 2 or signals.shape[0] < 3:
        raise ValueError(f"CH3 형상 오류: {path}")

    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"샘플링 주파수 오류: {path}")

    x = np.asarray(signals[2], dtype=np.float64)

    if x.size == 0:
        raise ValueError(f"빈 신호: {path}")

    if not np.isfinite(x).all():
        raise ValueError(f"NaN 또는 Inf 포함: {path}")

    # 평균을 빼기 전에 극단값을 검사해야
    # 평균 계산 과정의 overflow도 방지할 수 있음
    max_abs = float(np.max(np.abs(x)))

    if max_abs > MAX_ABS_SIGNAL:
        raise ValueError(
            f"극단값 신호 감지 "
            f"(최대 절댓값={max_abs:.3e}): {path.name}"
        )

    x = x - np.mean(x)

    if not np.isfinite(x).all():
        raise ValueError(f"평균 제거 후 비유한 신호: {path}")

    return x, fs


def analyze(path, rpm):
    x, fs = load_ch3(path)

    # 수치 오류를 조용히 무시하지 않고 예외로 처리
    with np.errstate(over="raise", invalid="raise"):
        rms = float(np.sqrt(np.mean(x ** 2)))
        peak = float(np.max(np.abs(x)))
        crest = peak / rms if rms > 0 else 0.0

        freq, psd = welch(
            x,
            fs=fs,
            window="hann",
            nperseg=min(8192, len(x)),
            noverlap=None,
            scaling="spectrum",
        )

    if not np.isfinite(psd).all():
        raise ValueError(f"PSD 계산 결과 비유한 값: {path.name}")

    rotational_frequency = rpm / 60.0
    harmonics = {}

    for order in (1, 2, 3):
        target = rotational_frequency * order

        mask = np.abs(freq - target) <= max(
            fs / 8192,
            2.0,
        )

        if not mask.any():
            raise ValueError(
                f"{order}X 주파수 구간 없음: {path.name}"
            )

        harmonics[f"{order}X"] = float(
            np.sqrt(np.max(psd[mask]))
        )

    return {
        "rms": rms,
        "peak": peak,
        "crest_factor": crest,
        **harmonics,
    }, freq, psd


def main():
    summary = []
    excluded = []

    for rpm in (900, 1200, 1800):
        print(f"\n========== {rpm} RPM ==========")

        for condition in ("Normal1", "Normal2"):
            files = sorted(
                DATA.glob(
                    f"*/**/*{rpm}RPM/{condition}/Vib/*.mat"
                )
            )

            if not files:
                print(f"{condition}: MAT 파일 없음")
                continue

            # 기존과 동일한 표본 선택 방식 유지
            indices = np.linspace(
                0,
                len(files) - 1,
                min(N_PER_GROUP, len(files)),
                dtype=int,
            )

            selected = [files[i] for i in indices]

            rows = []

            for path in selected:
                try:
                    values, _, _ = analyze(path, rpm)
                    rows.append(values)

                except (ValueError, FloatingPointError) as e:
                    print(f"[제외] {e}")

                    excluded.append({
                        "rpm": rpm,
                        "condition": condition,
                        "filename": path.name,
                        "reason": str(e),
                    })

            if not rows:
                print(
                    f"{condition}: 유효한 분석 파일 없음 "
                    f"(선택 {len(selected)}개)"
                )
                continue

            result = {
                "rpm": rpm,
                "condition": condition,
                "files": len(rows),
            }

            for metric in METRICS:
                values = np.array(
                    [r[metric] for r in rows],
                    dtype=np.float64,
                )

                result[f"{metric}_median"] = float(
                    np.median(values)
                )

            summary.append(result)

            print(
                f"{condition:8s} "
                f"N={len(rows):2d} "
                f"RMS={result['rms_median']:.6f} "
                f"1X={result['1X_median']:.6f} "
                f"2X={result['2X_median']:.6f} "
                f"3X={result['3X_median']:.6f}"
            )

    if not summary:
        raise RuntimeError(
            "분석 가능한 MAT 파일을 찾지 못했습니다."
        )

    # 정상 상태 비교 결과 저장
    summary_path = (
        OUTPUT / "normal1_vs_normal2_summary.csv"
    )

    with summary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary[0]),
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\n비교 결과 저장:", summary_path)

    # 제외 파일 목록 저장
    excluded_path = OUTPUT / "excluded_files.csv"

    with excluded_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rpm",
                "condition",
                "filename",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(excluded)

    print("제외 파일 목록 저장:", excluded_path)
    print(f"총 제외 파일: {len(excluded)}개")


if __name__ == "__main__":
    main()
