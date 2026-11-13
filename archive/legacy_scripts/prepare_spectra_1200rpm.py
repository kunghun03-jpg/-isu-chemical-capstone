# 보관된 초기 실험 코드: prepare_spectra_1200rpm. 현재 공통 실행 진입점은 run_project.py입니다.

from pathlib import Path
import csv
import numpy as np
from scipy.io import loadmat

project_dir = Path(__file__).resolve().parents[2]

# 1200RPM 결과 전용 폴더
output_dir = project_dir / "outputs_clean" / "1200rpm" / "first"
output_dir.mkdir(parents=True, exist_ok=True)

# 분할 목록은 outputs 바로 아래에 있음
split_path = project_dir / "outputs_clean" / "1200rpm" / "first" / "split_1200rpm.csv"

WINDOW_SIZE = 8192
EXPECTED_FS = 25600.0
MAX_FREQ = 10000

window = np.hanning(WINDOW_SIZE)
freq = np.fft.rfftfreq(WINDOW_SIZE, d=1 / EXPECTED_FS)
freq_mask = freq <= MAX_FREQ

with split_path.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

for split in ("train", "validation", "test_normal", "test_anomaly"):
    selected = [row for row in rows if row["split"] == split]

    spectra_list = []
    source_files = []
    conditions = []
    segment_ids = []

    for row in selected:
        path = project_dir / row["file"]

        # 다른 RPM 데이터가 섞이지 않도록 확인
        if "20260916_1200RPM" not in Path(row["file"]).parts:
            raise ValueError(f"1200RPM이 아닌 경로: {row['file']}")

        data = loadmat(path)
        signals = data["signals"]
        fs = float(data["fs"].item())

        if signals.shape != (6, 65536) or fs != EXPECTED_FS:
            raise ValueError(f"데이터 형식 확인 필요: {row['file']}")

        if not np.isfinite(signals).all() or (np.abs(signals) > 1e100).any():
            raise ValueError(f"분할 목록에 손상 의심 파일이 있습니다: {row['file']}")

        # 파일 하나를 8개 구간으로 분할
        segments = signals.reshape(6, 8, WINDOW_SIZE).transpose(1, 0, 2)

        segments = segments - segments.mean(axis=2, keepdims=True)

        amplitude = np.abs(np.fft.rfft(segments * window, axis=2)) / window.sum()

        amplitude[:, :, 1:-1] *= 2
        X_file = np.log1p(amplitude[:, :, freq_mask]).astype(np.float32)

        if not np.isfinite(X_file).all():
            raise ValueError(f"변환 결과 확인 필요: {row['file']}")

        spectra_list.append(X_file)

        for segment_id in range(8):
            source_files.append(row["file"])
            conditions.append(row["condition"])
            segment_ids.append(segment_id)

    if not spectra_list:
        raise ValueError(f"{split}: 사용할 데이터가 없습니다.")

    X = np.concatenate(spectra_list, axis=0)

    np.savez_compressed(
        output_dir / f"{split}_spectra.npz",
        X=X,
        frequency=freq[freq_mask],
        source_file=np.array(source_files),
        condition=np.array(conditions),
        segment_id=np.array(segment_ids),
    )

    print(f"{split}: {X.shape}, " f"사용 파일 {len(selected)}개", flush=True)

    del X, spectra_list

print("\n1200RPM 스펙트럼 저장 완료!")
print("저장 위치:", output_dir)
