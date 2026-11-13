# 진동 신호를 8개 구간으로 나누고 FFT 스펙트럼을 생성합니다.

from pathlib import Path
import csv
import json
import numpy as np
from scipy.io import loadmat
from vibration.preprocessing import ch3_log_spectra, frequency_bins
from vibration.dataset import (
    validate_assignments,
    validate_file_metadata,
    path_metadata,
    validate_date_groups,
)


def run(project_dir, output_dir, rpm):
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / f"split_{rpm}rpm.csv"

    # 1. 25,600Hz 신호를 8,192개 샘플(0.32초) 단위로 처리합니다.
    freq = frequency_bins()
    with split_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    # 전처리를 직접 실행해도 같은 Run이 여러 분할에 섞이지 않도록 다시 검사합니다.
    config_path = output_dir / f"split_config_{rpm}rpm.json"
    date_mode = (
        config_path.exists()
        and json.loads(config_path.read_text(encoding="utf-8")).get("mode")
        == "date_preliminary"
    )
    if date_mode:
        validate_date_groups(rows)
    else:
        validate_assignments(rows, require_run=any(row.get("run_id") for row in rows))
    for row in rows:
        validate_file_metadata(project_dir, row, rpm)
    for split in ("train", "validation", "test_normal", "test_anomaly"):
        selected = [row for row in rows if row["split"] == split]
        spectra_list = []
        source_files = []
        conditions = []
        segment_ids = []
        metadata = {key: [] for key in ("date", "rpm", "run_id")}
        if date_mode:
            metadata["date_group"] = []
        for row in selected:
            path = project_dir / row["file"]
            data = loadmat(path)
            signals = data["signals"]
            # 단일 파일 추론과 같은 함수로 전처리합니다. 기존 계산식은 유지합니다.
            X_file = ch3_log_spectra(signals, data["fs"])
            spectra_list.append(X_file)
            for segment_id in range(8):
                source_files.append(row["file"])
                conditions.append(row["condition"])
                segment_ids.append(segment_id)
                metadata["date"].append(
                    row.get("date") or path_metadata(row["file"])["date"]
                )
                metadata["rpm"].append(str(rpm))
                metadata["run_id"].append(row.get("run_id", ""))
                if date_mode:
                    metadata["date_group"].append(row["date_group"])
        if not spectra_list:
            raise ValueError(f"{split}: 사용할 데이터가 없습니다.")
        X = np.concatenate(spectra_list, axis=0)

        # 4. 스펙트럼과 원본 파일·상태·구간 번호를 함께 저장합니다.
        np.savez_compressed(
            output_dir / f"{split}_spectra.npz",
            X=X,
            frequency=freq,
            source_file=np.array(source_files),
            condition=np.array(conditions),
            segment_id=np.array(segment_ids),
            **{key: np.array(values, dtype=str) for key, values in metadata.items()},
        )
        print(f"{split}: {X.shape}, 사용 파일 {len(selected)}개", flush=True)
        del X, spectra_list
    print(f"\n{rpm}RPM 스펙트럼 저장 완료!")
    print("저장 위치:", output_dir)
