"""세 모델이 공유하는 Run 분할과 정규화 데이터의 일치 여부를 확인합니다."""

from collections import defaultdict
from pathlib import Path
import hashlib
import json

import numpy as np

from vibration.channel import validate_spectra
from vibration.dataset import read_rows, validate_assignments, validate_date_groups

PARTITIONS = ("train", "validation", "test_normal", "test_anomaly")


def experiment_metadata(output_dir, rpm):
    config = json.loads(
        (Path(output_dir) / f"split_config_{rpm}rpm.json").read_text(encoding="utf-8")
    )
    return {
        "experiment_mode": config["mode"],
        "group_type": config.get("group_type", "run"),
        "run_independence_verified": False,
        "normalization_mode": json.loads((Path(output_dir) / f"normalization_config_{rpm}rpm.json").read_text(encoding="utf-8"))["mode"]
            if (Path(output_dir) / f"normalization_config_{rpm}rpm.json").exists() else "per_frequency",
        "development_comparison": config.get("development_comparison", False),
        "test_previously_examined": config.get("test_previously_examined", False),
    }


def load_experiment(output_dir, rpm):
    """Run 또는 명시적 날짜 예비 실험을 검사합니다. 일반 미확정·파일 모드는 차단합니다."""
    output_dir = Path(output_dir)
    config_path = output_dir / f"split_config_{rpm}rpm.json"
    if not config_path.is_file():
        raise ValueError(
            "확정 Run 분할이 없습니다. Run 정보를 확인하고 split부터 실행하세요."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mode = config.get("mode")
    if mode not in {"run", "date_preliminary"}:
        raise ValueError(
            "Run 또는 명시적 date_preliminary 분할만 허용합니다. file 모드는 사용할 수 없습니다."
        )
    rows = read_rows(output_dir / f"split_{rpm}rpm.csv")
    if mode == "date_preliminary":
        if config.get("group_type") != "date_not_actual_run":
            raise ValueError("날짜 예비 실험 표기가 필요합니다.")
        validate_date_groups(rows)
    else:
        validate_assignments(rows, require_run=True)
    if not rows or any(row["rpm"] != str(rpm) or not row["date"] for row in rows):
        raise ValueError("분할 목록의 RPM·날짜를 확인하세요.")
    digest = hashlib.sha256(
        json.dumps(sorted(rows, key=lambda r: r["file"]), sort_keys=True).encode()
    )
    digest.update(str(mode).encode())
    datasets = {}
    for split in PARTITIONS:
        expected = {row["file"]: row for row in rows if row["split"] == split}
        if not expected:
            raise ValueError(f"공정한 비교에 필요한 {split} 파일이 없습니다.")
        path = output_dir / f"{split}_normalized.npz"
        with np.load(path, allow_pickle=False) as data:
            required = {
                "X",
                "source_file",
                "condition",
                "segment_id",
                "date",
                "rpm",
                "run_id",
            }
            if mode == "date_preliminary":
                required.add("date_group")
            if not required.issubset(data.files):
                raise ValueError(
                    f"{split}: Run 메타데이터가 없습니다. 전처리를 다시 실행하세요."
                )
            values = {key: data[key].copy() for key in sorted(required)}
        validate_spectra(values["X"])
        n = len(values["X"])
        if any(values[key].shape != (n,) for key in required - {"X"}):
            raise ValueError(f"{split}: 메타데이터 길이 또는 형태가 잘못되었습니다.")
        if values["segment_id"].dtype.kind not in "iu":
            raise ValueError(f"{split}: 구간 번호는 정수여야 합니다.")
        groups = defaultdict(list)
        for index, name in enumerate(values["source_file"].astype(str)):
            if name not in expected:
                raise ValueError(f"{split}: 분할 목록에 없는 파일: {name}")
            row = expected[name]
            for key in ("condition", "date", "rpm", "run_id") + (
                ("date_group",) if mode == "date_preliminary" else ()
            ):
                if str(values[key][index]) != row[key]:
                    raise ValueError(f"{split}: {name}의 {key}가 분할 목록과 다릅니다.")
            groups[name].append(int(values["segment_id"][index]))
        if set(groups) != set(expected) or any(
            len(ids) != 8 or set(ids) != set(range(8)) for ids in groups.values()
        ):
            raise ValueError(
                f"{split}: 파일 누락 또는 파일별 8개 구간 구성이 잘못되었습니다."
            )
        # 신호 값까지 식별하므로 재정규화된 데이터에 오래된 모델을 섞을 수 없습니다.
        for key, value in sorted(values.items()):
            digest.update(key.encode())
            digest.update(str((value.shape, value.dtype.str)).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        datasets[split] = values
    return datasets, digest.hexdigest()


def check_fingerprint(record, fingerprint):
    if record.get("data_fingerprint") != fingerprint:
        raise ValueError(
            "모델·결과와 현재 분할/전처리 데이터가 다릅니다. 같은 실험의 결과를 사용하세요."
        )


def save_context(output_dir, rpm, level, fingerprint):
    path = Path(output_dir) / f"evaluation_{level}_context_{rpm}rpm.json"
    path.write_text(
        json.dumps(
            {"data_fingerprint": fingerprint, **experiment_metadata(output_dir, rpm)},
            indent=2,
        ),
        encoding="utf-8",
    )
