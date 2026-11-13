"""완료된 날짜 예비 실험의 지표를 읽어 CSV/Markdown으로 정리합니다. 재학습하지 않습니다."""

from pathlib import Path
import argparse
import csv
import json

from vibration.dataset import read_rows, validate_date_groups
from vibration.comparison_metrics import write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="date_prelim_20261106")
    args = parser.parse_args()
    if not args.run_name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in args.run_name
    ):
        parser.error("잘못된 실험 이름")
    root = Path(__file__).resolve().parents[1]
    summaries, counts, conditions, level_rows = {}, [], [], {"file": [], "segment": []}
    for rpm in (900, 1200, 1800):
        folder = root / "outputs" / f"{rpm}rpm" / args.run_name
        rows = read_rows(folder / f"split_{rpm}rpm.csv")
        validate_date_groups(rows)
        n = {
            s: sum(r["split"] == s for r in rows)
            for s in ("train", "validation", "test_normal", "test_anomaly")
        }
        counts.append({"rpm": rpm, **n})
        summary = json.loads(
            (folder / "comparison_summary.json").read_text(encoding="utf-8")
        )
        assert (
            summary["experiment_mode"] == "date_preliminary"
            and summary["run_independence_verified"] is False
        )
        timings = json.loads(
            (folder / "stage_timings.json").read_text(encoding="utf-8")
        )
        assert len(timings) == 8 and all(r["exit_code"] == 0 for r in timings)
        summaries[str(rpm)] = summary
        for model, levels in summary["models"].items():
            for level, m in levels.items():
                factor = 8 if level == "segment" else 1
                assert (
                    m["normal_count"] == n["test_normal"] * factor
                    and m["anomaly_count"] == n["test_anomaly"] * factor
                )
                level_rows[level].append(
                    {
                        "rpm": rpm,
                        "model": model,
                        **{k: v for k, v in m.items() if k != "condition_alarm_rates"},
                    }
                )
                for condition, rate in m["condition_alarm_rates"].items():
                    conditions.append(
                        {
                            "rpm": rpm,
                            "model": model,
                            "level": level,
                            "condition": condition,
                            "alarm_rate": rate,
                        }
                    )
    report_dir = root / "outputs" / f"{args.run_name}_reports"
    report_dir.mkdir(exist_ok=True)
    for level, rows in level_rows.items():
        write_csv(report_dir / f"comparison_{level}.csv", rows)
    write_csv(report_dir / "condition_alarm_rates.csv", conditions)
    write_csv(report_dir / "split_counts.csv", counts)
    text = [
        "# 날짜 그룹 예비 실험 결과",
        "",
        "실험일: 2026-11-06. 사용자 지정 날짜 분할로 AE·OCSVM·IF를 RPM별로 각각 학습·평가했습니다.",
        "**실제 Run 경계를 확인한 독립 Run 실험이 아닙니다.** run_id는 빈칸이며 date_group만 기록했습니다. DAE는 실행하지 않았습니다.",
        "",
        "## 분할과 사전 검사",
        "",
        "- 2026-09-12 Normal1·Normal2: 학습",
        "- 2026-09-16 Normal1·Normal2: 검증",
        "- 2026-09-19 Normal1·Normal2: 정상 시험, 같은 날짜 결함 상태: 이상 시험",
        "- Normal3와 품질 문제 파일 제외. 9/12·9/16의 결함 데이터도 사용하지 않음.",
        "- 2,091개 사용 파일의 품질검사 SHA256 일치, 경로 중복 0, 내용 중복 0을 확인한 뒤 전처리·학습 시작.",
        "- 전체 실제 Run 미확정 목록은 그대로 유지. 분할·모델·평가 결과에 date_preliminary/date_not_actual_run 표기.",
        "",
        "| RPM | 학습 정상 | 검증 정상 | 시험 정상 | 시험 결함 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in counts:
        text.append(
            f"| {row['rpm']} | {row['train']} | {row['validation']} | {row['test_normal']} | {row['test_anomaly']} |"
        )
    text += [
        "",
        "## 파일 단위 비교",
        "",
        "파일 점수는 8구간 점수의 평균이며 임계값은 9/16 정상 검증 파일 점수의 Q95입니다.",
        "AUPRC는 Average Precision(AP)입니다. 결과를 보고 설정·임계값을 조정하지 않았습니다.",
        "",
        "| RPM | 모델 | 정상 오탐률 | 결함 탐지율 | AUROC | AUPRC(AP) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in level_rows["file"]:
        text.append(
            f"| {r['rpm']} | {r['model']} | {r['false_positive_rate']:.2%} | {r['detection_rate']:.2%} | {r['auroc']:.4f} | {r['auprc_average_precision']:.4f} |"
        )
    text += [
        "",
        "## 해석 범위와 설정",
        "",
        "- 날짜가 바뀐 정상 데이터에 대한 오탐률을 반드시 함께 봐야 합니다. 높은 탐지율이나 AP만으로 실사용 적합성을 주장하지 않습니다.",
        "- 날짜 간 분포 차이 가능성은 있지만, 현재 지표만으로 원인을 단정하지 않습니다. 실제 Run 정보와 센서·운전 이력 확인이 필요합니다.",
        "- 기본 전처리 CH3/8구간/Hann FFT/log1p/Train 기준 Z-score와 기존 모델 설정을 유지했습니다.",
        "- AE: seed 42, batch 32, Adam 0.001, 최대 30 epoch, 검증 기준 최적 저장·patience 8.",
        "- OCSVM: RBF, nu=0.05, gamma=scale. IF: 300 trees, random_state=42, n_jobs=1. 모두 정상 Train만 학습.",
        "- 구간·파일 임계값은 각각 검증 Q95. 모델별 점수의 숫자 크기를 직접 비교하지 않습니다.",
        "- stage_timings.json은 Python 프로세스 시작·파일 I/O 등을 포함하는 단계별 벽시계 시간이며 순수 모델 추론시간 비교가 아닙니다.",
        "",
        "## 저장 위치와 재현",
        "",
        f"각 RPM 결과: `outputs/{{RPM}}rpm/{args.run_name}/`.",
        f"통합 CSV: `outputs/{args.run_name}_reports/` (파일·구간 비교, 상태별 판정 비율, 분할 수).",
        "각 폴더에 preflight.json, split CSV/config, NPZ, AE 체크포인트·학습 이력, 모델별 결과, 실행 로그와 단계별 시간이 있습니다.",
        "새 실험 폴더만 허용하며 같은 이름으로 전체 재실행하면 중단합니다. 재현 시 패키지 버전과 고정 설정을 확인하세요.",
        "",
        "```powershell",
        f"python -m scripts.run_date_preliminary --run-name {args.run_name}_new",
        "```",
        "",
        "기본 Run 모드의 미확정 ID 차단은 유지합니다. 날짜 예비 실험은 명시적 별도 모드이며 실제 Run 확인을 대체하지 않습니다.",
    ]
    (root / "docs" / "DATE_PRELIMINARY_RESULTS.md").write_text(
        "\n".join(text) + "\n", encoding="utf-8"
    )
    print(json.dumps(level_rows["file"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
