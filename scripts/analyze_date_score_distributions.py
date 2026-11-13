"""기존 날짜 예비 실험을 읽기 전용으로 분석합니다. 학습·임계값 변경은 없습니다."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys


MODELS = ("autoencoder", "ocsvm", "iforest")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_saved(saved, computed, level):
    import numpy as np

    def key(row):
        result = (row["split"], row["file"])
        return result + (int(row["segment_id"]),) if level == "segment" else result

    expected = {key(r): r for r in computed}
    seen, maximum = set(), 0.0
    for row in saved:
        identity = key(row)
        if identity in seen or identity not in expected:
            raise ValueError("저장 점수 중복 또는 대상 불일치")
        seen.add(identity)
        score, target = float(row["score"]), expected[identity]["score"]
        np.testing.assert_allclose(score, target, rtol=1e-7, atol=1e-10)
        maximum = max(maximum, abs(score - target))
        if row["condition"] != expected[identity]["condition"]:
            raise ValueError("상태 불일치")
        if "threshold" in row and int(row["predicted_anomaly"]) != int(score > float(row["threshold"])):
            raise ValueError("저장 판정이 score > threshold와 다릅니다.")
    wanted = {k for k in expected if k[0] in {r["split"] for r in saved}}
    if seen != wanted:
        raise ValueError("저장 점수 파일·구간 누락")
    return maximum


def compute(root, output, rpm, run_name):
    import numpy as np
    import torch

    from sklearn.metrics import average_precision_score, roc_auc_score
    from vibration.comparison_metrics import file_scores, score_rows
    from vibration.experiment import check_fingerprint, load_experiment
    from vibration.model import Autoencoder
    from vibration.one_class import features, load_model

    folder = root / "outputs" / f"{rpm}rpm" / run_name
    datasets, fingerprint = load_experiment(folder, rpm)
    torch.set_num_threads(4)
    summaries, distributions, checks = [], [], []
    for name in MODELS:
        if name == "autoencoder":
            checkpoint = torch.load(folder / f"autoencoder_{rpm}rpm.pt", map_location="cpu", weights_only=True)
            check_fingerprint(checkpoint, fingerprint)
            model = Autoencoder()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            segment_threshold = json.loads((folder / f"threshold_{rpm}rpm.json").read_text(encoding="utf-8"))["threshold"]
            saved_segments = read_csv(folder / f"evaluation_{rpm}rpm.csv")
            saved_files = read_csv(folder / f"evaluation_file_{rpm}rpm.csv")
            file_thresholds = {float(r["threshold"]) for r in saved_files}
            if len(file_thresholds) != 1:
                raise ValueError("AE 파일 임계값이 일정하지 않습니다.")
            file_threshold = file_thresholds.pop()
        else:
            base = folder / "comparisons" / name
            record = load_model(base / "model.joblib", fingerprint=fingerprint)
            model = record["model"]
            thresholds = json.loads((base / "thresholds.json").read_text(encoding="utf-8"))
            np.testing.assert_allclose([thresholds["segment"], thresholds["file"]],
                                       [record["thresholds"]["segment"], record["thresholds"]["file"]])
            segment_threshold, file_threshold = thresholds["segment"], thresholds["file"]
            saved_segments = read_csv(base / "scores_segment.csv")
            saved_files = read_csv(base / "scores_file.csv")
        segments = []
        for split, data in datasets.items():
            if name == "autoencoder":
                parts = []
                # 고정 체크포인트 추론만 수행합니다. optimizer·backward·fit을 호출하지 않습니다.
                with torch.inference_mode():
                    for start in range(0, len(data["X"]), 32):
                        batch = torch.from_numpy(data["X"][start:start+32]).float()
                        reconstructed = model(batch)
                        errors = (reconstructed.double() - batch.double()).square().mean(dim=(1, 2))
                        parts.append(errors.numpy())
                scores = np.concatenate(parts)
                if (scores < 0).any():
                    raise ValueError("MSE에 음수 발생")
            else:
                # 구현 함수의 반환값을 믿는 대신 sklearn 원점수의 음수와 직접 대조합니다.
                scores = -np.asarray(model.score_samples(features(data["X"])), dtype=np.float64)
            segments.extend(score_rows(data, split, scores))
        files = file_scores(segments)
        segment_error = verify_saved(saved_segments, segments, "segment")
        file_error = verify_saved(saved_files, files, "file")
        q95_segment = float(np.percentile([r["score"] for r in segments if r["split"] == "validation"], 95))
        q95_file = float(np.percentile([r["score"] for r in files if r["split"] == "validation"], 95))
        np.testing.assert_allclose([segment_threshold, file_threshold], [q95_segment, q95_file], rtol=1e-7, atol=1e-10)
        for row in saved_segments:
            np.testing.assert_allclose(float(row["threshold"]), segment_threshold)
        for row in saved_files:
            np.testing.assert_allclose(float(row["threshold"]), file_threshold)
        # 그래프·오판정 집계에는 원래 저장된 점수를 우선 사용합니다. AE Train/Validation만 신규 추론값입니다.
        stored = {(r["split"], r["file"]): float(r["score"]) for r in saved_files}
        for row in files:
            row["score"] = stored.get((row["split"], row["file"]), row["score"])
            row.update(model=name, rpm=rpm, date_group=f"date:{row['date']}",
                       threshold=file_threshold, predicted_anomaly=int(row["score"] > file_threshold))
        write_csv(output / f"{rpm}_{name}_file_scores.csv", files)
        for split in ("train", "validation", "test_normal"):
            for condition in ("Normal1", "Normal2", "ALL"):
                selected = [r for r in files if r["split"] == split and (condition == "ALL" or r["condition"] == condition)]
                values = np.array([r["score"] for r in selected])
                distributions.append({"rpm":rpm, "model":name, "split":split, "condition":condition,
                    "count":len(values), "min":float(values.min()), "q25":float(np.percentile(values,25)),
                    "median":float(np.median(values)), "mean":float(values.mean()),
                    "q75":float(np.percentile(values,75)), "q95":float(np.percentile(values,95)),
                    "max":float(values.max()), "threshold":file_threshold,
                    "above_threshold":int((values > file_threshold).sum())})
        normal = [r for r in files if r["split"] == "test_normal"]
        anomaly = [r for r in files if r["split"] == "test_anomaly"]
        fp = sum(r["predicted_anomaly"] for r in normal)
        tp = sum(r["predicted_anomaly"] for r in anomaly)
        y = [0]*len(normal) + [1]*len(anomaly)
        test_scores = [r["score"] for r in normal + anomaly]
        summary = {"rpm":rpm, "model":name, "normal_files":len(normal), "anomaly_files":len(anomaly),
            "anomaly_prevalence":len(anomaly)/len(y), "false_positive":fp, "true_negative":len(normal)-fp,
            "false_negative":len(anomaly)-tp, "true_positive":tp, "misclassified":fp+len(anomaly)-tp,
            "false_positive_rate":fp/len(normal), "detection_rate":tp/len(anomaly),
            "auroc":float(roc_auc_score(y,test_scores)), "average_precision":float(average_precision_score(y,test_scores)),
            "normal1_fp":sum(r["predicted_anomaly"] for r in normal if r["condition"]=="Normal1"),
            "normal1_count":sum(r["condition"]=="Normal1" for r in normal),
            "normal2_fp":sum(r["predicted_anomaly"] for r in normal if r["condition"]=="Normal2"),
            "normal2_count":sum(r["condition"]=="Normal2" for r in normal)}
        original = json.loads((folder / "comparison_summary.json").read_text(encoding="utf-8"))["models"][name]["file"]
        for actual, expected in ((summary["false_positive_rate"],original["false_positive_rate"]),
                                 (summary["detection_rate"],original["detection_rate"]),
                                 (summary["auroc"],original["auroc"]),
                                 (summary["average_precision"],original["auprc_average_precision"])):
            np.testing.assert_allclose(actual, expected)
        summaries.append(summary)
        checks.append({"rpm":rpm,"model":name,"score_direction": "MSE; higher=anomaly" if name=="autoencoder" else "-score_samples; higher=anomaly",
            "aggregation":"arithmetic_mean_of_8_segment_scores","file_threshold":file_threshold,
            "validation_file_q95":q95_file,"segment_threshold":segment_threshold,"validation_segment_q95":q95_segment,
            "max_saved_segment_score_difference":segment_error,"max_saved_file_score_difference":file_error,
            "status":"passed"})
        print(f"{rpm} {name}: 점수 방향·8구간 평균·두 Q95·기존 결과 일치 확인", flush=True)
    for filename, data in (("confusion",summaries),("distributions",distributions),("verification",checks)):
        (output / f"{rpm}_{filename}.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")


def snapshot(paths):
    result = {}
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b""):
                digest.update(chunk)
        result[str(path)] = digest.hexdigest()
    return result


def make_report(root, output, run_name):
    summaries, distributions, checks = [], [], []
    for rpm in (900,1200,1800):
        for name, target in (("confusion",summaries),("distributions",distributions),("verification",checks)):
            target.extend(json.loads((output/f"{rpm}_{name}.json").read_text(encoding="utf-8")))
    write_csv(output/"confusion_counts.csv",summaries)
    write_csv(output/"normal_distribution_summary.csv",distributions)
    write_csv(output/"verification.csv",checks)
    text = ["# 날짜 예비 실험 정상 점수 분포 분석", "", "기존 실험 보존·재학습 없음. 저장 모델로 점수만 확인하고 기존 임계값을 그대로 사용했습니다.",
            "기존 CSV에 없는 AE Train/Validation 점수는 저장된 최적 체크포인트의 추론으로 보완했습니다.",
            "AE 시험 및 OCSVM·IF 점수는 재계산값과 일치 여부를 확인한 뒤 원래 저장된 점수를 분석에 사용했습니다.", "",
            "## 그래프", "", "각 그래프는 파일 점수 기준입니다. x축은 학습(9/12)·검증(9/16)·시험 정상(9/19), 색은 Normal1·Normal2입니다.",
            "박스는 중앙값·사분위 범위, 점은 개별 파일입니다. 빨간 점선은 두 정상 상태를 합친 기존 검증 파일 Q95입니다.",
            "모든 y축은 원래 점수이고 위쪽일수록 이상입니다. 음수인 OCSVM 점수도 부호를 다시 바꾸지 않았습니다.",
            "모델·RPM 사이 y축 범위는 독립이므로 점수의 크기를 직접 비교하지 않습니다.", "", "![9개 모델 전체 분포](normal_scores_overview.png)", "",
            "## 확인 결과", "", "- AE: double 정밀도 제곱 복원오차의 채널·주파수 평균(MSE), 비음수이며 클수록 이상.",
            "- OCSVM·IF: sklearn score_samples의 음수. 원점수·저장 점수와 대조했으며 클수록 이상.",
            "- 파일별 8개 구간(0~7)을 산술평균한 값이 원래 파일 CSV와 일치.",
            "- 세 모델 모두 구간은 검증 구간 Q95, 파일은 검증 파일 평균 점수 Q95와 일치. 검증 집합은 Normal1·Normal2 합산.",
            "- 판정은 score > threshold. 재학습·최적화·임계값 변경 없음.",
            "- 허용 오차 rtol=1e-7, atol=1e-10으로 저장 점수/Q95를 대조했으며 실제 차이는 verification.csv에 기록.", "",
            "## 시험 파일 오판정 개수", "", "FP: 정상→이상, FN: 결함→정상. Normal1·Normal2 열은 FP/해당 정상 파일 수입니다.", "",
            "| RPM | 모델 | FP | TN | FN | TP | 오판정 합계 | Normal1 FP/수 | Normal2 FP/수 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in summaries:
        text.append(f"| {r['rpm']} | {r['model']} | {r['false_positive']} | {r['true_negative']} | {r['false_negative']} | {r['true_positive']} | {r['misclassified']} | {r['normal1_fp']}/{r['normal1_count']} | {r['normal2_fp']}/{r['normal2_count']} |")
    text += ["", "## AP 해석에 필요한 이상 파일 비율", "", "| RPM | 시험 정상 | 시험 이상 | 시험 전체 | 이상 파일 비율 |", "| --- | ---: | ---: | ---: | ---: |"]
    for rpm in (900,1200,1800):
        r=next(r for r in summaries if r["rpm"]==rpm)
        text.append(f"| {rpm} | {r['normal_files']} | {r['anomaly_files']} | {r['normal_files']+r['anomaly_files']} | {r['anomaly_prevalence']:.2%} |")
    text += ["", "AUPRC는 Average Precision(AP)입니다. AP 0.95를 정확도 95%로 해석하지 않습니다.",
             "시험 집합은 약 79~80%가 이상 파일이므로 AP를 볼 때 양성 비율을 함께 제시합니다.",
             "무작위 순위 PR 기준선은 양성 비율 부근이며 유한 표본 AP의 기대값과 정확히 동일하다는 뜻은 아닙니다.",
             "전부 이상으로 판정하면 recall은 100%지만 precision은 이상 파일 비율이고 정상 오탐률은 100%입니다.",
             "AP는 여러 점수 경계에서의 순위 성능, 현재 FP/FN은 기존 Q95 한 지점의 판정입니다. 높은 AP와 높은 오탐률은 동시에 나타날 수 있습니다.", "",
             "## 관찰과 한계", "", "모든 RPM·모델에서 9/19 Normal2 시험 파일은 전부 이상으로 판정됐습니다.",
             "900RPM AE의 정상 오탐 51개는 모두 Normal2이며 Normal1 오탐은 0개입니다.",
             "점수 방향·파일 평균·Q95 적용에서 불일치는 발견하지 못했습니다. 정상 상태와 날짜별 점수 차이를 추가로 확인할 필요가 있습니다.",
             "다만 측정일·운전 조건·센서·정상 복구 여부 중 어느 것이 원인인지 이 분석만으로 확정할 수 없습니다.",
             "시험 결과를 보고 임계값을 올리거나 모델을 재학습하지 않았습니다. 독립 Run 확인과 별개의 날짜 그룹 예비 분석입니다.", "",
             "## 산출물", "", "- normal_scores_overview.png 및 figures/{RPM}_{model}.png: 전체·개별 그래프",
             "- *_file_scores.csv: 분석에 사용한 학습·검증·시험 파일 점수",
             "- normal_distribution_summary.csv: 정상 상태별/합산 분포 통계와 임계값 초과 수",
             "- confusion_counts.csv: FP/TN/FN/TP·AP·이상 파일 비율",
             "- verification.csv: 점수·집계·두 단위 Q95 대조 결과",
             "- preservation.json: 분석 전후 기존 실험 산출물 SHA256 비교", ""]
    (output/"REPORT.md").write_text("\n".join(text),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",default="date_prelim_20261106")
    parser.add_argument("--worker-rpm",type=int,choices=(900,1200,1800))
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if not args.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_name):
        parser.error("잘못된 실험 이름")
    root=Path(__file__).resolve().parents[1]
    output=args.output or root/"outputs"/f"{args.run_name}_score_analysis"
    output=output.resolve()
    experiment_folders=[root/"outputs"/f"{rpm}rpm"/args.run_name for rpm in (900,1200,1800)]
    if not output.is_relative_to(root/"outputs") or any(output.is_relative_to(p) for p in experiment_folders):
        raise ValueError("분석 결과는 outputs 내 기존 실험 폴더 밖의 새 경로에 저장해야 합니다.")
    if args.worker_rpm:
        compute(root,output,args.worker_rpm,args.run_name)
        return
    if output.exists():
        raise ValueError("기존 분석 폴더를 덮어쓰지 않습니다. --output으로 새 경로를 지정하세요.")
    sources=[p for rpm in (900,1200,1800) for p in (root/"outputs"/f"{rpm}rpm"/args.run_name).rglob('*') if p.is_file()]
    sources += [p for p in (root/"outputs"/f"{args.run_name}_reports").rglob('*') if p.is_file()]
    source_summary=root/"outputs"/f"{args.run_name}_summary.json"
    if source_summary.exists():sources.append(source_summary)
    before=snapshot(sources)
    output.mkdir(parents=True)
    environment={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONIOENCODING":"utf-8"}
    try:
        for rpm in (900,1200,1800):
            subprocess.run([sys.executable,"-B","-m","scripts.analyze_date_score_distributions","--run-name",args.run_name,
                            "--worker-rpm",str(rpm),"--output",str(output)],cwd=root,env=environment,check=True)
        subprocess.run([sys.executable,"-B","-m","scripts.plot_date_score_distributions","--output",str(output)],cwd=root,env=environment,check=True)
        make_report(root,output,args.run_name)
    finally:
        after=snapshot(sources)
        unchanged=before==after
        (output/"preservation.json").write_text(json.dumps({"files_checked":len(before),"unchanged":unchanged,
            "before_sha256":before,"after_sha256":after},ensure_ascii=False,indent=2),encoding="utf-8")
        if not unchanged:
            raise RuntimeError("기존 실험 파일이 변경되었습니다.")
    print(f"분석 완료. 기존 실험 {len(before)}개 파일 SHA256 보존 확인: {output}",flush=True)


if __name__=="__main__":
    main()
