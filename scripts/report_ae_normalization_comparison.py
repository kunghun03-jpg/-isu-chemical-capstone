"""주파수별/Train 공통 정규화 AE의 고정된 두 결과만 비교합니다. 설정 탐색 없음."""

from pathlib import Path
from collections import Counter
import argparse
import csv
import json
import os
import subprocess
import sys


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        for key in ("score","threshold"):row[key]=float(row[key])
        row["predicted_anomaly"]=int(row["predicted_anomaly"])
        if "segment_id" in row:row["segment_id"]=int(row["segment_id"])
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",default="date_prelim_20261106")
    parser.add_argument("--run-name",default="ae_globalnorm_dev_20261106")
    args=parser.parse_args()
    for name in (args.baseline,args.run_name):
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):parser.error("잘못된 실험 이름")
    import numpy as np
    import torch
    from vibration.comparison_metrics import metrics,write_csv
    from vibration.experiment import load_experiment,check_fingerprint
    from vibration.model import Autoencoder
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1]
    output=root/"outputs"/f"{args.run_name}_reports"
    output.mkdir(exist_ok=True)
    summaries=[];conditions=[];audits=[]
    for rpm in (900,1200,1800):
        baseline=root/"outputs"/f"{rpm}rpm"/args.baseline
        changed=root/"outputs"/f"{rpm}rpm"/args.run_name
        assert (baseline/f"split_{rpm}rpm.csv").read_bytes()==(changed/f"split_{rpm}rpm.csv").read_bytes()
        data,fingerprint=load_experiment(changed,rpm)
        with np.load(changed/"train_spectra.npz") as pre:train=pre["X"].astype(np.float64)
        with np.load(changed/f"normalization_{rpm}rpm.npz") as reference:
            mean,std=reference["mean"].copy(),reference["std"].copy()
        assert mean.shape==std.shape==(1,1,1)
        np.testing.assert_allclose(mean,train.mean())
        np.testing.assert_allclose(std,max(train.std(),1e-6))
        del train
        for split in ("train","validation","test_normal","test_anomaly"):
            assert (baseline/f"{split}_spectra.npz").read_bytes()==(changed/f"{split}_spectra.npz").read_bytes()
            with np.load(changed/f"{split}_spectra.npz") as pre:
                np.testing.assert_array_equal(data[split]["X"],((pre["X"].astype(np.float64)-mean)/std).astype(np.float32))
        checkpoint=torch.load(changed/f"autoencoder_{rpm}rpm.pt",map_location="cpu",weights_only=True)
        check_fingerprint(checkpoint,fingerprint)
        assert checkpoint["normalization_mode"]=="global_train" and checkpoint["development_comparison"] and checkpoint["test_previously_examined"]
        model=Autoencoder();model.load_state_dict(checkpoint["model_state_dict"]);model.eval()
        validation=data["validation"]
        score_parts=[]
        with torch.inference_mode():
            for start in range(0,len(validation["X"]),32):
                batch=torch.from_numpy(validation["X"][start:start+32]).float()
                score_parts.append((model(batch).double()-batch.double()).square().mean(dim=(1,2)).numpy())
        val_scores=np.concatenate(score_parts)
        val_files=[float(val_scores[validation["source_file"]==file].mean()) for file in np.unique(validation["source_file"])]
        q95_segment=float(np.percentile(val_scores,95));q95_file=float(np.percentile(val_files,95))
        segment_threshold=json.loads((changed/f"threshold_{rpm}rpm.json").read_text(encoding="utf-8"))["threshold"]
        np.testing.assert_allclose(segment_threshold,q95_segment)
        preserved=json.loads((changed/"baseline_preservation.json").read_text(encoding="utf-8"))
        assert preserved["unchanged"]
        state={}
        for variant,folder in (("per_frequency",baseline),("global_train",changed)):
            for level,filename in (("segment",f"evaluation_{rpm}rpm.csv"),("file",f"evaluation_file_{rpm}rpm.csv")):
                rows=read_csv(folder/filename)
                ids={(r["split"],r["file"],r.get("segment_id")) for r in rows}
                assert len(ids)==len(rows)
                if variant=="per_frequency":state[level]=ids
                else:assert ids==state[level]
                assert all(r["predicted_anomaly"]==int(r["score"]>r["threshold"]) for r in rows)
                threshold={r["threshold"] for r in rows};assert len(threshold)==1
                threshold=threshold.pop()
                if variant=="global_train":np.testing.assert_allclose(threshold,q95_segment if level=="segment" else q95_file)
                m=metrics(rows)
                normal=[r for r in rows if r["split"]=="test_normal"]
                faulty=[r for r in rows if r["split"]=="test_anomaly"]
                fp=sum(r["predicted_anomaly"] for r in normal);tp=sum(r["predicted_anomaly"] for r in faulty)
                summaries.append({"rpm":rpm,"normalization":variant,"level":level,"threshold":threshold,
                    "normal_count":len(normal),"anomaly_count":len(faulty),"false_positive":fp,"false_negative":len(faulty)-tp,
                    "fpr":m["false_positive_rate"],"detection_rate":m["detection_rate"],"auroc":m["auroc"],
                    "ap":m["auprc_average_precision"],"anomaly_prevalence":len(faulty)/(len(normal)+len(faulty))})
                for condition in sorted({r["condition"] for r in rows}):
                    selected=[r for r in rows if r["condition"]==condition]
                    conditions.append({"rpm":rpm,"normalization":variant,"level":level,"condition":condition,
                        "count":len(selected),"alarm_count":sum(r["predicted_anomaly"] for r in selected),
                        "rate":sum(r["predicted_anomaly"] for r in selected)/len(selected),
                        "metric":"false_positive_rate" if condition in {"Normal1","Normal2"} else "detection_rate"})
        audits.append({"rpm":rpm,"mean":float(mean.item()),"std":float(std.item()),"best_epoch":checkpoint["epoch"],
            "validation_segment_q95":q95_segment,"validation_file_q95":q95_file,"train_only_stats_verified":True,
            "identical_pre_z_and_split":True,"all_new_normalized_values_verified":True,"baseline_preserved":True})
        print(f"{rpm}RPM: Train 공통 통계·동일 입력·새 Validation Q95·개발용 표기 확인",flush=True)
    write_csv(output/"metrics_comparison.csv",summaries)
    write_csv(output/"condition_comparison.csv",conditions)
    write_csv(output/"normalization_audit.csv",audits)
    (output/"comparison.json").write_text(json.dumps({"development_comparison":True,"test_previously_examined":True,
        "additional_search":False,"metrics":summaries,"condition_rates":conditions,"audit":audits},ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# AE 정규화 방식 개발용 비교", "", "이미 점수·원본을 분석한 9/19 시험 데이터를 재사용하는 **개발용 비교**입니다. 독립적인 최종 성능 검증이나 정규화 방식의 최종 선정 결과가 아닙니다.",
        "AE만 각 RPM에서 한 번씩 추가 학습했습니다. OCSVM·IF·DAE는 재실행하지 않았고 결과를 보고 추가 설정 탐색을 하지 않았습니다.","",
        "## 변경한 요소와 유지한 조건", "", "- 기존: 9/12 Train의 주파수별 평균·표준편차 `(1,1,3201)`.",
        "- 비교: 해당 RPM의 9/12 CH3 Train log1p 스펙트럼 모든 값에서 계산한 공통 평균·표준편차 `(1,1,1)`, ddof=0.",
        "- 9/12 학습·9/16 검증·9/19 시험, 품질 제외·Normal3 제외·파일 목록을 그대로 유지했습니다.",
        "- FFT·log1p NPZ와 split CSV를 바이트 동일하게 복사했고 원본 MAT는 다시 처리하지 않았습니다.",
        "- AE 구조·seed 42·batch 32·Adam 0.001·최대30 epoch·patience8·검증 기준 모델 선택을 유지했습니다.",
        "- 파일 점수는 8개 구간 MSE 평균입니다. 변경 모델은 새 검증 구간/파일 점수에서 각각 Q95를 계산했습니다.",
        "- σ 하한 1e-6도 유지했습니다. 실제 Run ID는 미확정이며 날짜 그룹 표기를 유지했습니다.","",
        "## 파일 단위 전체 지표", "", "| RPM | 정규화 | 정상 오탐률 | 이상 탐지율 | AUROC | AP | FP | FN |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in summaries:
        if r["level"]=="file":lines.append(f"| {r['rpm']} | {r['normalization']} | {r['fpr']:.2%} | {r['detection_rate']:.2%} | {r['auroc']:.4f} | {r['ap']:.4f} | {r['false_positive']} | {r['false_negative']} |")
    lines += ["", "## 상태별 파일 판정 비율", "", "각 셀은 기존 주파수별 → 공통 정규화입니다. Normal1·Normal2는 **오탐률**, 나머지 결함은 **탐지율**입니다.",
        "| 상태 | 900RPM | 1200RPM | 1800RPM |", "| --- | ---: | ---: | ---: |"]
    for condition in sorted({r["condition"] for r in conditions}):
        values=[]
        for rpm in (900,1200,1800):
            pair=[next(r for r in conditions if r["rpm"]==rpm and r["normalization"]==v and r["level"]=="file" and r["condition"]==condition) for v in ("per_frequency","global_train")]
            values.append(f"{pair[0]['rate']:.2%} → {pair[1]['rate']:.2%}")
        lines.append(f"| {condition} | "+" | ".join(values)+" |")
    lines += ["", "## 새 Train 공통 기준과 검증 임계값", "", "임계값·MSE의 수치 크기는 정규화 단위가 달라 직접 비교하지 않습니다.",
        "| RPM | 공통 평균 | 공통 σ | 최적 epoch | 구간 Q95 | 파일 Q95 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in audits:lines.append(f"| {r['rpm']} | {r['mean']:.8g} | {r['std']:.8g} | {r['best_epoch']} | {r['validation_segment_q95']:.8g} | {r['validation_file_q95']:.8g} |")
    lines += ["", "## 해석의 한계와 저장 위치", "", "정규화만 바꾼 비교이지만 시험 데이터를 이미 살펴본 뒤 설계했으므로 최종 일반화 성능을 주장하지 않습니다.",
        "AP는 Average Precision이며 정확도가 아닙니다. 동일 시험 데이터의 이상 파일 비율은 RPM별 약79~80%로 유지됩니다.",
        "오탐률·전체/상태별 탐지율·AUROC·AP를 함께 보고, 특정 지표 하나의 개선만으로 모델을 선정하지 않습니다.",
        f"새 실험: `outputs/{{RPM}}rpm/{args.run_name}/`. 기존 실험 `{args.baseline}`은 보존했습니다.",
        f"전체 파일/구간 지표·상태별 분자/분모: `outputs/{args.run_name}_reports/`.",
        "각 RPM의 comparison_design.json·normalization_config·새 모델·임계값·학습/평가 로그와 baseline_preservation.json에서 재현 조건을 확인할 수 있습니다.",""]
    subprocess.run([sys.executable,"-B","-m","scripts.plot_ae_normalization_comparison","--output",str(output)],
                   cwd=root,env={**os.environ,"PYTHONIOENCODING":"utf-8","PYTHONDONTWRITEBYTECODE":"1"},check=True)
    lines.extend(["",f"![두 정규화 방식 파일 지표](../outputs/{args.run_name}_reports/metrics_comparison.png)",""])
    (root/"docs/AE_NORMALIZATION_COMPARISON.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps([r for r in summaries if r["level"]=="file"],ensure_ascii=False,indent=2))


if __name__=="__main__":main()
