"""정상 CH3 원본·정규화 전 스펙트럼·고정 Train 표준편차를 읽기 전용 분석합니다."""

from pathlib import Path
from collections import defaultdict
from io import BytesIO
import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys

DATES = ("20260912", "20260916", "20260919")
CONDITIONS = ("Normal1", "Normal2")
BANDS = (("low", 0, 100), ("mid", 100, 1000), ("high", 1000, 10000))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compute(root, output, rpm, run_name):
    import numpy as np
    from scipy.io import loadmat
    from vibration.channel import select_ch3

    quality=[r for r in read_csv(root/"outputs/data_preparation/quality_all_files.csv")
             if r["rpm"]==str(rpm) and r["condition"] in CONDITIONS and r["date"] in DATES]
    rows=[r for r in quality if r["status"]=="pass"]
    write_csv(output/f"{rpm}_excluded.csv",[r for r in quality if r["status"]!="pass"] or
              [{"file":"", "reason":"no exclusions", "rpm":str(rpm)}])
    folder=root/"outputs"/f"{rpm}rpm"/run_name
    reference={}
    train_arrays=[]
    for split in ("train", "validation", "test_normal"):
        with np.load(folder/f"{split}_spectra.npz",allow_pickle=False) as data:
            x=data["X"].copy()
            if split=="train":train_arrays.append(x.astype(np.float64))
            for file in np.unique(data["source_file"]):
                mask=data["source_file"]==file
                ids=data["segment_id"][mask]
                reference[str(file)]=x[mask][np.argsort(ids),0,:]
    if set(reference)!={r["file"] for r in rows}:
        raise ValueError("품질 통과 정상 파일과 기존 실험 전처리 파일 목록이 다릅니다.")
    training=np.concatenate(train_arrays,axis=0)
    raw_std=training.std(axis=0,keepdims=True).reshape(-1)
    with np.load(folder/f"normalization_{rpm}rpm.npz") as data:
        mean=data["mean"].reshape(-1).copy()
        std=data["std"].reshape(-1).copy()
    np.testing.assert_allclose(raw_std.clip(1e-6),std)
    np.testing.assert_allclose(training.mean(axis=0).reshape(-1),mean)
    del training,train_arrays
    frequency=np.fft.rfftfreq(8192,1/25600.0)[:3201]
    window=np.hanning(8192)
    small=np.zeros(3201,dtype=bool)
    # 동률일 때 주파수 인덱스 순으로 정해 항상 동일한 321개를 선택합니다.
    small[np.argsort(std,kind="stable")[:math.ceil(0.1*len(std))]]=True
    band_masks={name:(frequency>=low)&((frequency<=high) if name=="high" else (frequency<high)) for name,low,high in BANDS}
    metrics,amplitudes,logs,waves=[],[],[],[]
    max_spectrum_difference=0.0
    for index,row in enumerate(rows,1):
        raw=(root/row["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row["sha256"]:
            raise ValueError(f"품질검사 이후 원본 변경: {row['file']}")
        data=loadmat(BytesIO(raw))
        if float(data["fs"].item())!=25600.0:
            raise ValueError("fs mismatch")
        signal=select_ch3(data["signals"])[0]
        if not np.isfinite(signal).all():raise ValueError("CH3 nonfinite")
        segments=signal.reshape(8,8192).copy()
        segments-=segments.mean(axis=1,keepdims=True)
        amplitude=np.abs(np.fft.rfft(segments*window,axis=1))/window.sum()
        amplitude[:,1:-1]*=2
        amplitude=amplitude[:,:3201]
        log_segments=np.log1p(amplitude).astype(np.float32)
        np.testing.assert_array_equal(log_segments,reference[row["file"]])
        max_spectrum_difference=max(max_spectrum_difference,float(np.max(np.abs(log_segments-reference[row["file"]]))))
        z=(log_segments.astype(np.float64)-mean)/std
        small_energy=float(np.square(z[:,small]).sum())
        total_energy=float(np.square(z).sum())
        item={key:row[key] for key in ("file","date","rpm","condition")}
        item.update(raw_mean=float(signal.mean()),rms=float(np.sqrt(np.mean(signal**2))),
                    ac_rms=float(np.sqrt(np.mean((signal-signal.mean())**2))),
                    peak_abs=float(np.abs(signal).max()),peak_to_peak=float(np.ptp(signal)),
                    normalized_input_msd=float(np.mean(z**2)),
                    small_std_input_squared_share=small_energy/total_energy if total_energy else 0.0)
        for name,mask in band_masks.items():
            item[f"{name}_band_rss"]=float(np.sqrt(np.sum(amplitude[:,mask]**2,axis=1)).mean())
            item[f"{name}_band_mean_amplitude"]=float(amplitude[:,mask].mean())
        metrics.append(item)
        amplitudes.append(amplitude.mean(axis=0))
        logs.append(log_segments.mean(axis=0,dtype=np.float64))
        waves.append(signal.astype(np.float64))
        if index%50==0:print(f"{rpm}RPM 원본 CH3 분석 {index}/{len(rows)}",flush=True)
    amplitudes=np.array(amplitudes)
    logs=np.array(logs)
    dates=np.array([r["date"] for r in rows]);conditions=np.array([r["condition"] for r in rows])
    np.savez_compressed(output/f"{rpm}_file_profiles.npz",frequency=frequency,amplitude=amplitudes,
                        log1p=logs,date=dates,condition=conditions,source_file=np.array([r["file"] for r in rows]),
                        train_std=std,unfloored_train_std=raw_std,small_std_mask=small,train_mean=mean)
    write_csv(output/f"{rpm}_file_metrics.csv",metrics)
    grouped_stats=[];shift_stats=[];std_stats=[];representatives={}
    fields=[k for k in metrics[0] if k not in {"file","date","rpm","condition"}]
    for condition in CONDITIONS:
        baseline=np.median(logs[(dates==DATES[0])&(conditions==condition)],axis=0)
        baseline_metrics={k:float(np.median([r[k] for r in metrics if r["date"]==DATES[0] and r["condition"]==condition])) for k in fields}
        for date in DATES:
            indices=np.flatnonzero((dates==date)&(conditions==condition))
            for field in fields:
                values=np.array([metrics[i][field] for i in indices])
                reference_median=baseline_metrics[field]
                grouped_stats.append({"rpm":rpm,"date":date,"condition":condition,"metric":field,"count":len(values),
                    "min":float(values.min()),"q10":float(np.percentile(values,10)),"q25":float(np.percentile(values,25)),
                    "median":float(np.median(values)),"q75":float(np.percentile(values,75)),"q90":float(np.percentile(values,90)),
                    "max":float(values.max()),"mean":float(values.mean()),
                    "median_ratio_to_sep12":float(np.median(values)/reference_median) if reference_median else None})
            rms=np.array([metrics[i]["rms"] for i in indices])
            representative=int(indices[np.argmin(np.abs(rms-np.median(rms)))])
            key=f"{date}_{condition}"
            representatives[key]={"file":rows[representative]["file"],"rms":metrics[representative]["rms"],
                                   "selection":"closest to group median raw RMS; visualization only"}
            delta=np.median(logs[indices],axis=0)-baseline
            scaled=delta/std
            raw_energy=float(np.sum(delta**2));scaled_energy=float(np.sum(scaled**2))
            before=float(np.sum(delta[small]**2)/raw_energy) if raw_energy else None
            after=float(np.sum(scaled[small]**2)/scaled_energy) if scaled_energy else None
            shift_stats.append({"rpm":rpm,"date":date,"condition":condition,"files":len(indices),
                "delta_log1p_rms":float(np.sqrt(np.mean(delta**2))),
                "delta_standardized_rms":float(np.sqrt(np.mean(scaled**2))),
                "small_std_bins":int(small.sum()),"small_std_raw_shift_share":before,
                "small_std_standardized_shift_share":after,
                "small_std_share_enrichment":after/before if before else None,
                "max_standardized_shift_frequency":float(frequency[np.argmax(np.abs(scaled))]),
                "max_abs_standardized_shift":float(np.max(np.abs(scaled)))})
    selected_waves={key:waves[next(i for i,r in enumerate(rows) if r["file"]==info["file"])] for key,info in representatives.items()}
    np.savez_compressed(output/f"{rpm}_representative_waves.npz",**selected_waves)
    (output/f"{rpm}_representatives.json").write_text(json.dumps(representatives,ensure_ascii=False,indent=2),encoding="utf-8")
    for j,f in enumerate(frequency):
        std_stats.append({"rpm":rpm,"frequency_hz":float(f),"train_std_raw":float(raw_std[j]),
                          "train_std_used":float(std[j]),"gain_1_over_std":float(1/std[j]),
                          "std_floor_active":int(raw_std[j]<1e-6),"smallest_10pct_rank":int(small[j])})
    write_csv(output/f"{rpm}_group_statistics.csv",grouped_stats)
    write_csv(output/f"{rpm}_normalization_shift.csv",shift_stats)
    write_csv(output/f"{rpm}_frequency_std.csv",std_stats)
    audit={"rpm":rpm,"raw_normal_files":len(quality),"quality_passed":len(rows),"excluded":len(quality)-len(rows),
           "matching_original_pre_z_files":len(reference),"max_pre_z_spectrum_difference":max_spectrum_difference,
           "std_min":float(std.min()),"std_median":float(np.median(std)),"std_max":float(std.max()),
           "small_std_cutoff":float(std[small].max()),"std_floor_bins":int((raw_std<1e-6).sum()),
           "small_std_gain_median":float(np.median(1/std[small])),
           "other_std_gain_median":float(np.median(1/std[~small]))}
    (output/f"{rpm}_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"{rpm}RPM: 전 파일 통계·원본 파형·Train std 대조 완료",flush=True)


def report(root,output):
    stats=[];shifts=[];audits=[]
    for rpm in (900,1200,1800):
        stats+=read_csv(output/f"{rpm}_group_statistics.csv")
        shifts+=read_csv(output/f"{rpm}_normalization_shift.csv")
        audits.append(json.loads((output/f"{rpm}_audit.json").read_text(encoding="utf-8")))
    write_csv(output/"group_statistics.csv",stats)
    write_csv(output/"normalization_shift.csv",shifts)
    def median(rpm,date,condition,metric):
        return float(next(r["median"] for r in stats if r["rpm"]==str(rpm) and r["date"]==date and r["condition"]==condition and r["metric"]==metric))
    lines=["# 날짜별 정상 CH3 원본 및 정규화 확대 분석", "", "기존 모델 재학습·임계값 변경 없이 원본 CH3와 기존 9/12 Train 정규화 기준을 분석했습니다.",
        "원본 Normal1·Normal2 920개를 목록으로 점검하고, 기존 품질 정책에 따라 12개를 제외한 908개 전체를 통계에 사용했습니다. Normal3는 범위에서 제외했습니다.",
        "제외 파일은 각 RPM excluded.csv에 기록했습니다. 일부 CH3만 유효하더라도 기존 전체 6채널 품질 정책을 유지했습니다.","",
        "## 방법과 그림", "", "- 원본 파형: CH3의 65,536샘플(2.56초), 입력 수치의 단위를 그대로 사용합니다. 물리 단위·교정은 확인되지 않았습니다.",
        "- 파일 RMS·최대 절댓값·평균(DC)·평균 제거 RMS를 계산했습니다. 모든 분포 점은 개별 파일입니다.",
        "- 스펙트럼: 기존과 동일한 8×8,192구간, 구간 평균 제거, Hann, 단측 FFT 진폭 보정, 0~10kHz. 파일은 8구간 평균으로 요약합니다.",
        "- 선형 진폭과 log1p 후/Z-score 전 스펙트럼을 모두 저장했습니다. 기존 pre-Z NPZ와 파일별 8구간 스펙트럼이 정확히 일치함을 확인했습니다.",
        "- 대역: [0,100), [100,1000), [1000,10000]Hz. 대역 크기는 구간별 sqrt(sum(진폭²))의 8구간 평균(RSS)이며 PSD 적분이나 시간 RMS가 아닙니다. 폭이 다른 대역끼리 절대 크기를 비교하지 않습니다.",
        "- 스펙트럼 선은 전체 파일의 주파수별 중앙값, 음영은 10~90백분위입니다. 파형 예시는 각 날짜·상태에서 RMS 중앙값에 가장 가까운 파일이며 원인 결론의 근거로 단독 사용하지 않습니다.", "",
        "![전체 파일 원본 RMS·최대값·DC 분포](figures/raw_metric_distributions.png)",
        "![전체 파일 대역별 RSS 분포](figures/band_distributions.png)", "",
        "각 RPM의 원본 파형·전체/저주파 진폭·pre-Z log1p 비교:"]
    for rpm in (900,1200,1800):lines.append(f"- [{rpm}RPM 파형 및 스펙트럼](figures/{rpm}_waveform_spectra.png)")
    lines += ["", "## 전체 파일 RMS 중앙값", "", "| RPM | 상태 | 9/12 | 9/16 | 9/19 | 9/19 ÷ 9/12 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for rpm in (900,1200,1800):
        for condition in CONDITIONS:
            values=[median(rpm,d,condition,"rms") for d in DATES]
            lines.append(f"| {rpm} | {condition} | {values[0]:.6g} | {values[1]:.6g} | {values[2]:.6g} | {values[2]/values[0]:.3f} |")
    lines += ["", "## 작은 Train 표준편차의 확대 효과", "", "기존 정규화 기준은 변경하지 않았습니다. 주파수별 실제 사용 σ=max(Train std,1e-6)를 사용합니다.",
        "각 상태의 9/12 파일 평균 log1p 스펙트럼 중앙값을 기준으로 날짜별 차이 Δ를 구하고, 동일 차이가 정규화 후 Δ/σ가 되는지 분석합니다.",
        "σ가 작은 하위 10% 순위 321개 bin을 고정 선택했습니다. 표의 기여도는 이 bin이 전체 주파수의 제곱차 합에서 차지하는 비율이며 물리적 진동 에너지가 아닙니다.",
        "정규화 전/후 기여도가 커지는지로 작은 σ가 특정 날짜 차이를 상대적으로 강조하는지 확인합니다. 단순히 1/σ가 크다는 사실만으로 주원인이라고 결론 내리지 않습니다.", "",
        "![Train std 및 주파수별 날짜 차이](figures/normalization_diagnostics.png)",
        "![작은 std bin의 제곱차 기여도](figures/small_std_contributions.png)", "",
        "| RPM | 상태 | 비교 날짜 | Δ log1p RMS | Δ/σ RMS | 작은 σ bin 기여: 전 | 후 |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for r in shifts:
        if r["date"]==DATES[0]:continue
        before=float(r["small_std_raw_shift_share"]);after=float(r["small_std_standardized_shift_share"])
        lines.append(f"| {r['rpm']} | {r['condition']} | {r['date'][4:6]}/{r['date'][6:]} | {float(r['delta_log1p_rms']):.6g} | {float(r['delta_standardized_rms']):.3f} | {before:.2%} | {after:.2%} |")
    lines += ["", "| RPM | 사용 σ 최소 | 중앙값 | 하한 적용 bin | 하위10% σ 상한 |", "| --- | ---: | ---: | ---: | ---: |"]
    for a in audits:lines.append(f"| {a['rpm']} | {a['std_min']:.6g} | {a['std_median']:.6g} | {a['std_floor_bins']} | {a['small_std_cutoff']:.6g} |")
    lines += ["", "## 관찰 사실과 원인 추정의 구분", "", "위 표와 그래프는 전체 파일의 관찰 통계입니다. RMS·대역 크기·주파수별 차이는 센서 교정이나 실제 운전 조건이 같은지를 증명하지 않습니다.",
        "Δ/σ와 normalized_input_msd는 입력의 표준화 편차이며 AE 복원오차가 아닙니다. 이 분석만으로 작은 σ가 이전 오탐의 원인 또는 유일한 원인이라고 확정할 수 없습니다.",
        "센서 위치·감도·단위·측정 이득, 장착/재조립·베어링 교체, 실제 RPM과 정상 복구 여부가 확인되지 않았습니다. 이들은 원인 후보일 뿐 현재 확인된 사실이 아닙니다.",
        "파형은 각 파일의 시작 시각을 0으로 둔 것으로 회전 위상을 동기화하지 않았습니다. 날짜별 위상 차이를 결함 증거로 해석하지 않습니다.",
        "실제 Run 경계는 여전히 미확정입니다. 분석 결과를 보고 기존 전처리 기준·임계값을 바꾸거나 모델을 재학습하지 않았습니다.", "",
        "## 산출물", "", "file_metrics/group_statistics CSV: 전체 파일과 날짜·RPM·상태별 분포. file_profiles NPZ: 파일 평균 진폭·log1p 스펙트럼.",
        "normalization_shift/frequency_std CSV: 작은 σ와 날짜 차이의 정량 분석. representatives JSON: 예시 파일 선택 근거.",
        "preservation.json: 원본 정상 MAT와 기존 outputs 산출물의 분석 전후 SHA256 확인. 실제 원본이나 기존 실험 파일에는 쓰지 않았습니다.", ""]
    (output/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name",default="date_prelim_20261106")
    parser.add_argument("--worker-rpm",type=int,choices=(900,1200,1800))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--resume",action="store_true",help="검사된 원본이 그대로인 중단 분석만 재개")
    args=parser.parse_args()
    if not args.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_name):parser.error("잘못된 실험 이름")
    root=Path(__file__).resolve().parents[1]
    output=(args.output or root/"outputs"/f"{args.run_name}_ch3_signal_analysis").resolve()
    if not output.is_relative_to(root/"outputs") or any(output.is_relative_to(root/"outputs"/f"{rpm}rpm") for rpm in (900,1200,1800)):
        raise ValueError("기존 RPM 실험 폴더 밖의 별도 outputs 분석 폴더만 허용합니다.")
    if args.worker_rpm:
        compute(root,output,args.worker_rpm,args.run_name);return
    if output.exists() and not args.resume:raise ValueError("기존 분석 결과를 덮어쓰지 않습니다.")
    from scripts.analyze_date_score_distributions import snapshot
    quality=read_csv(root/"outputs/data_preparation/quality_all_files.csv")
    targets=[r for r in quality if r["condition"] in CONDITIONS and r["date"] in DATES]
    sources=[root/r["file"] for r in targets]
    sources += [p for p in (root/"outputs").rglob('*') if p.is_file() and not p.is_relative_to(output)]
    before=snapshot(sources)
    if args.resume:
        previous=json.loads((output/"preservation.json").read_text(encoding="utf-8"))
        if not previous["unchanged"] or previous["before_sha256"]!=before:
            raise ValueError("중단 후 원본/기존 실험이 달라져 재개할 수 없습니다.")
    for row in targets:
        if before[str(root/row["file"])]!=row["sha256"]:raise ValueError("정상 원본이 기존 품질검사 이후 변경되었습니다.")
    output.mkdir(parents=True,exist_ok=args.resume)
    env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONIOENCODING":"utf-8"}
    try:
        for rpm in (900,1200,1800):
            subprocess.run([sys.executable,"-B","-m","scripts.analyze_normal_ch3_dates","--run-name",args.run_name,
                            "--worker-rpm",str(rpm),"--output",str(output)],cwd=root,env=env,check=True)
        subprocess.run([sys.executable,"-B","-m","scripts.plot_normal_ch3_dates","--output",str(output)],cwd=root,env=env,check=True)
        report(root,output)
    finally:
        after=snapshot(sources)
        (output/"preservation.json").write_text(json.dumps({"raw_normal_files":len(targets),"existing_output_files":len(sources)-len(targets),
            "unchanged":before==after,"before_sha256":before,"after_sha256":after},ensure_ascii=False,indent=2),encoding="utf-8")
        if before!=after:raise RuntimeError("원본 또는 기존 실험 파일의 변경 발견")
    print(f"분석 완료: {len(targets)}개 원본 정상 MAT와 기존 결과 SHA256 보존: {output}",flush=True)


if __name__=="__main__":main()
