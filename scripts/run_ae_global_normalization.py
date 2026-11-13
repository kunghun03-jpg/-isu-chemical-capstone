"""AE 정규화만 바꾸는 단일 개발용 비교. 이미 살펴본 시험 데이터를 재사용합니다."""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

from scripts.analyze_date_score_distributions import snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",default="date_prelim_20261106")
    parser.add_argument("--run-name",default="ae_globalnorm_dev_20261106")
    args=parser.parse_args()
    for value in (args.baseline,args.run_name):
        if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):parser.error("잘못된 실험 이름")
    root=Path(__file__).resolve().parents[1]
    folders=[root/"outputs"/f"{rpm}rpm"/args.run_name for rpm in (900,1200,1800)]
    if any(p.exists() for p in folders):raise ValueError("새 실험 폴더만 허용합니다.")
    source_files=[p for p in (root/"outputs").rglob('*') if p.is_file()]
    before=snapshot(source_files)
    copies=[]
    try:
        for rpm,folder in zip((900,1200,1800),folders):
            baseline=root/"outputs"/f"{rpm}rpm"/args.baseline
            config=json.loads((baseline/f"split_config_{rpm}rpm.json").read_text(encoding="utf-8"))
            if config["mode"]!="date_preliminary":raise ValueError("기준 날짜 예비 실험이 아닙니다.")
            folder.mkdir(parents=True)
            names=[f"{s}_spectra.npz" for s in ("train","validation","test_normal","test_anomaly")]
            names += [f"split_{rpm}rpm.csv",f"quality_check_{rpm}rpm.csv",f"quality_checked_{rpm}rpm.csv","preflight.json"]
            for name in names:
                source=baseline/name;target=folder/name
                shutil.copyfile(source,target)
                digest=hashlib.sha256(target.read_bytes()).hexdigest()
                if digest!=before[str(source)]:raise ValueError("분할/전처리 복사본 불일치")
                copies.append({"rpm":rpm,"file":name,"sha256":digest,"matches_baseline":True})
            config.update(development_comparison=True,test_previously_examined=True,
                          normalization_mode="global_train",baseline_run=args.baseline,
                          only_changed_factor="normalization",models=["autoencoder"],
                          no_additional_parameter_search=True)
            (folder/f"split_config_{rpm}rpm.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
            (folder/"comparison_design.json").write_text(json.dumps({
                "development_comparison":True,"test_previously_examined":True,"only_changed":"normalization",
                "baseline_normalization":"per_frequency","comparison_normalization":"global_train",
                "training_reference":"same RPM; 20260912 Normal1/Normal2 only; all segment/channel/frequency cells",
                "baseline_run":args.baseline,"seed":42,"batch_size":32,"optimizer":"Adam",
                "learning_rate":0.001,"max_epochs":30,"patience":8,"std_floor":1e-6,
                "file_aggregation":"arithmetic mean of 8 segment MSE scores",
                "threshold":"recomputed validation Q95 separately for segment and file scores",
                "source_code_sha256":snapshot([root/"vibration"/f"{n}.py" for n in ("model","train","spectra","evaluate","file_evaluate")])
            },ensure_ascii=False,indent=2),encoding="utf-8")
            (folder/"run_log.json").write_text(json.dumps([{"stage":"split","rpm":rpm,
                "note":"byte-identical baseline split and pre-Z spectra reused for one-factor development comparison"}],indent=2),encoding="utf-8")
        print("세 RPM의 분할·FFT/log1p NPZ 복사본 SHA256 일치 확인. 정규화만 변경합니다.",flush=True)
        timings=[]
        env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1","PYTHONIOENCODING":"utf-8"}
        for rpm,folder in zip((900,1200,1800),folders):
            for stage in ("normalize","train","evaluate","file_evaluate"):
                started=time.perf_counter()
                print(f"시작 {rpm}RPM / {stage}",flush=True)
                command=[sys.executable,"-B",str(root/"run_project.py"),"--rpm",str(rpm),"--stage",stage,
                    "--run-name",args.run_name,"--split-mode","date_preliminary","--normalization","global_train"]
                with (folder/f"{stage}.log").open("w",encoding="utf-8") as stream:
                    result=subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
                timings.append({"rpm":rpm,"stage":stage,"wall_seconds":time.perf_counter()-started,"exit_code":result.returncode})
                (folder/"stage_timings.json").write_text(json.dumps([r for r in timings if r["rpm"]==rpm],indent=2),encoding="utf-8")
                if result.returncode:raise RuntimeError(f"{rpm}RPM {stage} 실패: {folder/(stage+'.log')}")
                print(f"완료 {rpm}RPM / {stage} ({timings[-1]['wall_seconds']:.1f}s)",flush=True)
    finally:
        after=snapshot(source_files)
        preservation={"unchanged":before==after,"existing_output_files":len(before),
                      "before_sha256":before,"after_sha256":after,"copied_baseline_artifacts":copies}
        for folder in folders:
            if folder.exists():(folder/"baseline_preservation.json").write_text(json.dumps(preservation,ensure_ascii=False,indent=2),encoding="utf-8")
        if before!=after:raise RuntimeError("기존 outputs 결과가 변경되었습니다.")
    print("AE 공통 정규화 개발 비교 완료. 기존 결과 보존, OCSVM·IF·DAE 실행 안 함.",flush=True)


if __name__=="__main__":main()
