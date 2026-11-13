"""추론 프로세스와 분리하여 정상 파일 점수 분포를 그립니다."""

from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def panel(ax, rows, rpm, model):
    colors={"Normal1":"#2378B5","Normal2":"#E27C24"}
    rng=np.random.default_rng(42)
    threshold=float(rows[0]["threshold"])
    for i,split in enumerate(("train","validation","test_normal")):
        for j,condition in enumerate(("Normal1","Normal2")):
            values=np.array([float(r["score"]) for r in rows if r["split"]==split and r["condition"]==condition])
            x=i+(-0.18 if j==0 else 0.18)
            box=ax.boxplot([values],positions=[x],widths=0.27,patch_artist=True,showfliers=False,
                           medianprops={"color":"black","linewidth":1.4})
            box["boxes"][0].set(facecolor=colors[condition],alpha=0.30,edgecolor=colors[condition])
            ax.scatter(x+rng.uniform(-0.095,0.095,len(values)),values,s=12,c=colors[condition],alpha=0.65,zorder=3)
    ax.axhline(threshold,color="#C52222",linestyle="--",linewidth=1.6)
    label={"autoencoder":"AE","ocsvm":"OCSVM","iforest":"IF"}[model]
    ax.set_title(f"{rpm} RPM / {label} | File threshold = {threshold:.6g}")
    ax.set_xticks([0,1,2],["Train\nSep 12","Validation\nSep 16","Test normal\nSep 19"])
    ax.set_xlim(-0.6,2.6)
    ax.set_ylabel("File anomaly score (higher = more anomalous)")
    ax.ticklabel_format(axis="y",style="plain",useOffset=False)
    ax.grid(axis="y",alpha=0.22)
    handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor=c,label=n) for n,c in colors.items()]
    handles.append(Line2D([0],[0],color="#C52222",linestyle="--",label="Existing validation file Q95"))
    ax.legend(handles=handles,fontsize=8,loc="best")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    figures=args.output/"figures"
    figures.mkdir(exist_ok=True)
    overview,axes=plt.subplots(3,3,figsize=(21,15),constrained_layout=True)
    for i,rpm in enumerate((900,1200,1800)):
        for j,model in enumerate(("autoencoder","ocsvm","iforest")):
            with (args.output/f"{rpm}_{model}_file_scores.csv").open(encoding="utf-8-sig",newline="") as stream:
                rows=[r for r in csv.DictReader(stream) if r["split"] in {"train","validation","test_normal"}]
            panel(axes[i,j],rows,rpm,model)
            fig,ax=plt.subplots(figsize=(11,6),constrained_layout=True)
            panel(ax,rows,rpm,model)
            fig.savefig(figures/f"{rpm}_{model}.png",dpi=180)
            plt.close(fig)
    overview.suptitle("Normal file-score distributions | fixed models, original thresholds | date-group preliminary experiment",fontsize=16)
    overview.savefig(args.output/"normal_scores_overview.png",dpi=160)
    plt.close(overview)
    print("정상 분포 그래프 9개와 전체 비교 그래프 저장 완료",flush=True)


if __name__=="__main__":
    main()
