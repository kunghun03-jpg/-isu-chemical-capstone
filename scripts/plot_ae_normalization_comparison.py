"""AE 두 정규화 방식의 파일 단위 지표만 시각화합니다."""

from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    with (args.output/"metrics_comparison.csv").open(encoding="utf-8-sig",newline="") as stream:
        rows=[r for r in csv.DictReader(stream) if r["level"]=="file"]
    fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    for ax,(metric,title) in zip(axes.flat,[("fpr","Normal false-positive rate (lower is better)"),
                                         ("detection_rate","Anomaly detection rate"),("auroc","AUROC"),("ap","Average Precision (not accuracy)")]):
        for variant,offset,color,label in (("per_frequency",-.18,"#2878B5","Original per-frequency"),
                                           ("global_train",.18,"#D85F26","Global Train scalar")):
            values=[float(next(r[metric] for r in rows if r["rpm"]==str(rpm) and r["normalization"]==variant)) for rpm in (900,1200,1800)]
            bars=ax.bar(np.arange(3)+offset,values,width=.34,color=color,label=label)
            for bar,value in zip(bars,values):
                text=f"{value:.1%}" if metric in {"fpr","detection_rate"} else f"{value:.3f}"
                ax.annotate(text,(bar.get_x()+bar.get_width()/2,value),xytext=(0,3),textcoords="offset points",ha="center",fontsize=9)
        ax.set_xticks([0,1,2],["900 RPM","1200 RPM","1800 RPM"])
        ax.set_ylim(0,1.12);ax.set_title(title);ax.legend(fontsize=9);ax.grid(axis="y",alpha=.2)
    fig.suptitle("AE normalization development comparison | previously examined test data | one fixed alternative only",fontsize=14)
    fig.savefig(args.output/"metrics_comparison.png",dpi=170)
    plt.close(fig)


if __name__=="__main__":main()
