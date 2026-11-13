"""전체 파일 통계와 RMS 중앙값 대표 파형을 그립니다. 원본·모델은 읽지 않습니다."""

from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

DATES=("20260912","20260916","20260919")
CONDITIONS=("Normal1","Normal2")
DATE_COLORS={"20260912":"#2878B5","20260916":"#2D9C54","20260919":"#D85F26"}
CONDITION_COLORS={"Normal1":"#2878B5","Normal2":"#D85F26"}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))


def distribution(ax,rows,metric,title):
    rng=np.random.default_rng(42)
    for i,date in enumerate(DATES):
        for j,condition in enumerate(CONDITIONS):
            values=np.array([float(r[metric]) for r in rows if r["date"]==date and r["condition"]==condition])
            position=i+(-.18 if j==0 else .18)
            box=ax.boxplot([values],positions=[position],widths=.28,showfliers=False,patch_artist=True,medianprops={"color":"black"})
            box["boxes"][0].set(facecolor=CONDITION_COLORS[condition],alpha=.25)
            ax.scatter(position+rng.uniform(-.09,.09,len(values)),values,s=10,color=CONDITION_COLORS[condition],alpha=.6)
    ax.set_xticks([0,1,2],["Sep 12","Sep 16","Sep 19"])
    ax.set_title(title);ax.set_xlim(-.6,2.6);ax.grid(axis="y",alpha=.2)
    ax.legend(handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor=c,label=n) for n,c in CONDITION_COLORS.items()],fontsize=8)


def spectrum(ax,frequency,profiles,dates,conditions,condition,low=False):
    mask=frequency<=200 if low else np.ones(len(frequency),dtype=bool)
    for date in DATES:
        values=profiles[(dates==date)&(conditions==condition)]
        med=np.median(values,axis=0);lower,upper=np.percentile(values,[10,90],axis=0)
        ax.plot(frequency[mask],np.maximum(med[mask],1e-15),color=DATE_COLORS[date],lw=1.1,label=f"{date[4:6]}/{date[6:]} (n={len(values)})")
        ax.fill_between(frequency[mask],np.maximum(lower[mask],1e-15),np.maximum(upper[mask],1e-15),color=DATE_COLORS[date],alpha=.12)
    ax.set_yscale("log");ax.set_xlabel("Frequency (Hz)");ax.grid(alpha=.17);ax.legend(fontsize=8)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();output=args.output;figures=output/"figures";figures.mkdir(exist_ok=True)
    all_metrics={rpm:read_csv(output/f"{rpm}_file_metrics.csv") for rpm in (900,1200,1800)}
    for filename,fields in [("raw_metric_distributions",[("rms","Raw RMS"),("ac_rms","AC RMS (whole-file mean removed)"),("peak_abs","Raw maximum absolute value")]),
                            ("band_distributions",[("low_band_rss","0 <= f < 100 Hz"),("mid_band_rss","100 <= f < 1000 Hz"),("high_band_rss","1000 <= f <= 10000 Hz")])]:
        fig,axes=plt.subplots(3,3,figsize=(19,13),constrained_layout=True)
        for i,rpm in enumerate((900,1200,1800)):
            for j,(field,label) in enumerate(fields):
                distribution(axes[i,j],all_metrics[rpm],field,f"{rpm} RPM | {label}")
                axes[i,j].set_ylabel("Native input units" if filename.startswith("raw") else "Mean segment band RSS (not time RMS)")
        fig.suptitle("All quality-passed Normal1/Normal2 files | fixed dates, no retraining",fontsize=15)
        fig.savefig(figures/f"{filename}.png",dpi=160);plt.close(fig)
    norm,na=plt.subplots(3,3,figsize=(21,14),constrained_layout=True)
    contribution,ca=plt.subplots(3,2,figsize=(14,12),constrained_layout=True)
    for i,rpm in enumerate((900,1200,1800)):
        with np.load(output/f"{rpm}_file_profiles.npz") as data:profiles={k:data[k] for k in data.files}
        f=profiles["frequency"];dates=profiles["date"];conditions=profiles["condition"]
        representatives=json.loads((output/f"{rpm}_representatives.json").read_text(encoding="utf-8"))
        with np.load(output/f"{rpm}_representative_waves.npz") as waves:
            fig,axes=plt.subplots(4,2,figsize=(17,18),constrained_layout=True)
            for j,condition in enumerate(CONDITIONS):
                for date in DATES:
                    wave=waves[f"{date}_{condition}"]
                    axes[0,j].plot(np.arange(len(wave))/25600,wave,lw=.5,alpha=.65,color=DATE_COLORS[date],label=f"{date[4:6]}/{date[6:]}")
                axes[0,j].set_title(f"{rpm} RPM / {condition} | RMS-median examples ONLY")
                axes[0,j].set_xlabel("Time from file start (s); phases not aligned")
                axes[0,j].set_ylabel("Raw CH3 (native units)");axes[0,j].legend(fontsize=8);axes[0,j].grid(alpha=.15)
                spectrum(axes[1,j],f,profiles["amplitude"],dates,conditions,condition)
                axes[1,j].set_title(f"{condition} | all-file median and 10-90% envelope")
                axes[1,j].set_ylabel("Linear FFT amplitude (log display)")
                spectrum(axes[2,j],f,profiles["amplitude"],dates,conditions,condition,low=True)
                axes[2,j].set_title(f"{condition} | low-frequency detail")
                axes[2,j].set_ylabel("Linear FFT amplitude (log display)")
                spectrum(axes[3,j],f,profiles["log1p"],dates,conditions,condition)
                axes[3,j].set_title(f"{condition} | log1p AFTER segment transform, BEFORE Z-score")
                axes[3,j].set_ylabel("Pre-Z log1p values (log display)")
            fig.suptitle(f"{rpm} RPM | waveform examples plus ALL-file spectral statistics",fontsize=16)
            fig.savefig(figures/f"{rpm}_waveform_spectra.png",dpi=150);plt.close(fig)
        std=profiles["train_std"];small=profiles["small_std_mask"]
        na[i,0].semilogy(f,std,lw=.8,label="Stored Train std (floored)")
        na[i,0].scatter(f[small],std[small],s=7,color="#D85F26",label="Lowest 10% by rank (321 bins)")
        na[i,0].axhline(1e-6,color="gray",linestyle="--",label="Existing floor 1e-6")
        na[i,0].set_title(f"{rpm} RPM | fixed Sep 12 Train std")
        na[i,0].set_ylabel("Train std of segment log1p amplitude");na[i,0].legend(fontsize=7)
        shift_rows=read_csv(output/f"{rpm}_normalization_shift.csv")
        for j,condition in enumerate(CONDITIONS):
            baseline=np.median(profiles["log1p"][(dates==DATES[0])&(conditions==condition)],axis=0)
            for date in DATES[1:]:
                current=np.median(profiles["log1p"][(dates==date)&(conditions==condition)],axis=0)
                delta=current-baseline;style="-" if condition=="Normal1" else "--"
                label=f"{condition} {date[4:6]}/{date[6:]} vs 09/12"
                na[i,1].semilogy(f,np.maximum(np.abs(delta),1e-12),color=DATE_COLORS[date],ls=style,lw=.8,label=label)
                na[i,2].semilogy(f,np.maximum(np.abs(delta/std),1e-12),color=DATE_COLORS[date],ls=style,lw=.8,label=label)
            selected=[next(r for r in shift_rows if r["condition"]==condition and r["date"]==date) for date in DATES[1:]]
            before=np.array([float(r["small_std_raw_shift_share"])*100 for r in selected])
            after=np.array([float(r["small_std_standardized_shift_share"])*100 for r in selected])
            ax=ca[i,j];x=np.arange(2)
            for values,offset,color,label in ((before,-.16,"#999999","Before Z-score"),(after,.16,"#8053B5","After fixed Z-score")):
                bars=ax.bar(x+offset,values,width=.30,color=color,label=label)
                for b,value in zip(bars,values):ax.annotate(f"{value:.2f}%",(b.get_x()+b.get_width()/2,value),xytext=(0,3),textcoords="offset points",ha="center",fontsize=8)
            ax.set_title(f"{rpm} RPM / {condition} | lowest-std 321 bins")
            ax.set_xticks(x,["Sep 16 vs Sep 12","Sep 19 vs Sep 12"])
            ax.set_ylim(0,max(float(max(before.max(),after.max()))*1.25,1))
            ax.set_ylabel("Share of squared spectral shift (%)")
            ax.grid(axis="y",alpha=.2);ax.legend(fontsize=8)
        na[i,1].set_title(f"{rpm} RPM | absolute log1p median shift")
        na[i,2].set_title(f"{rpm} RPM | same shift divided by stored std")
        na[i,1].set_ylabel("|delta log1p| (log display)")
        na[i,2].set_ylabel("|delta log1p| / std (log display)")
        for ax in na[i]:ax.set_xlabel("Frequency (Hz)");ax.grid(alpha=.15)
        for ax in na[i,1:]:ax.legend(fontsize=7)
    norm.suptitle("Fixed normalization diagnostic | per-condition all-file median shift | NOT AE reconstruction error",fontsize=15)
    norm.savefig(figures/"normalization_diagnostics.png",dpi=150);plt.close(norm)
    contribution.suptitle("Do the lowest-std bins gain disproportionate weight? | original normalization unchanged",fontsize=14)
    contribution.savefig(figures/"small_std_contributions.png",dpi=160);plt.close(contribution)
    print("원본 통계·파형/스펙트럼·정규화 확대 그래프 7개 저장 완료",flush=True)


if __name__=="__main__":main()
