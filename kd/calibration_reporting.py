"""Readable report and figures for the triggered calibration experiment."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def render_calibration_report(directory: Path) -> Path:
    report = json.loads((directory/"report.json").read_text())
    aggregate, runs, deltas = report["aggregate"], report["runs"], report["deltas"]
    seeds = (42,43,44)
    outcome = "met" if report["primary_success"] else "did not meet"
    uncertain = all(runs[f"kd_mmce_seed{s}"]["paired_calibration"]["ece_paired_bootstrap_95"][0] <= 0 <=
                    runs[f"kd_mmce_seed{s}"]["paired_calibration"]["ece_paired_bootstrap_95"][1] for s in seeds)
    interpretation = ("Every seed's paired ECE interval includes zero: the observed mean change is inconclusive, even if the numerical success rule is met."
                      if uncertain else "The numerical success rule is descriptive; inspect the paired intervals and seed variation before claiming a reliable improvement.")
    lines = ["# Triggered calibration-loss intervention on CIFAR-10", "",
             f"The intervention **{outcome} the predeclared success criterion**: lower mean test ECE with an accuracy loss of at most 0.5 percentage points.", "",
             f"Relative to matched fixed-KD controls, mean ECE changed by **{100*deltas['ece_15_bins']:+.2f} percentage points**, accuracy by **{100*deltas['accuracy']:+.2f} points**, NLL by **{deltas['nll']:+.4f}**, and Brier score by **{deltas['brier_score']:+.4f}**. Lower ECE/NLL/Brier is better.", "",
             interpretation, "",
             "## Protocol", "",
             "- Six fresh runs: fixed-KD control and triggered-MMCE student for each seed 42, 43, 44.",
             "- Same pretrained teacher, 45,000 training / 5,000 validation split, augmentation, initialization/data seeds, batch size 128, optimizer, and 20-epoch student schedule as the original CIFAR-10 study.",
             "- Same classical KD: temperature 4, weight 0.5, five-epoch warmup. The added penalty uses the student's ordinary T=1 probabilities.",
             "- Trigger: starting at epoch 5, validation ECE >3% AND mean confidence minus accuracy >1 percentage point for three consecutive checks.",
             "- Activation affects the following epoch. Gamma increases to 1 over three training epochs, then stays fixed; no repeated escalation or test-driven adjustment.",
             "- Penalty: unweighted empirical MMCE norm with Laplacian kernel exp(-|p_i-p_j|/0.4), including diagonal pairs and a small numerical square-root smoothing constant.",
             "- Checkpoint selection is unchanged: highest validation accuracy, earliest epoch on ties. ECE is not used to cherry-pick checkpoints.",
             "- All checkpoint hashes are frozen before final evaluation on the official 10,000-image test set. No post-training temperature scaling is applied.", "",
             "This is an exploratory follow-up: the official test set was already examined in the prior study. The present intervention settings were fixed before these new runs, and test labels did not drive the controller.", "",
             "## Test results", "",
             "Values are mean ± sample standard deviation across the three training seeds.", "",
             "| Condition | Accuracy (%) ↑ | ECE (%) ↓ | NLL ↓ | Brier ↓ | Macro F1 ↑ |",
             "|---|---:|---:|---:|---:|---:|"]
    for condition in ("control","mmce"):
        values=[]
        for metric in ("accuracy","ece_15_bins","nll","brier_score","macro_f1"):
            value=aggregate[condition][metric]
            factor=100 if metric in {"accuracy","ece_15_bins"} else 1
            digits=2 if factor==100 else 4
            values.append(f"{factor*value['mean']:.{digits}f} ± {factor*value['std']:.{digits}f}")
        lines.append(f"| {'Fixed KD' if condition=='control' else 'Triggered MMCE'} | " + " | ".join(values)+" |")
    lines.extend(["", "![Training feedback and held-out performance](figures/intervention.png)", "",
                  "## Activation and matched runs", "",
                  "| Seed | Triggered after epoch | First active epoch | Selected MMCE epoch | Control accuracy | MMCE accuracy | Control ECE | MMCE ECE | Student training overhead |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|",])
    rows=[]
    for seed in seeds:
        b,c=runs[f"kd_control_seed{seed}"],runs[f"kd_mmce_seed{seed}"]
        activation=c["calibration_controller"]["activated_after_epoch"]
        first=activation+1 if activation is not None else None
        overhead=c["training"]["mean_epoch_seconds"]/b["training"]["mean_epoch_seconds"]
        lines.append(f"| {seed} | {activation} | {first} | {c['best_epoch']} | {100*b['test']['accuracy']:.2f}% | {100*c['test']['accuracy']:.2f}% | {100*b['test']['ece_15_bins']:.2f}% | {100*c['test']['ece_15_bins']:.2f}% | {overhead:.2f}× |")
        for condition,result in (("control",b),("mmce",c)):
            q=result["test"]
            rows.append({"seed":seed,"condition":condition,"best_epoch":result["best_epoch"],
                         "activated_after_epoch":result["calibration_controller"]["activated_after_epoch"],
                         "accuracy":q["accuracy"],"ece":q["ece_15_bins"],"nll":q["nll"],"brier":q["brier_score"],
                         "parameters":result["inference"]["parameters"],"estimated_flops":result["inference"]["estimated_flops"],
                         "mean_epoch_seconds":result["training"]["mean_epoch_seconds"]})
    lines.extend(["", "Training overhead excludes teacher pretraining. MMCE and the controller are training-only: all students have the same inference parameter count and operations.", "",
                  "## Paired uncertainty", "",
                  "Intervals use 2,000 paired test-example bootstrap resamples; ECE is recomputed within each resample using the same 15 confidence bins. These intervals condition on fitted models and do not measure retraining uncertainty.", "",
                  "| Seed | Accuracy delta and 95% CI (pp) | ECE delta and 95% CI (pp) | NLL delta and 95% CI | Brier delta and 95% CI |",
                  "|---|---|---|---|---|"])
    for seed in seeds:
        result=runs[f"kd_mmce_seed{seed}"]
        a,c=result["paired_accuracy"],result["paired_calibration"]
        cells=[]
        for delta,interval,factor in ((a["accuracy_delta"],a["paired_bootstrap_95"],100),
                                      (c["ece_delta"],c["ece_paired_bootstrap_95"],100),
                                      (c["nll_delta"],c["nll_paired_bootstrap_95"],1),
                                      (c["brier_delta"],c["brier_paired_bootstrap_95"],1)):
            cells.append(f"{factor*delta:+.4f} [{factor*interval[0]:+.4f}, {factor*interval[1]:+.4f}]")
        lines.append(f"| {seed} | " + " | ".join(cells)+" |")
    lines.extend(["", "![Reliability and confidence](figures/reliability.png)", "",
                  "## Limits", "", *[f"- {value}" for value in report["limitations"]],
                  "- This implements the unweighted MMCE norm, not the paper's reweighted correct/incorrect variant. The fixed coefficient is not a claim about optimal calibration regularization.",
                  "- If the original training predictions become nearly all correct, a training-batch calibration penalty may fail to represent held-out errors.", "",
                  "## Reproduce", "", "```bash",
                  f"python -m kd calibration-study --baseline runs/cifar10 --output {directory} --device mps",
                  f"MPLCONFIGDIR=/private/tmp/kd-matplotlib python -m kd.calibration_reporting --study {directory}",
                  "```", "",
                  "`protocol.json` captures the source hashes and all settings before training. Each run's history contains the observed calibration signal, trigger counter, activation epoch, next-epoch weight, and actual penalty. `selection.json` freezes checkpoint hashes before the test phase. Predictions and class-level metrics are stored under each run's `test/` directory.", "",
                  "Loss reference: Kumar, Sarawagi, and Jain, [Trainable Calibration Measures for Neural Networks from Kernel Mean Embeddings (ICML 2018)](https://proceedings.mlr.press/v80/kumar18a.html), empirical unweighted MMCE formulation.", ""])
    path=directory/"report.md"
    path.write_text("\n".join(lines))
    with (directory/"results.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_intervention(directory,report)
    return path


def plot_intervention(directory: Path, report: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    figures=directory/"figures"
    figures.mkdir(exist_ok=True)
    colors={"control":"#52616b","mmce":"#2274a5"}
    names={"control":"Fixed KD","mmce":"Triggered MMCE"}
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for condition in ("control","mmce"):
        histories=[json.loads((directory/f"kd_{condition}_seed{s}"/"history.json").read_text()) for s in (42,43,44)]
        for ax,metric in ((axes[0,0],"ece_15_bins"),(axes[0,1],"accuracy")):
            values=np.array([[100*h["val"][metric] for h in history] for history in histories])
            x=np.arange(1,values.shape[1]+1)
            mean,sd=values.mean(0),values.std(0,ddof=1)
            ax.plot(x,mean,label=names[condition],color=colors[condition])
            ax.fill_between(x,mean-sd,mean+sd,color=colors[condition],alpha=.15)
        if condition=="mmce":
            for seed,history in zip((42,43,44),histories):
                axes[1,0].plot([h["epoch"] for h in history],[h["calibration_weight"] for h in history],label=f"Seed {seed}")
    axes[0,0].set(title="Validation ECE (mean ± SD)",xlabel="Epoch",ylabel="ECE (%)")
    axes[0,1].set(title="Validation accuracy (mean ± SD)",xlabel="Epoch",ylabel="Accuracy (%)")
    axes[1,0].set(title="Actual calibration-loss coefficient",xlabel="Training epoch",ylabel="Gamma")
    for i,condition in enumerate(("control","mmce")):
        scores=100*np.asarray(report["aggregate"][condition]["ece_15_bins"]["values"])
        axes[1,1].errorbar(i,scores.mean(),yerr=scores.std(ddof=1),fmt="D",color=colors[condition],capsize=5)
        axes[1,1].scatter(i+np.linspace(-.1,.1,3),scores,color=colors[condition],s=25)
    axes[1,1].set(title="Official test ECE (three seeds)",xticks=[0,1],xticklabels=list(names.values()),ylabel="ECE (%)")
    for ax in (axes[0,0],axes[0,1],axes[1,0]):
        ax.legend(frameon=False)
    for ax in axes.flat:
        ax.grid(alpha=.15)
    fig.savefig(figures/"intervention.png",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    axes[0].plot([0,1],[0,1],"--",color="#999999",label="Perfect calibration")
    for condition in ("control","mmce"):
        probabilities,labels=[],[]
        for seed in (42,43,44):
            with np.load(directory/f"kd_{condition}_seed{seed}"/"test"/"predictions.npz") as archive:
                probabilities.append(archive["probabilities"])
                labels.append(archive["labels"])
        p,y=np.concatenate(probabilities),np.concatenate(labels)
        confidence=p.max(1)
        correct=p.argmax(1)==y
        bins=np.minimum((confidence*15).astype(int),14)
        points=[(confidence[bins==i].mean(),correct[bins==i].mean()) for i in range(15) if (bins==i).sum()>=30]
        axes[0].plot(*zip(*points),"o-",label=names[condition],color=colors[condition],markersize=4)
        axes[1].hist(confidence,bins=np.linspace(0,1,21),weights=np.ones(len(confidence))/len(confidence),
                     histtype="step",linewidth=2,color=colors[condition],label=names[condition])
    axes[0].set(title="Reliability (pooled seeds; bins ≥30 samples)",xlabel="Mean confidence",ylabel="Observed accuracy",xlim=(0,1),ylim=(0,1))
    axes[1].set(title="Confidence distribution (pooled seeds)",xlabel="Confidence",ylabel="Fraction of predictions")
    for ax in axes:
        ax.legend(frameon=False)
        ax.grid(alpha=.15)
    fig.savefig(figures/"reliability.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=Path("runs/cifar10-calibration"))
    print(render_calibration_report(parser.parse_args().study))
