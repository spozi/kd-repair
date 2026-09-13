"""Render saved study results; never trains or selects models using test scores."""

import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np


def render_report(study: Path) -> Path:
    report = json.loads((study / "report.json").read_text())
    methods = ("supervised", "kd", "dkd")
    seeds = report["aggregate"]["supervised"]["seeds"]
    runs = report["runs"]
    selected = report["selection"]["method"]
    teacher_config = report["protocol"]["configs"][0]
    student_config = report["protocol"]["configs"][1]
    environment_path = study / "environment.json"
    environment = json.loads(environment_path.read_text()) if environment_path.exists() else {}
    baseline_accuracy = report["aggregate"]["supervised"]["test_accuracy_mean"]
    findings = []
    for method in ("kd", "dkd"):
        delta = 100 * (report["aggregate"][method]["test_accuracy_mean"] - baseline_accuracy)
        findings.append(f"{method.upper()} changed mean test accuracy by **{delta:+.2f} percentage points** relative to supervised training.")
    ece_means = {method: statistics.mean(runs[f"{method}_seed{seed}"]["test"]["ece_15_bins"] for seed in seeds)
                 for method in methods}
    findings.append("Mean calibration error (ECE; lower is better): " + ", ".join(
        f"{method.upper()} {100*ece_means[method]:.2f}%" for method in methods) + ".")
    lines = ["# CIFAR-10 knowledge distillation evaluation", "",
             f"The validation-selected method was **{selected.upper()}**. All results below use the official 10,000-image test set.", "",
             " ".join(findings), "",
             "## Protocol", "",
             f"Execution device: {teacher_config['train']['device']}; hardware: {environment.get('processor', 'see inference metadata')}. Software versions, source hashes, and dataset checksum are recorded in `environment.json` when captured.", "",
             "- Official CIFAR-10: 45,000 training images, 5,000 stratified validation images, and 10,000 test images.",
             "- Exactly 4,500 training and 500 validation images per class; split seed 2026, independent of model seeds.",
             f"- One six-convolution teacher (seed 41), trained for {teacher_config['train']['epochs']} epochs.",
             f"- A smaller six-convolution student, trained for {student_config['train']['epochs']} epochs per run; seeds {', '.join(map(str, seeds))}.",
             "- Equal student optimizer, learning-rate schedule, augmentation, initialization seed, data order, and epoch budgets within each seed.",
             "- Fixed response settings: temperature 4, outer weight 0.5; DKD alpha 1, beta 8, five-epoch warmup.",
             "- Best checkpoint selected by validation accuracy. Method selected by mean validation accuracy across seeds before opening the test split.",
             "- Random padded crops and horizontal flips for training; unmodified 32×32 geometry for evaluation. Fixed input scaling to [-1, 1].", "",
             "## Test accuracy", "",
             "Student values are mean ± sample standard deviation across training seeds, in percentage points. The teacher is one fitted model.", "",
             "| Model / method | Validation accuracy | Test accuracy | Parameters | Estimated MFLOPs | Median latency (ms) |",
             "|---|---:|---:|---:|---:|---:|"]
    rows = []
    t = runs["teacher"]
    lines.append(f"| Teacher | {100*t['validation']['accuracy']:.2f}% | {100*t['test']['accuracy']:.2f}% | {t['inference']['parameters']:,} | {t['inference']['estimated_flops']/1e6:.2f} | {t['inference']['latency_median_ms']:.3f} |")
    for method in methods:
        names = [f"{method}_seed{seed}" for seed in seeds]
        aggregate = report["aggregate"][method]
        latency = statistics.mean(runs[name]["inference"]["latency_median_ms"] for name in names)
        cost = runs[names[0]]["inference"]
        sd = aggregate["test_accuracy_std"]
        uncertainty = f" ± {100*sd:.2f}" if sd is not None else " (one seed)"
        lines.append(f"| {method.upper()} | {100*aggregate['mean_validation_accuracy']:.2f}% | {100*aggregate['test_accuracy_mean']:.2f}{uncertainty}% | {cost['parameters']:,} | {cost['estimated_flops']/1e6:.2f} | {latency:.3f} |")
    lines.extend(["", "Student latency entries average the per-run median latency measurements; teacher latency is from its single run. All student variants have the same inference architecture and operation count, so timing differences among student methods reflect measurement conditions rather than a structural speed improvement.", "",
                  "![Validation curves and test accuracy](figures/learning-and-accuracy.png)", "",
                  "## Per-run results", "", "Wilson 95% intervals quantify finite test-sample uncertainty for each fitted model. They do not include uncertainty from retraining.", "",
                  "| Run | Best epoch | Test accuracy | Wilson 95% | Macro F1 | NLL | Brier | ECE | Train seconds |",
                  "|---|---:|---:|---|---:|---:|---:|---:|---:|"])
    for name, result in runs.items():
        q = result["test"]
        low, high = q["accuracy_wilson_95"]
        seconds = result["training"]["total_epoch_seconds"]
        lines.append(f"| {name} | {result['best_epoch']} | {100*q['accuracy']:.2f}% | [{100*low:.2f}, {100*high:.2f}]% | {q['macro_f1']:.4f} | {q['nll']:.4f} | {q['brier_score']:.4f} | {q['ece_15_bins']:.4f} | {seconds:.1f} |")
        rows.append({"run": name, "best_epoch": result["best_epoch"], "test_accuracy": q["accuracy"],
                     "macro_f1": q["macro_f1"], "nll": q["nll"], "brier": q["brier_score"],
                     "ece": q["ece_15_bins"], "train_seconds": seconds,
                     "parameters": result["inference"]["parameters"],
                     "estimated_flops": result["inference"]["estimated_flops"],
                     "latency_ms": result["inference"]["latency_median_ms"]})
    lines.extend(["", "## Paired differences from supervised students", "",
                  "Each comparison uses the same seed and the same test examples. Bootstrap intervals resample paired test-example outcomes. McNemar p-values are exploratory and unadjusted for multiple comparisons. Training overhead compares student epoch times and excludes the one-time teacher pretraining cost, which is shown separately in the teacher row above.", "",
                  "| Candidate | Accuracy difference (pp) | Paired bootstrap 95% (pp) | Baseline only correct | Candidate only correct | Exact McNemar p | Training overhead |",
                  "|---|---:|---|---:|---:|---:|---:|"])
    for seed in seeds:
        for method in ("kd", "dkd"):
            result = runs[f"{method}_seed{seed}"]
            paired = result["vs_supervised"]
            low, high = paired["paired_bootstrap_95"]
            overhead = result["training"]["mean_epoch_seconds"] / runs[f"supervised_seed{seed}"]["training"]["mean_epoch_seconds"]
            lines.append(f"| {method}_seed{seed} | {100*paired['accuracy_delta']:+.2f} | [{100*low:+.2f}, {100*high:+.2f}] | {paired['baseline_only_correct']} | {paired['candidate_only_correct']} | {paired['mcnemar_exact_two_sided_p']:.4g} | {overhead:.2f}× |")
    lines.extend(["", "## Class errors and calibration", "",
                  "The selected method's confusion matrix averages counts over its independently trained students, then normalizes each true-class row. Calibration pools predictions across those same fitted students; repeated test images are not independent new observations.", "",
                  "![Confusion and calibration](figures/confusion-and-calibration.png)", "",
                  "## Interpretation and limits", ""])
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.extend(["- Model sizes differ by width; this does not establish how KD behaves for other architectures or driving tasks.",
                  "- FLOPs include twice the Conv2d/Linear MAC count only. Batch-one latency excludes preprocessing and transfer and is specific to this machine.",
                  "- Training overhead includes loading and training, but excludes validation, checkpointing, and profiling. Timing can vary with system load.", "",
                  "## Reproduce and inspect", "", "```bash",
                  f"python -m kd cifar10 --output {study} --root {teacher_config['data']['root']} --device {teacher_config['train']['device']} --teacher-epochs {teacher_config['train']['epochs']} --student-epochs {student_config['train']['epochs']} --seeds {' '.join(map(str, seeds))}",
                  f"python -m kd.reporting --study {study}", "```", "",
                  "The same study command resumes completed epochs/runs and refuses a different protocol. Use another output directory for a new study. `protocol.json` fixes the configuration; `selection.json` records validation selection and checkpoint hashes; each run contains `history.json`, `data.json`, checkpoints, and `test/predictions.npz`, `metrics.json`, and `confusion.csv`.", "",
                  "Dataset: Alex Krizhevsky, [Learning Multiple Layers of Features from Tiny Images (2009), CIFAR project](https://cave.cs.toronto.edu/kriz/cifar.html).", ""])
    with (study / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path = study / "report.md"
    path.write_text("\n".join(lines))
    plot_report(study, report)
    return path


def plot_report(study: Path, report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    directory = study / "figures"
    directory.mkdir(exist_ok=True)
    methods = ("supervised", "kd", "dkd")
    colors = {"supervised": "#52616b", "kd": "#2274a5", "dkd": "#d97926"}
    seeds = report["aggregate"]["supervised"]["seeds"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    teacher_history = json.loads((study / "teacher" / "history.json").read_text())
    axes[0].plot([h["epoch"] for h in teacher_history], [100*h["val"]["accuracy"] for h in teacher_history],
                  color="#222222", linestyle="--", label="Teacher (one seed)")
    for i, method in enumerate(methods):
        histories = [json.loads((study / f"{method}_seed{seed}" / "history.json").read_text()) for seed in seeds]
        values = np.asarray([[100*h["val"]["accuracy"] for h in history] for history in histories])
        x = np.arange(1, values.shape[1] + 1)
        mean, sd = values.mean(0), values.std(0, ddof=1) if len(seeds) > 1 else np.zeros(values.shape[1])
        axes[0].plot(x, mean, label=method.upper(), color=colors[method])
        axes[0].fill_between(x, mean-sd, mean+sd, color=colors[method], alpha=0.15)
        scores = 100*np.asarray(report["aggregate"][method]["test_accuracies"])
        axes[1].errorbar(i, scores.mean(), yerr=scores.std(ddof=1) if len(seeds)>1 else None,
                         fmt="D", color=colors[method], markersize=8, capsize=5)
        axes[1].scatter(i + np.linspace(-0.12, 0.12, len(seeds)), scores, color="#111111", s=24, zorder=3)
        axes[1].text(i, scores.max() + 0.7, f"{scores.mean():.2f}%", ha="center")
    axes[0].set(xlabel="Epoch", ylabel="Validation accuracy (%)", title="Validation learning curves (mean ± SD)")
    axes[0].legend(frameon=False)
    axes[1].set(xticks=range(3), xticklabels=[m.upper() for m in methods], ylabel="Test accuracy (%)",
                title="Test accuracy (dots: seeds; diamonds: mean ± SD)")
    all_scores = [100*s for item in report["aggregate"].values() for s in item["test_accuracies"]]
    axes[1].set_ylim(max(0, min(all_scores)-3), min(100, max(all_scores)+3))
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
    fig.savefig(directory / "learning-and-accuracy.png", dpi=180)
    plt.close(fig)

    selected = report["selection"]["method"]
    archives = [np.load(study / f"{selected}_seed{seed}" / "test" / "predictions.npz") for seed in seeds]
    classes = archives[0]["classes"].tolist()
    confusion = sum(np.asarray(report["runs"][f"{selected}_seed{seed}"]["test"]["confusion_matrix"]) for seed in seeds)
    normalized = confusion / confusion.sum(1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    im = axes[0].imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    axes[0].set(xticks=range(10), yticks=range(10), xticklabels=classes, yticklabels=classes,
                xlabel="Predicted class", ylabel="True class", title=f"{selected.upper()} confusion, all seeds")
    plt.setp(axes[0].get_xticklabels(), rotation=50, ha="right")
    for row in range(10):
        for col in range(10):
            if normalized[row, col] >= 0.015:
                axes[0].text(col, row, f"{100*normalized[row,col]:.0f}", ha="center", va="center", fontsize=8,
                              color="white" if normalized[row,col] > 0.5 else "#222222")
    fig.colorbar(im, ax=axes[0], label="Fraction within true class", shrink=0.8)
    axes[1].plot([0, 1], [0, 1], "--", color="#888888", label="Perfect calibration")
    for method in methods:
        with_files = [np.load(study / f"{method}_seed{seed}" / "test" / "predictions.npz") for seed in seeds]
        probs = np.concatenate([a["probabilities"] for a in with_files])
        labels = np.concatenate([a["labels"] for a in with_files])
        confidence = probs.max(1)
        correct = probs.argmax(1) == labels
        bins = np.minimum((confidence * 15).astype(int), 14)
        points = [(confidence[bins == i].mean(), correct[bins == i].mean()) for i in range(15) if np.any(bins == i)]
        axes[1].plot(*zip(*points), "o-", color=colors[method], label=method.upper(), markersize=4)
        for archive in with_files:
            archive.close()
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted confidence", ylabel="Observed accuracy",
                title="Reliability diagram (15 bins, pooled seeds)")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.15)
    for archive in archives:
        archive.close()
    fig.savefig(directory / "confusion-and-calibration.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("runs/cifar10"))
    print(render_report(parser.parse_args().study))
