"""Render CPC study artifacts without model inference or parameter fitting."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CONDITIONS = ("KD", "KD+TS", "KD+CPC", "KD+CPC+TS")
SCALARS = ("accuracy", "aurc", "ece_15_bins", "ece_equal_mass_15_bins", "ece_l2_15_bins",
           "classwise_ece_15_bins", "nll", "brier_score", "macro_f1",
           "selective_accuracy_80", "selective_accuracy_90")


def _comparison_cells(comparison, metric, scale=1):
    value = comparison["metrics"][metric]
    lo, hi = value["paired_bootstrap_95"]
    return f"{value['delta'] * scale:+.5f} [{lo * scale:+.5f}, {hi * scale:+.5f}]"


def _plots(directory, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    colors = {c: f"C{i}" for i, c in enumerate(CONDITIONS)}
    for condition in CONDITIONS:
        for index, row in enumerate(r for r in report["rows"] if r["condition"] == condition):
            curve = row["test"]["risk_coverage_curve"]
            axes[0].plot(curve["coverage"], np.array(curve["risk"]) * 100,
                         color=colors[condition], alpha=.65, linewidth=1,
                         label=condition if index == 0 else None)
    axes[0].set(xlabel="Coverage: fraction of predictions accepted", ylabel="Error among accepted predictions (%)",
                title="Risk–coverage: one curve per seed", xlim=(0, 1))
    axes[0].legend(fontsize=8)
    means = [report["aggregate"][c]["ece_15_bins"]["mean"] * 100 for c in CONDITIONS]
    deviations = [report["aggregate"][c]["ece_15_bins"]["std"] * 100 for c in CONDITIONS]
    axes[1].bar(CONDITIONS, means, yerr=deviations, capsize=4, color=[colors[c] for c in CONDITIONS])
    axes[1].set(ylabel="ECE (%) — lower is better", title="Mean ECE ± sample SD across three seeds")
    axes[1].tick_params(axis="x", rotation=15)
    fig.savefig(figures / "comparison.png", dpi=160)
    plt.close(fig)


def render_report(output="runs/cifar10-cpc"):
    """Regenerate report.md, results.csv and figures from the signed report.json."""
    directory = Path(output)
    from .checkpoints import fingerprint
    manifest = json.loads((directory / "report_manifest.json").read_text())
    for key, filename in (("report_sha256", "report.json"), ("protocol_sha256", "protocol.json"),
                          ("calibration_selection_sha256", "calibration_selection.json")):
        if manifest[key] != fingerprint(directory / filename):
            raise ValueError(f"Report manifest mismatch: {filename}")
    report = json.loads((directory / "report.json").read_text())
    aggregate, primary = report["aggregate"], report["comparisons"]["primary"]
    delta = report["primary_deltas"]
    aurc_interval = primary["metrics"]["aurc"]["paired_bootstrap_95"]
    if report["primary_numerical_success"]:
        headline = "CPC passes the preliminary numerical rule: lower mean AURC after temperature scaling, within the accuracy guardrail."
    else:
        headline = "CPC does not pass the preliminary numerical rule for this experiment."
    if aurc_interval[1] < 0:
        evidence = "The paired test-image interval supports a lower AURC for these fitted models."
    elif aurc_interval[0] > 0:
        evidence = "The paired test-image interval supports a higher (worse) AURC for these fitted models."
    else:
        evidence = "The paired test-image interval includes zero, so the AURC direction remains uncertain."
    lines = ["# CPC during student KD: results", "", f"**{headline}** {evidence}", "",
             "The decisive comparison is **KD+CPC+TS versus KD+TS**. It asks whether changing student training adds value after the simple temperature correction. "
             "AURC measures how effectively confidence separates useful predictions from errors; lower is better. This is an exploratory three-seed experiment.", "",
             "## Main comparison", "",
             "Values are mean ± sample standard deviation across seeds 42, 43 and 44. Accuracy and ECE are percentages; AURC, NLL and Brier are unscaled.", "",
             "| Condition | What changed | Accuracy ↑ (%) | AURC ↓ | ECE ↓ (%) | NLL ↓ | Brier ↓ |",
             "|---|---|---:|---:|---:|---:|---:|"]
    descriptions = {"KD": "Standard student training", "KD+TS": "Same KD checkpoint; fitted temperature",
                    "KD+CPC": "CPC added during student training", "KD+CPC+TS": "Same CPC checkpoint; fitted temperature"}
    for condition in CONDITIONS:
        cells = []
        for metric, scale, precision in (("accuracy", 100, 2), ("aurc", 1, 5), ("ece_15_bins", 100, 3),
                                         ("nll", 1, 5), ("brier_score", 1, 5)):
            value = aggregate[condition][metric]
            cells.append(f"{scale * value['mean']:.{precision}f} ± {scale * value['std']:.{precision}f}")
        lines.append(f"| {condition} | {descriptions[condition]} | " + " | ".join(cells) + " |")
    lines += ["", f"The primary AURC change is **{delta['aurc']:+.6f}** and the mean accuracy change is **{100 * delta['accuracy']:+.3f} percentage points**. "
              "The predeclared numerical rule requires AURC to decrease and mean accuracy loss to be no greater than 0.5 percentage points.", "",
              "## What the metrics tell us", "", "| Metric | Plain-language interpretation | Better direction |", "|---|---|---|",
              "| Accuracy | How often the chosen class is correct | Higher |",
              "| AURC | Error remaining when we progressively accept predictions in confidence order | Lower |",
              "| Top-label ECE | Average gap between confidence and observed accuracy, using 15 fixed-width bins | Lower |",
              "| Class-wise ECE | Calibration of every class probability against whether that class occurs | Lower |",
              "| NLL / Brier | Overall quality of the full probability distribution | Lower |",
              "| Accuracy at 80% / 90% coverage | Accuracy after retaining the most confident 80% / 90% of predictions | Higher |", "",
              "Temperature scaling preserves each model's predicted classes. It can change confidence rankings across images, so AURC is recomputed after scaling.", "",
              "## Additional calibration and selection metrics", "",
              "| Condition | Equal-mass ECE (%) ↓ | L2 ECE (%) ↓ | Class-wise ECE (%) ↓ | Accuracy at 80% (%) ↑ | Accuracy at 90% (%) ↑ |",
              "|---|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        cells = [f"{100 * aggregate[condition][m]['mean']:.3f}" for m in
                 ("ece_equal_mass_15_bins", "ece_l2_15_bins", "classwise_ece_15_bins", "selective_accuracy_80", "selective_accuracy_90")]
        lines.append("| " + condition + " | " + " | ".join(cells) + " |")
    lines += ["", "## Paired changes and uncertainty", "",
              "Each entry is candidate minus control with a 95% percentile interval from 2,000 paired test-image bootstrap resamples. "
              "The same image indices are drawn for all three seed pairs; ranking and calibration statistics are recomputed. "
              "These intervals condition on the six fitted models and temperatures. They do not quantify uncertainty from retraining.", "",
              "| Comparison | Accuracy change (pp) | AURC change | ECE change (pp) | Class-wise ECE change (pp) | NLL change | Brier change |",
              "|---|---|---|---|---|---|---|"]
    for name, label in (("primary", "KD+CPC+TS − KD+TS (primary)"), ("raw_secondary", "KD+CPC − KD (secondary)")):
        cells = [_comparison_cells(report["comparisons"][name], m, scale) for m, scale in
                 (("accuracy", 100), ("aurc", 1), ("ece_15_bins", 100), ("classwise_ece_15_bins", 100), ("nll", 1), ("brier_score", 1))]
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines += ["", "The primary comparison within each seed is shown below; a pooled interval does not replace this view of seed variation.", "",
              "| Seed | Accuracy change (pp), 95% CI | AURC change, 95% CI | Class-wise ECE change (pp), 95% CI |",
              "|---:|---|---|---|"]
    for seed, comparison in zip(report['protocol']['seeds'], primary['per_seed']):
        cells = [_comparison_cells(comparison, m, scale) for m, scale in
                 (("accuracy",100), ("aurc",1), ("classwise_ece_15_bins",100))]
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")
    lines += ["", "## Individual checkpoints", "",
              "All six students were trained afresh. Each temperature was fitted independently on 5,000 validation examples by NLL, on the fixed grid 0.01–5.00 in steps of 0.01. "
              "All checkpoints and temperatures were recorded before constructing the test dataset.", "",
              "| Seed | Condition | Selected epoch | Temperature | Accuracy (%) | AURC | ECE (%) | NLL |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    csv_rows = []
    for row in report["rows"]:
        q = row["test"]
        lines.append(f"| {row['seed']} | {row['condition']} | {row['best_epoch']} | {row['parameters']['temperature']:.2f} | "
                     f"{100*q['accuracy']:.2f} | {q['aurc']:.6f} | {100*q['ece_15_bins']:.3f} | {q['nll']:.5f} |")
        csv_rows.append({"seed": row["seed"], "condition": row["condition"], "selected_epoch": row["best_epoch"],
                         "temperature": row["parameters"]["temperature"], "training_seconds": row["training_seconds"],
                         **{k: q[k] for k in SCALARS}})
    lines += ["", "![Risk–coverage and ECE comparison](figures/comparison.png)", "",
              "## Conditional perfect-calibration references", "",
              "Each evaluation condition has 2,000 synthetic label draws from its own fixed probability distributions. "
              "These show the finite-sample calibration statistics expected if those probabilities were correct. "
              "They are **not hard noise floors**, confidence intervals for true ECE, or values to subtract from the measured ECE.", "",
              "Detailed observed metrics, null means, central 95% simulation ranges and percentiles are saved for all twelve conditions:", ""]
    for row in report["rows"]:
        location = f"{row['run']}/evaluation/{row['variant']}"
        lines.append(f"- Seed {row['seed']}, {row['condition']}: [metrics and class diagnostics]({location}/metrics.json), [null references]({location}/calibration_null.json).")
    lines += ["", "<details>", "<summary>Observed calibration errors and conditional-null ranges for every seed and condition</summary>", "",
              "All error values below are percentages. A percentile locates the observed error within the simulated distribution; it is a descriptive diagnostic, not an adjusted significance test.", "",
              "| Seed | Condition | Metric | Observed (%) | Null mean (%) | Null central 95% (%) | Observed percentile |",
              "|---:|---|---|---:|---:|---|---:|"]
    for row in report['rows']:
        for metric, label in (("ece_15_bins","Equal-width ECE"), ("ece_equal_mass_15_bins","Equal-mass ECE"),
                              ("ece_l2_15_bins","L2 ECE"), ("classwise_ece_15_bins","Class-wise ECE")):
            value = row['calibration_null']['metrics'][metric]
            lo, hi = value['null_central_95']
            lines.append(f"| {row['seed']} | {row['condition']} | {label} | {100*value['observed']:.3f} | "
                         f"{100*value['null_mean']:.3f} | [{100*lo:.3f}, {100*hi:.3f}] | {value['observed_percentile']:.1f} |")
    lines += ["", "</details>", "", "## Training cost", "",
              "Times sum measured training epochs, including image loading, and exclude validation, checkpointing and final evaluation. "
              "TS reuses each trained checkpoint. Both training arms export the same student architecture; calibrated inference latency was not measured.", "",
              "| Seed | KD training (min) | CPC training (min) | CPC / KD |", "|---:|---:|---:|---:|"]
    for seed in report['protocol']['seeds']:
        control = next(r['training_seconds'] for r in report['rows'] if r['seed']==seed and r['condition']=='KD')
        cpc = next(r['training_seconds'] for r in report['rows'] if r['seed']==seed and r['condition']=='KD+CPC')
        lines.append(f"| {seed} | {control/60:.2f} | {cpc/60:.2f} | {cpc/control:.3f}× |")
    lines += ["", "## Protocol and limitations", "",
              "CPC adds binary-discrimination and binary-exclusion losses with fixed weights 0.1 and 0.1 from the first epoch. "
              "The original teacher stays frozen. Both arms retain the same initialization per seed, data recipe, 20-epoch budget, and validation-accuracy checkpoint rule. "
              "The first two augmented batches and initial model tensors are checked for equality within each seed; subsequent streams use the same deterministic RNG and loader recipe.", "",
              *["- " + note for note in report["limitations"]], "",
              "CPC loss specification: [Cheng and Vasconcelos, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Calibrating_Deep_Neural_Networks_by_Pairwise_Constraints_CVPR_2022_paper.html).", "",
              "Regenerate tables and figures from the signed artifacts without training, fitting or inference:", "", "```bash",
              f"python -m kd.cpc_reporting --study {directory}", "```", ""]
    (directory / "report.md").write_text("\n".join(lines))
    with (directory / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    _plots(directory, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", "--output", dest="output", default="runs/cifar10-cpc")
    args = parser.parse_args()
    render_report(args.output)


if __name__ == "__main__":
    main()
