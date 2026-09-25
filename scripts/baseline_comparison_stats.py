"""Paired statistics for Experiment 1 students against the DKD, RLD and LoCa baselines.

Reads the same matrix as baseline_comparison_table.py and prints Markdown tables. The unit
of inference is the study: each of the 23 studies contributes one paired difference, the
mean over its three student seeds, and differences are tested with a two-sided Wilcoxon
signed-rank test across studies. Seed-level exact McNemar tests, recorded by each
baseline run against both KD students, are counted as supporting, unadjusted evidence.
Nothing is trained or modified.

    python scripts/baseline_comparison_stats.py --matrix runs/experiment1-multidataset
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from scipy.stats import wilcoxon

from baseline_comparison_table import BASELINES, COLUMNS, DATASETS, PROFILES, REFERENCES

METRICS = {"accuracy": "Test accuracy", "target_macro_recall": "Target-class recall",
           "ece_15_bins": "ECE (15 bins)"}
VERSUS = {"kd_original": "kd_original_teacher", "kd_repaired": "kd_repaired_teacher"}


def _read(path: Path):
    return json.loads(path.read_text())


# Weights on (cross-entropy, teacher term) in DistillationObjective after warm-up: KD and LoCa
# split one unit between them; DKD and RLD add the teacher term at full strength.
LOSS_WEIGHTS = {"kd_original": (0.5, 0.5), "dkd": (1.0, 1.0), "rld": (1.0, 1.0), "loca": (0.5, 0.5)}


def load(matrix: Path) -> list[dict]:
    """Per study: each column's per-seed metrics, and each baseline's seed-level tests."""
    studies = []
    for job in _read(matrix / "matrix_summary.json")["jobs"]:
        if job["status"] != "complete":
            continue
        profile = matrix / job["dataset"] / job["profile"]
        evaluation = profile / "studies" / "confirmatory" / "student_evaluation"
        seeds = sorted(evaluation.glob("seed*"), key=lambda path: int(path.name[4:]))
        values, tests = {}, {}
        for column, variant in REFERENCES.items():
            runs = [_read(seed / variant / "metrics.json") for seed in seeds]
            values[column] = {metric: [run[metric] for run in runs] for metric in METRICS}
        for method in BASELINES:
            comparison = _read(profile / "baselines" / "distillation" / method / "comparison.json")
            if comparison.get("status") != "complete":
                raise ValueError(f"{job['id']} {method} is not scored yet")
            values[method] = {metric: [row["metrics"][metric] for row in comparison["seeds"]]
                              for metric in METRICS}
            tests[method] = {reference: [row["versus"][name] for row in comparison["seeds"]]
                             for reference, name in VERSUS.items()}
        studies.append({"job": job, "values": values, "tests": tests,
                        "diagnostics": _diagnostics(job, profile, seeds)})
    return studies


def _diagnostics(job: dict, profile: Path, seeds: list[Path]) -> dict | None:
    """Loss balance, epoch-to-epoch stability and tail/head recall, from files the runs wrote.

    The classical-KD and teacher histories come from the Experiment 1 archive; a study whose
    archived histories are not present gets None.
    """
    target = _read(profile / "studies" / "confirmatory" / "comparison.json")["target_classes"]
    result = {"ratio": {}, "swing": {}, "tail": {}, "head": {}}
    for column, (ce_weight, response_weight) in LOSS_WEIGHTS.items():
        ratios, swings, tails, heads = [], [], [], []
        for seed in seeds:
            number = seed.name[4:]
            if column == "kd_original":
                history_path = (profile / "baselines" / "confirmatory"
                                / f"kd_{job['id']}_seed{number}" / "history.json")
                metrics = seed / "baseline" / "metrics.json"
            else:
                output = profile / "baselines" / "distillation" / column
                history_path = output / f"{column}_{job['id']}_seed{number}" / "history.json"
                metrics = output / "evaluation" / seed.name / "metrics.json"
            if not history_path.is_file():
                return None
            history = _read(history_path)[5:]  # after the 5 warm-up epochs
            ratios.append(statistics.median(response_weight * epoch["train"]["response"]
                                            / (ce_weight * epoch["train"]["ce"])
                                            for epoch in history if epoch["train"]["ce"] > 0))
            accuracy = [epoch["val"]["accuracy"] for epoch in history]
            late = accuracy[len(accuracy) // 2:]
            swings.append(_mean([abs(b - a) for a, b in zip(late, late[1:])]))
            recall = [row["recall"] for row in _read(metrics)["per_class"]]
            tails.append(_mean([recall[i] for i in target]))
            heads.append(_mean([recall[i] for i in range(len(recall)) if i not in target] or [0.0]))
        result["ratio"][column], result["swing"][column] = ratios, swings
        result["tail"][column], result["head"][column] = _mean(tails), _mean(heads)
    teacher = profile / "baselines" / "confirmatory" / f"teacher_{job['id']}" / "history.json"
    result["teacher_train_accuracy"] = (_read(teacher)[-1]["train"]["accuracy"]
                                        if teacher.is_file() else None)
    return result


def _mean(values: list[float]) -> float:
    return statistics.mean(values)


def _pp(value: float) -> str:
    return f"{100 * value:+.2f}"


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _paired(studies, left: str, right: str, metric: str) -> dict:
    """Study-level differences left - right, with wins and a Wilcoxon signed-rank test."""
    differences = [_mean(study["values"][left][metric]) - _mean(study["values"][right][metric])
                   for study in studies]
    nonzero = [difference for difference in differences if difference]
    return {"mean": _mean(differences), "median": statistics.median(differences),
            "wins": sum(difference > 0 for difference in differences), "n": len(differences),
            "p": wilcoxon(nonzero).pvalue if len(nonzero) > 1 else 1.0}


def render(studies: list[dict]) -> str:
    n = len(studies)
    lines = ["# Distillation baselines: paired statistics", "",
             "Generated by `scripts/baseline_comparison_stats.py`. "
             f"Studies: {n}. Differences are percentage points; each study contributes the mean "
             "over its three seeds. p: two-sided Wilcoxon signed-rank test across studies.", ""]

    lines += ["## Method means", "",
              "| Method | Test accuracy (%) | Target recall (%) | ECE (%) | Median seed SD, accuracy (pp) |",
              "|---|---:|---:|---:|---:|"]
    for column, label in COLUMNS.items():
        means = {metric: _mean([_mean(study["values"][column][metric]) for study in studies])
                 for metric in METRICS}
        spread = statistics.median(statistics.stdev(study["values"][column]["accuracy"])
                                   for study in studies)
        lines.append(f"| {label} | {100 * means['accuracy']:.2f} | "
                     f"{100 * means['target_macro_recall']:.2f} | {100 * means['ece_15_bins']:.2f} | "
                     f"{100 * spread:.2f} |")

    lines += ["", "## KD from the repaired teacher against each alternative", "",
              "Positive differences favour KD from the repaired teacher.", "",
              "| Compared with | Δ accuracy mean (median) | Wins | p | "
              "Δ target recall mean (median) | Wins | p |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for column in ("kd_original", *BASELINES):
        accuracy = _paired(studies, "kd_repaired", column, "accuracy")
        recall = _paired(studies, "kd_repaired", column, "target_macro_recall")
        lines.append(
            f"| {COLUMNS[column]} | {_pp(accuracy['mean'])} ({_pp(accuracy['median'])}) | "
            f"{accuracy['wins']}/{n} | {_p(accuracy['p'])} | "
            f"{_pp(recall['mean'])} ({_pp(recall['median'])}) | {recall['wins']}/{n} | {_p(recall['p'])} |")

    lines += ["", "## Each baseline loss against classical KD from the same original teacher", "",
              "Positive differences favour the baseline loss.", "",
              "| Baseline | Δ accuracy mean (median) | Wins | p | Δ target recall mean (median) | Wins | p |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for method in BASELINES:
        accuracy = _paired(studies, method, "kd_original", "accuracy")
        recall = _paired(studies, method, "kd_original", "target_macro_recall")
        lines.append(
            f"| {COLUMNS[method]} | {_pp(accuracy['mean'])} ({_pp(accuracy['median'])}) | "
            f"{accuracy['wins']}/{n} | {_p(accuracy['p'])} | "
            f"{_pp(recall['mean'])} ({_pp(recall['median'])}) | {recall['wins']}/{n} | {_p(recall['p'])} |")

    lines += ["", "## Target-recall gain of KD from the repaired teacher, by imbalance profile", "",
              "| Profile | Studies | vs KD, original | vs DKD | vs RLD | vs LoCa |",
              "|---|---:|---:|---:|---:|---:|"]
    for profile, label in PROFILES.items():
        group = [study for study in studies if study["job"]["profile"] == profile]
        if not group:
            continue
        cells = [_pp(_paired(group, "kd_repaired", column, "target_macro_recall")["mean"])
                 for column in ("kd_original", *BASELINES)]
        lines.append(f"| {label} | {len(group)} | " + " | ".join(cells) + " |")

    lines += ["", "## Seed-level exact McNemar tests (unadjusted, p < 0.05)", "",
              "Each baseline student against the KD student of the same seed; "
              f"{3 * n} pairs per row. Counts say which student made more correct predictions.", "",
              "| Baseline | vs KD, original: baseline better | KD better | n.s. | "
              "vs KD, repaired: baseline better | KD better | n.s. |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for method in BASELINES:
        cells = []
        for reference in VERSUS:
            rows = [row for study in studies for row in study["tests"][method][reference]]
            significant = [row for row in rows if row["mcnemar_exact_two_sided_p"] < 0.05]
            better = sum(row["accuracy_delta"] > 0 for row in significant)
            cells += [str(better), str(len(significant) - better), str(len(rows) - len(significant))]
        lines.append(f"| {COLUMNS[method]} | " + " | ".join(cells) + " |")

    lines += ["", "## Studies where a baseline loses more than 5 pp of accuracy to KD, original", "",
              "| Study | Baseline | Accuracy (%) | KD, original (%) | Seed SD (pp) | Target recall (%) |",
              "|---|---|---:|---:|---:|---:|"]
    for study in studies:
        reference = _mean(study["values"]["kd_original"]["accuracy"])
        for method in BASELINES:
            accuracy = study["values"][method]["accuracy"]
            if _mean(accuracy) < reference - 0.05:
                name = (f"{DATASETS.get(study['job']['dataset'], study['job']['dataset'])} "
                        f"{PROFILES[study['job']['profile']]}")
                lines.append(
                    f"| {name} | {COLUMNS[method]} | {100 * _mean(accuracy):.2f} | {100 * reference:.2f} | "
                    f"{100 * statistics.stdev(accuracy):.2f} | "
                    f"{100 * _mean(study['values'][method]['target_macro_recall']):.2f} |")
    diagnosed = [study for study in studies if study["diagnostics"]]
    if diagnosed:
        lines += ["", "## Why the losses differ from classical KD", "",
                  f"From the training histories of {len(diagnosed)} studies. Loss balance is the "
                  "weighted teacher term divided by the weighted cross-entropy after warm-up; swing "
                  "is the mean change in validation accuracy between consecutive epochs over the "
                  "second half of training. Medians over studies and seeds.", "",
                  "| Method | Teacher term ÷ cross-entropy | Validation swing (pp) | Largest swing (pp) |",
                  "|---|---:|---:|---:|"]
        for column in LOSS_WEIGHTS:
            ratios = [r for s in diagnosed for r in s["diagnostics"]["ratio"][column]]
            swings = [w for s in diagnosed for w in s["diagnostics"]["swing"][column]]
            lines.append(f"| {COLUMNS[column]} | {statistics.median(ratios):.1f}× | "
                         f"{100 * statistics.median(swings):.2f} | {100 * max(swings):.2f} |")
        tailed = [s for s in diagnosed if s["job"]["profile"] != "balanced"]
        lines += ["", f"Recall change against classical KD from the same teacher, {len(tailed)} "
                      "long-tailed studies (pp; tail = target classes, head = the rest):", "",
                  "| Method | Tail | Head | Studies where tail fell more |", "|---|---:|---:|---:|"]
        for method in BASELINES:
            tail = [s["diagnostics"]["tail"][method] - s["diagnostics"]["tail"]["kd_original"]
                    for s in tailed]
            head = [s["diagnostics"]["head"][method] - s["diagnostics"]["head"]["kd_original"]
                    for s in tailed]
            lines.append(f"| {COLUMNS[method]} | {_pp(_mean(tail))} | {_pp(_mean(head))} | "
                         f"{sum(t < h for t, h in zip(tail, head))}/{len(tailed)} |")
        accuracies = sorted((s["diagnostics"]["teacher_train_accuracy"], s["job"]["id"])
                            for s in diagnosed if s["diagnostics"]["teacher_train_accuracy"] is not None)
        if accuracies:
            values = [value for value, _ in accuracies]
            lines += ["", "Teacher accuracy on its own augmented training images in its final "
                          f"epoch: median {100 * statistics.median(values):.1f}%, range "
                          f"{100 * values[0]:.1f}% to {100 * values[-1]:.1f}% (lowest: "
                          + ", ".join(f"{name} {100 * value:.1f}%" for value, name in accuracies[:3])
                          + "). LoCa changes a teacher distribution only where the teacher's top "
                          "class is wrong."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", type=Path, default=Path("runs/experiment1-multidataset"))
    parser.add_argument("--output", type=Path, help="Write Markdown here instead of stdout")
    args = parser.parse_args()
    text = render(load(args.matrix.resolve()))
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
