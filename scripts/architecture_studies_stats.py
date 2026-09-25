"""Statistics for the architecture studies and the matched-balance DKD/RLD rerun.

Reads the Experiment 1 matrix plus the outputs of student_architecture_study.py,
distillation_ablation.py (matched-balance variants) and teacher_architecture_study.py, and
prints Markdown. Study-level tests treat each study as one unit (the mean over its seeds) with a
two-sided Wilcoxon signed-rank test. Nothing is trained or modified.

    python scripts/architecture_studies_stats.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from scipy.stats import spearmanr, wilcoxon

from baseline_comparison_table import DATASETS, PROFILES

STUDENTS = {"cifar_student": "cifar_student (Exp. 1)", "resnet8": "ResNet-8", "resnet20": "ResNet-20"}
METRICS = ("accuracy", "target_recall")
STOPS = {"Too few channels passed the configured localization guardrails.": "localization",
         "No repaired teacher strictly improved validation target recall within the 0.5-point "
         "accuracy guardrail.": "development selection"}


def _read(path: Path):
    return json.loads(path.read_text())


def _pp(value: float) -> str:
    return f"{100 * value:+.2f}"


def _p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def _label(job: dict) -> str:
    return f"{DATASETS.get(job['dataset'], job['dataset'])} {PROFILES.get(job['profile'], job['profile'])}"


def _test(differences: list[float]) -> dict:
    nonzero = [d for d in differences if d]
    return {"mean": statistics.mean(differences), "median": statistics.median(differences),
            "wins": sum(d > 0 for d in differences), "n": len(differences),
            "p": wilcoxon(nonzero).pvalue if len(nonzero) > 1 else 1.0}


def _merge(paths) -> dict:
    merged: dict = {}
    def deep(target, source):
        for key, value in source.items():
            if isinstance(value, dict) and not ("accuracy" in value and "target_recall" in value):
                deep(target.setdefault(key, {}), value)
            else:
                target[key] = value
    for path in paths:
        deep(merged, _read(path))
    return merged


def _exp1_students(matrix: Path, job: dict) -> dict:
    """Experiment 1's cifar_student pair, in the architecture study's format."""
    evaluation = matrix / job["dataset"] / job["profile"] / "studies" / "confirmatory" / "student_evaluation"
    pair = {"original": {}, "repaired": {}}
    for seed in sorted(evaluation.glob("seed*")):
        for label, variant in (("original", "baseline"), ("repaired", "candidate")):
            metrics = _read(seed / variant / "metrics.json")
            pair[label][seed.name[4:]] = {"accuracy": metrics["accuracy"],
                                          "target_recall": metrics["target_macro_recall"]}
    return pair


def _transfer(pair: dict) -> bool:
    """Experiment 1's rule: mean target recall +1 pp, two positive seeds, accuracy guardrail."""
    seeds = list(pair["original"])
    recall = [pair["repaired"][s]["target_recall"] - pair["original"][s]["target_recall"] for s in seeds]
    accuracy = [pair["repaired"][s]["accuracy"] - pair["original"][s]["accuracy"] for s in seeds]
    return (statistics.mean(recall) >= 0.01 and sum(r > 0 for r in recall) >= 2
            and statistics.mean(accuracy) >= -0.005)


def student_section(matrix: Path, jobs: list[dict], root: Path) -> list[str]:
    report = _merge(sorted(root.glob("*/report.json")))
    pairs = {s: {} for s in STUDENTS}
    for job in jobs:
        pairs["cifar_student"][job["id"]] = _exp1_students(matrix, job)
        for student in ("resnet8", "resnet20"):
            pairs[student][job["id"]] = report[job["id"]][student]
    delta = {s: {j: {m: statistics.mean(p["repaired"][k][m] - p["original"][k][m] for k in p["original"])
                     for m in METRICS} for j, p in pairs[s].items()} for s in STUDENTS}
    n = len(jobs)
    lines = ["## Student architectures (Study B)", "",
             "Repaired-teacher student minus original-teacher student of the same architecture and seed. "
             f"{n} studies; Wilcoxon signed-rank across studies.", "",
             "| Student | Δ accuracy mean (median) | Higher | p | Δ target recall mean (median) | Higher | p | Transfer rule passed |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for student, label in STUDENTS.items():
        a = _test([delta[student][j["id"]]["accuracy"] for j in jobs])
        r = _test([delta[student][j["id"]]["target_recall"] for j in jobs])
        passed = sum(_transfer(pairs[student][j["id"]]) for j in jobs)
        lines.append(f"| {label} | {_pp(a['mean'])} ({_pp(a['median'])}) | {a['wins']}/{n} | {_p(a['p'])} | "
                     f"{_pp(r['mean'])} ({_pp(r['median'])}) | {r['wins']}/{n} | {_p(r['p'])} | {passed}/{n} |")
    lines += ["", "Do the same studies transfer to every student? Spearman correlation of target-recall gains across studies:", "",
              "| Pair | ρ | p |", "|---|---:|---:|"]
    for left, right in (("cifar_student", "resnet8"), ("cifar_student", "resnet20"), ("resnet8", "resnet20")):
        rho, p = spearmanr([delta[left][j["id"]]["target_recall"] for j in jobs],
                           [delta[right][j["id"]]["target_recall"] for j in jobs])
        lines.append(f"| {STUDENTS[left]} vs {STUDENTS[right]} | {rho:+.2f} | {_p(p)} |")
    lines += ["", "Target-recall gain by imbalance profile (pp):", "",
              "| Profile | Studies | " + " | ".join(STUDENTS.values()) + " |", "|---|---:|" + "---:|" * len(STUDENTS)]
    for profile, label in PROFILES.items():
        group = [j for j in jobs if j["profile"] == profile]
        if group:
            lines.append(f"| {label} | {len(group)} | " + " | ".join(
                _pp(statistics.mean(delta[s][j["id"]]["target_recall"] for j in group)) for s in STUDENTS) + " |")
    lines += ["", "Per study, target-recall gain (pp):", "",
              "| Study | " + " | ".join(STUDENTS.values()) + " |", "|---|" + "---:|" * len(STUDENTS)]
    for job in jobs:
        lines.append(f"| {_label(job)} | " + " | ".join(
            _pp(delta[s][job["id"]]["target_recall"]) + (" ✓" if _transfer(pairs[s][job["id"]]) else "")
            for s in STUDENTS) + " |")
    return lines + ["", "✓ marks studies that pass Experiment 1's transfer rule for that student.", ""]


def matched_section(matrix: Path, jobs: list[dict], root: Path) -> list[str]:
    report = _merge(sorted(root.glob("*/report.json")))
    columns = {"kd_original": "KD, original teacher", "kd_repaired": "KD, repaired teacher",
               "dkd": "DKD, published", "dkd_w0.15": "DKD, weight 0.15", "rld": "RLD, published",
               "rld_w0.2": "RLD, weight 0.2", "loca": "LoCa"}
    values = {c: {} for c in columns}
    for job in jobs:
        profile = matrix / job["dataset"] / job["profile"]
        pair = _exp1_students(matrix, job)
        values["kd_original"][job["id"]], values["kd_repaired"][job["id"]] = pair["original"], pair["repaired"]
        for method in ("dkd", "rld", "loca"):
            comparison = _read(profile / "baselines" / "distillation" / method / "comparison.json")
            values[method][job["id"]] = {str(row["seed"]): {"accuracy": row["metrics"]["accuracy"],
                                                            "target_recall": row["metrics"]["target_macro_recall"]}
                                         for row in comparison["seeds"]}
        for variant in ("dkd_w0.15", "rld_w0.2"):
            values[variant][job["id"]] = report[job["id"]][variant]
    mean = {c: {j: {m: statistics.mean(v[m] for v in seeds.values()) for m in METRICS}
                for j, seeds in values[c].items()} for c in columns}
    n = len(jobs)
    lines = ["## DKD and RLD at classical KD's loss balance", "",
             f"Means over {n} studies (each the mean of 3 seeds), in percent.", "",
             "| Student | Accuracy | Target recall |", "|---|---:|---:|"]
    for c, label in columns.items():
        lines.append(f"| {label} | {100 * statistics.mean(mean[c][j['id']]['accuracy'] for j in jobs):.2f} | "
                     f"{100 * statistics.mean(mean[c][j['id']]['target_recall'] for j in jobs):.2f} |")
    lines += ["", "Paired across studies; positive differences favour the first-named student.", "",
              "| Comparison | Δ accuracy | Higher | p | Δ target recall | Higher | p |", "|---|---:|---:|---:|---:|---:|---:|"]
    for left, right in (("kd_repaired", "dkd_w0.15"), ("kd_repaired", "rld_w0.2"), ("kd_repaired", "kd_original"),
                        ("dkd_w0.15", "kd_original"), ("rld_w0.2", "kd_original"),
                        ("dkd_w0.15", "dkd"), ("rld_w0.2", "rld")):
        a = _test([mean[left][j["id"]]["accuracy"] - mean[right][j["id"]]["accuracy"] for j in jobs])
        r = _test([mean[left][j["id"]]["target_recall"] - mean[right][j["id"]]["target_recall"] for j in jobs])
        lines.append(f"| {columns[left]} vs {columns[right]} | {_pp(a['mean'])} | {a['wins']}/{n} | {_p(a['p'])} | "
                     f"{_pp(r['mean'])} | {r['wins']}/{n} | {_p(r['p'])} |")
    return lines + [""]


def teacher_section(matrix: Path, root: Path) -> list[str]:
    exp1 = {j["id"]: j for j in _read(matrix / "matrix_summary.json")["jobs"]}
    rows, counts = [], {}
    for path in sorted(root.glob("*/studies/confirmatory/comparison.json")):
        name = path.parts[-4]
        job_id, teacher = name.rsplit("-", 1)
        c = _read(path)
        base = exp1[job_id]
        before = "repair selected" if base["status"] == "complete" else STOPS.get(base.get("reason"), base["status"])
        if base["status"] == "complete":
            before += ", transferred" if base.get("success") else ", no transfer"
        if c.get("status") == "complete":
            t, a = c["teacher"], c["aggregate"]
            outcome = "transferred" if c["success"]["passed"] else "no transfer"
            detail = (f"{c['teacher_selection']['selected']['budget']} ch; teacher recall "
                      f"{100 * t['baseline']['target_macro_recall']:.2f}→{100 * t['candidate']['target_macro_recall']:.2f}; "
                      f"student Δrecall {_pp(a['target_recall_delta']['mean'])}")
            stage = f"repair selected, {outcome}"
        else:
            stage, detail = STOPS.get(c.get("reason"), c.get("status")), ""
        key = {"r18": "ResNet-18", "r34": "ResNet-34"}[teacher]
        counts.setdefault(key, {}).setdefault(stage.split(",")[0], 0)
        counts[key][stage.split(",")[0]] += 1
        counts[key].setdefault("transferred", 0)
        counts[key]["transferred"] += stage.endswith("transferred") and not stage.endswith("no transfer")
        rows.append(f"| {job_id} | {key} | {before} | {stage} | {detail} |")
    lines = ["## Teacher architectures (Study A)", "",
             "| Job | Teacher | cifar_teacher (Exp. 1) | This teacher | Detail |", "|---|---|---|---|---|", *rows, "",
             "| Teacher | Stopped at localization | Stopped at development selection | Repair selected | Transferred |",
             "|---|---:|---:|---:|---:|"]
    for key, c in counts.items():
        lines.append(f"| {key} | {c.get('localization', 0)} | {c.get('development selection', 0)} | "
                     f"{c.get('repair selected', 0)} | {c.get('transferred', 0)} |")
    return lines + [""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", type=Path, default=Path("runs/experiment1-multidataset"))
    parser.add_argument("--students", type=Path, default=Path("runs/student-architecture"))
    parser.add_argument("--matched", type=Path, default=Path("runs/matched-balance"))
    parser.add_argument("--teachers", type=Path, default=Path("runs/teacher-architecture"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    jobs = [j for j in _read(args.matrix / "matrix_summary.json")["jobs"] if j["status"] == "complete"]
    order = {p: i for i, p in enumerate(PROFILES)}
    jobs.sort(key=lambda j: (DATASETS.get(j["dataset"], j["dataset"]).lower(), order[j["profile"]]))
    lines = ["# Architecture studies and matched-balance baselines", "",
             "Generated by `scripts/architecture_studies_stats.py`.", "",
             *teacher_section(args.matrix, args.teachers), *student_section(args.matrix, jobs, args.students),
             *matched_section(args.matrix, jobs, args.matched)]
    text = "\n".join(lines)
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
