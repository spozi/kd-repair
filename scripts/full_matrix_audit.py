"""Report which parts of the full 5-pairing, 52-setting matrix are finished.

Counts artifacts only (finished comparisons and student summaries), never their results, so
the report can be shared while the results stay private. Sections:

1. Teacher repair decisions for CNN-T, ResNet-18 and ResNet-34, fixed and scaled budget arms.
2. Matched students for every pairing: original-teacher students in every decided setting and
   repaired-teacher students where a repair was selected.
3. Alternative distillation losses for every pairing, counting CNN-T -> CNN-S runs that
   alternative_losses.py --link-existing can reuse.

    python scripts/full_matrix_audit.py
    python scripts/full_matrix_audit.py --pending   # also list what is left, per setting
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from alternative_losses import VARIANTS  # noqa: E402
from pairing_cells import PAIRINGS  # noqa: E402

FINISHED = {"complete", "no_repair_selected"}
SEEDS = (42, 43, 44)
MATCHED = {"dkd": 0.15, "rld": 0.2}  # the CNN-T -> CNN-S lower response weights
ARMS = {"CNN-T": ("cifar_teacher", None), "ResNet-18": ("cifar_resnet18", "teacher-architecture"),
        "ResNet-34": ("cifar_resnet34", "teacher-architecture"),
        "ResNet-18 scaled": ("cifar_resnet18", "teacher-architecture-scaled"),
        "ResNet-34 scaled": ("cifar_resnet34", "teacher-architecture-scaled")}
SUFFIX = {"cifar_resnet18": "r18", "cifar_resnet34": "r34"}


def _read(path: Path):
    return json.loads(path.read_text())


def _done(path: Path) -> bool:
    return (path / "summary.json").is_file()


def _root(runs: Path, matrix: Path, job: dict, teacher: str, arm_dir: str | None) -> tuple[Path, str]:
    if arm_dir is None:
        return matrix / job["dataset"] / job["profile"], job["id"]
    name = f"{job['id']}-{SUFFIX[teacher]}"
    return runs / arm_dir / name, name


def _decision(root: Path) -> str:
    comparison = root / "studies" / "confirmatory" / "comparison.json"
    if comparison.is_file() and _read(comparison).get("status") in FINISHED:
        return _read(comparison)["status"]
    return "started" if root.exists() else "not_started"


def audit(runs: Path, matrix: Path) -> dict:
    jobs = _read(matrix / "matrix_summary.json")["jobs"]
    decisions = {arm: {job["id"]: _decision(_root(runs, matrix, job, teacher, arm_dir)[0]) for job in jobs}
                 for arm, (teacher, arm_dir) in ARMS.items()}

    students = {}
    for pairing, (teacher, student) in PAIRINGS.items():
        arm = {"cifar_teacher": "CNN-T", "cifar_resnet18": "ResNet-18", "cifar_resnet34": "ResNet-34"}[teacher]
        rows = {}
        for job in jobs:
            status = decisions[arm][job["id"]]
            root, name = _root(runs, matrix, job, teacher, ARMS[arm][1])
            if student == "cifar_student":
                original = [root / "baselines" / "confirmatory" / f"kd_{name}_seed{s}" for s in SEEDS]
                repaired = [root / "studies" / "confirmatory" / f"kd_repaired_{name}_seed{s}" for s in SEEDS]
            else:
                def find(kind, seed, job=job["id"]):
                    hits = sorted((runs / "student-architecture").glob(f"*/{job}/{student}_{kind}_{job}_seed{seed}"))
                    return hits[0] if hits else runs / "student-architecture" / "_missing" / job
                original = [find("original", s) for s in SEEDS]
                repaired = [find("repaired", s) for s in SEEDS]
            count = sum(map(_done, repaired))
            if student == "cifar_student" and status == "complete":
                # The study trains these itself and scores every seed before writing a complete
                # comparison, so the comparison stands in for student folders not copied locally.
                comparison = _read(root / "studies" / "confirmatory" / "comparison.json")
                count = max(count, len({row["seed"] for row in comparison["students"]} & set(SEEDS)))
            rows[job["id"]] = {"decision": status, "original": sum(map(_done, original)),
                               "repaired": count if status == "complete" else None}
        students[pairing] = rows

    losses = {}
    for pairing in PAIRINGS:
        teacher = PAIRINGS[pairing][0]
        arm = {"cifar_teacher": "CNN-T", "cifar_resnet18": "ResNet-18", "cifar_resnet34": "ResNet-34"}[teacher]
        rows = {}
        for job in jobs:
            output = runs / "alternative-losses" / pairing / job["id"]
            done = reusable = 0
            for variant in VARIANTS:
                for seed in SEEDS:
                    if _done(output / f"{variant}_{job['id']}_seed{seed}"):
                        done += 1
                    elif pairing == "cnn-cnn":
                        reusable += _reusable(runs, matrix, job, variant, seed)
            rows[job["id"]] = {"decision": decisions[arm][job["id"]], "done": done, "reusable": reusable}
        losses[pairing] = rows
    return {"decisions": decisions, "students": students, "losses": losses}


def _reusable(runs: Path, matrix: Path, job: dict, variant: str, seed: int) -> bool:
    method = variant.removesuffix("_matched")
    if variant == method:
        return _done(matrix / job["dataset"] / job["profile"] / "baselines" / "distillation" / method
                     / f"{method}_{job['id']}_seed{seed}")
    return any(_done(path) for path in (runs / "matched-balance").glob(
        f"*/{job['id']}/{method}_w{MATCHED[method]:g}_{job['id']}_seed{seed}"))


def render(result: dict, pending: bool) -> str:
    total = len(next(iter(result["decisions"].values())))
    per_cell = len(VARIANTS) * len(SEEDS)
    lines = ["## Teacher repair decisions", "",
             "| Teacher arm | Repair selected | No repair | Started, unfinished | Not started |",
             "|---|---:|---:|---:|---:|"]
    for arm, rows in result["decisions"].items():
        counts = {key: sum(value == key for value in rows.values())
                  for key in ("complete", "no_repair_selected", "started", "not_started")}
        lines.append(f"| {arm} | {counts['complete']} | {counts['no_repair_selected']} | "
                     f"{counts['started']} | {counts['not_started']} |")
    lines += ["", "## Matched students (3 seeds per condition)", "",
              "| Pairing | Settings decided | Original students | Repaired students | Settings finished |",
              "|---|---:|---:|---:|---:|"]
    for pairing, rows in result["students"].items():
        decided = [r for r in rows.values() if r["decision"] in FINISHED]
        repaired_needed = 3 * sum(r["decision"] == "complete" for r in decided)
        finished = sum(r["original"] == 3 and r["repaired"] in (None, 3) for r in decided)
        lines.append(f"| {pairing} | {len(decided)}/{total} | {sum(r['original'] for r in decided)}/"
                     f"{3 * len(decided)} | {sum(r['repaired'] or 0 for r in decided)}/{repaired_needed} | "
                     f"{finished}/{total} |")
    lines += ["", f"## Alternative losses ({len(VARIANTS)} variants x 3 seeds = {per_cell} per setting)", "",
              "| Pairing | Settings ready | Trained | Reusable | Still to train in ready settings | "
              "Waiting on a teacher decision |",
              "|---|---:|---:|---:|---:|---:|"]
    for pairing, rows in result["losses"].items():
        ready = [r for r in rows.values() if r["decision"] in FINISHED]
        done = sum(r["done"] for r in ready)
        reusable = sum(r["reusable"] for r in ready)
        lines.append(f"| {pairing} | {len(ready)}/{total} | {done} | {reusable} | "
                     f"{per_cell * len(ready) - done - reusable} | {per_cell * (total - len(ready))} |")
    if pending:
        lines += ["", "## Pending, per setting", ""]
        for arm, rows in result["decisions"].items():
            left = [job for job, value in rows.items() if value not in FINISHED]
            if left:
                lines.append(f"- {arm} teacher decisions ({len(left)}): {', '.join(left)}")
        for pairing, rows in result["students"].items():
            left = [job for job, r in rows.items() if r["decision"] in FINISHED
                    and (r["original"] < 3 or r["repaired"] not in (None, 3))]
            if left:
                lines.append(f"- {pairing} students ({len(left)}): {', '.join(left)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--matrix", type=Path, default=Path("runs/experiment1-multidataset"))
    parser.add_argument("--pending", action="store_true", help="List the unfinished settings")
    parser.add_argument("--json", type=Path, default=None, help="Also write the counts as JSON")
    args = parser.parse_args()
    result = audit(args.runs, args.matrix)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(render(result, args.pending), end="")


if __name__ == "__main__":
    main()
