"""Compare channel-selection arms on the repaired teacher.

One row per setting: each randomized arm contributes the mean over its draws, so every
arm offers one number per setting and the comparison stays paired. Differences are
screened minus control, in percentage points, tested with a two-sided Wilcoxon
signed-rank test over settings with exact zeros dropped, as the other studies do.

    python3 scripts/channel_selection_stats.py --predeclared-margin 0.5

The decision rule belongs in the note before the numbers exist. Pass the margin that
was predeclared and this prints the verdict against it rather than inviting a threshold
chosen after the fact.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

METRIC = "target_macro_recall"
CONTROLS = ("random", "stage_matched_random", "antiscreened")


def arm_metric(directory: Path, metric: str) -> float | None:
    """The repaired teacher's validation metric for one arm run, as a percentage."""
    candidates = directory / "candidate_results.json"
    if not candidates.is_file():
        return None
    rows = json.loads(candidates.read_text())
    if not rows:
        return None
    # The budget and learning rate are pinned per arm, so one candidate is expected.
    values = [row["validation"][metric] for row in rows if metric in row["validation"]]
    return 100.0 * max(values) if values else None


def baseline_metric(source: Path, metric: str) -> float | None:
    selection = source / "teacher_selection.json"
    if not selection.is_file():
        return None
    value = json.loads(selection.read_text()).get("baseline_validation", {}).get(metric)
    return None if value is None else 100.0 * value


def collect(runs_path: Path, metric: str) -> tuple[dict, list[str]]:
    record = json.loads(runs_path.read_text())
    per_setting: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    baselines: dict[str, float] = {}
    missing = []
    for row in record["results"] + record.get("failures", []):
        setting = f"{row['dataset']}/{row['profile']}"
        value = arm_metric(Path(row["output"]), metric)
        if value is None:
            missing.append(f"{setting} {row['arm']} draw{row['draw']}")
            continue
        per_setting[setting][row["arm"]].append(value)
        baselines.setdefault(setting, baseline_metric(Path(row["source"]), metric) or float("nan"))
    table = {setting: {arm: statistics.fmean(values) for arm, values in arms.items()}
             for setting, arms in per_setting.items()}
    return {"table": table, "baselines": baselines}, missing


def compare(table: dict, screened: str, control: str) -> dict | None:
    pairs = [(setting, arms[screened], arms[control]) for setting, arms in sorted(table.items())
             if screened in arms and control in arms]
    differences = [screened_value - control_value for _, screened_value, control_value in pairs]
    nonzero = [value for value in differences if value != 0.0]
    if len(nonzero) < 2:
        return None
    result = stats.wilcoxon(nonzero, alternative="two-sided")
    return {"control": control, "settings": len(pairs), "nonzero": len(nonzero),
            "mean": statistics.fmean(differences), "median": statistics.median(differences),
            "screened_higher": sum(value > 0 for value in differences),
            "p_value": float(result.pvalue)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path,
                        default=Path("runs/channel-selection-ablation/ablation_runs.json"))
    parser.add_argument("--metric", default=METRIC)
    parser.add_argument("--predeclared-margin", type=float,
                        help="Median advantage in pp that was declared before running")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    if not args.runs.is_file():
        print(f"No run record at {args.runs}; run the ablation first", file=sys.stderr)
        return 1
    collected, missing = collect(args.runs, args.metric)
    table = collected["table"]
    if not table:
        print("No arm produced a candidate result", file=sys.stderr)
        return 1

    arms = sorted({arm for arms in table.values() for arm in arms})
    print(f"Metric: {args.metric} (percentage points)   settings: {len(table)}\n")
    header = f"{'setting':26}" + "".join(f"{arm:>22}" for arm in arms)
    print(header)
    print("-" * len(header))
    for setting, values in sorted(table.items()):
        row = f"{setting:26}"
        for arm in arms:
            row += f"{values[arm]:22.2f}" if arm in values else f"{'-':>22}"
        print(row)

    print("\nScreened minus control, paired over settings:\n")
    print(f"{'control':22}{'n':>4}{'mean':>9}{'median':>9}{'higher':>8}{'p':>10}")
    verdicts = []
    for control in CONTROLS:
        if control not in arms:
            continue
        outcome = compare(table, "screened", control)
        if outcome is None:
            print(f"{control:22}{'too few nonzero differences to test':>40}")
            continue
        print(f"{control:22}{outcome['settings']:>4}{outcome['mean']:>9.2f}"
              f"{outcome['median']:>9.2f}{outcome['screened_higher']:>4}/{outcome['settings']:<3}"
              f"{outcome['p_value']:>10.4f}")
        verdicts.append(outcome)

    if args.predeclared_margin is not None:
        print(f"\nAgainst the predeclared rule (p < {args.alpha}, "
              f"median >= {args.predeclared_margin} pp):")
        for outcome in verdicts:
            passed = outcome["p_value"] < args.alpha and outcome["median"] >= args.predeclared_margin
            print(f"  vs {outcome['control']:22} "
                  f"{'screening is load-bearing' if passed else 'not supported'}")
        print("\nIf screening fails against stage_matched_random, the claim is stage-level\n"
              "localization plus validation selection, and the title should say so.")

    if missing:
        print(f"\n{len(missing)} arm runs produced no candidate result:")
        for label in missing[:10]:
            print(f"    {label}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
