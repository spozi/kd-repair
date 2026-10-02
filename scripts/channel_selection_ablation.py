"""Phase 1 of the channel-selection ablation: repair controls, no students.

Tests whether causal screening is what helps, or whether any set of channels of the
same size would do. For each setting that produced a repair, the study runs again with
``channel_selection`` set to each control mode, holding the budget, learning rate and
guardrails at the values the original run selected.

Nothing is distilled here: ``downstream_kd=False`` stops after the repaired teacher, so
the claim is tested on the teacher where it is made, at a fraction of the cost.

    python3 scripts/channel_selection_ablation.py --plan      # what would run
    python3 scripts/channel_selection_ablation.py --device mps

Settings whose validated pool is too small for a disjoint ``antiscreened`` set skip that
arm, and the plan says so rather than running a control that restates the screened set.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kd.checkpoints import write_json
from kd.neuron_surgery_matrix import matrix_spec
from kd.neuron_surgery_study import CHANNEL_SELECTIONS, run_neuron_surgery_study

DRAWN_ARMS = ("random", "stage_matched_random")


def finished_settings(root: Path) -> list[dict]:
    """Settings whose original run selected a repair, with the choices it made."""
    settings = []
    for selection_path in sorted(root.glob("*/*/studies/confirmatory/teacher_selection.json")):
        selection = json.loads(selection_path.read_text())
        chosen = selection.get("selected")
        if not isinstance(chosen, dict):
            continue
        study = selection_path.parent
        localization = json.loads((study / "localization.json").read_text())
        settings.append({
            "dataset": study.parents[2].name,
            "profile": study.parents[1].name,
            "budget": int(chosen["budget"]),
            "learning_rate": float(chosen["learning_rate"]),
            "validated_channels": len(localization["payload"]["ranking"]),
            "source": str(study),
        })
    return settings


def arm_jobs(settings: list[dict], arms: tuple[str, ...], draws: int,
             seed: int, output_root: Path) -> tuple[list[dict], list[dict]]:
    jobs, skipped = [], []
    for setting in settings:
        for arm in arms:
            if arm == "antiscreened" and setting["validated_channels"] < 2 * setting["budget"]:
                skipped.append({**setting, "arm": arm,
                                "reason": f"{setting['validated_channels']} validated channels "
                                          f"cannot yield a disjoint set of {setting['budget']}"})
                continue
            for draw in range(draws if arm in DRAWN_ARMS else 1):
                name = arm if arm not in DRAWN_ARMS else f"{arm}-draw{draw}"
                jobs.append({**setting, "arm": arm, "draw": draw,
                             "selection_seed": seed + 1000 * draw,
                             "output": str(output_root / setting["dataset"] /
                                           setting["profile"] / name)})
    return jobs, skipped


def run_job(job: dict, baseline_root: Path, device: str, dry_run: bool) -> dict:
    spec = replace(matrix_spec(job["dataset"], job["profile"]),
                   channel_selection=job["arm"],
                   channel_selection_seed=job["selection_seed"],
                   channel_budgets=(job["budget"],),
                   learning_rates=(job["learning_rate"],),
                   downstream_kd=False)
    baseline = baseline_root / job["dataset"] / job["profile"] / "baselines" / "confirmatory"
    return run_neuron_surgery_study(baseline=str(baseline), output=job["output"],
                                    device=device, seeds=spec.student_seeds,
                                    dry_run=dry_run, spec=spec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=Path("runs/experiment1-multidataset"),
                        help="Finished Experiment 1 runs, for each setting's chosen budget")
    parser.add_argument("--baseline-root", type=Path,
                        help="Where the baseline teachers live (defaults to --source-root)")
    parser.add_argument("--output-root", type=Path, default=Path("runs/channel-selection-ablation"))
    parser.add_argument("--arms", nargs="+", default=list(CHANNEL_SELECTIONS),
                        choices=list(CHANNEL_SELECTIONS))
    parser.add_argument("--draws", type=int, default=3,
                        help="Draws per randomized arm; deterministic arms run once")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plan", action="store_true", help="List the jobs and stop")
    parser.add_argument("--dry-run", action="store_true",
                        help="Call each study in dry-run mode to validate its protocol")
    parser.add_argument("--keep-going", action="store_true",
                        help="Record failures and continue instead of stopping")
    args = parser.parse_args()

    settings = finished_settings(args.source_root)
    if not settings:
        print(f"No settings with a selected repair under {args.source_root}", file=sys.stderr)
        return 1
    jobs, skipped = arm_jobs(settings, tuple(args.arms), args.draws, args.seed, args.output_root)

    print(f"{len(settings)} settings with a repair, {len(jobs)} jobs")
    for arm in args.arms:
        print(f"  {arm:22} {sum(job['arm'] == arm for job in jobs):4} jobs")
    if skipped:
        print(f"\n{len(skipped)} arm/setting pairs skipped:")
        for row in skipped:
            print(f"    {row['dataset']}/{row['profile']:12} {row['arm']:14} {row['reason']}")
    if args.plan:
        return 0

    baseline_root = args.baseline_root or args.source_root
    results, failures = [], []
    for index, job in enumerate(jobs, start=1):
        label = f"{job['dataset']}/{job['profile']} {job['arm']} draw{job['draw']}"
        print(f"[{index}/{len(jobs)}] {label}", flush=True)
        try:
            outcome = run_job(job, baseline_root, args.device, args.dry_run)
            results.append({**job, "status": "ok", "result": outcome})
        except Exception as error:
            failures.append({**job, "status": "failed", "error": f"{type(error).__name__}: {error}"})
            print(f"    FAILED {type(error).__name__}: {error}", file=sys.stderr)
            if not args.keep_going:
                traceback.print_exc()
                break

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_json(args.output_root / "ablation_runs.json",
               {"seed": args.seed, "draws": args.draws, "arms": list(args.arms),
                "device": args.device, "dry_run": args.dry_run, "skipped": skipped,
                "results": results, "failures": failures})
    print(f"\n{len(results)} ok, {len(failures)} failed -> "
          f"{args.output_root / 'ablation_runs.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
