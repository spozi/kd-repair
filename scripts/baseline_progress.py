"""Show how far the distillation-baseline runs of an Experiment 1 matrix have progressed.

Reads only files the runs write as they go: history.json after every epoch, completion.json
for each finished student and comparison.json for each scored run. It never touches
training, so it is safe to run at any time from another shell, and its counts survive
restarts because they come from disk rather than from a log.

Progress is weighted by training images processed (epochs x training-set size), so a
balanced study's student counts for more than a small long-tailed one.

    python scripts/baseline_progress.py --matrix runs/experiment1-multidataset --watch 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

METHODS = ("dkd", "rld", "loca")
WIDTH = 30


def _read(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _epochs_done(student: Path, epochs: int) -> int:
    if (student / "completion.json").is_file():
        return epochs
    try:
        return min(len(_read(student / "history.json")), epochs)
    except (FileNotFoundError, ValueError):
        return 0


def collect(matrix: Path, jobs: list[str] | None = None, methods=METHODS) -> list[dict]:
    """One record per (study, method) run, with epochs completed by each of its students."""
    complete = [job for job in _read(matrix / "matrix_summary.json")["jobs"]
                if job["status"] == "complete" and (not jobs or job["id"] in jobs)]
    runs = []
    for job in complete:
        profile = matrix / job["dataset"] / job["profile"]
        study = _read(profile / "studies" / "confirmatory" / "comparison.json")["study"]
        seeds = study["student_seeds"]
        control = profile / "baselines" / "confirmatory" / study["student_run_template"].format(seed=seeds[0])
        epochs = _read(control / "config.json")["train"]["epochs"]
        images = sum(_read(control / "summary.json")["data"]["train_per_class"])
        for method in methods:
            output = profile / "baselines" / "distillation" / method
            comparison = output / "comparison.json"
            runs.append({
                "job": job["id"], "method": method, "epochs": epochs, "images": images,
                "done": [_epochs_done(output / f"{method}_{study['name']}_seed{seed}", epochs)
                         for seed in seeds],
                "scored": comparison.is_file() and _read(comparison).get("status") == "complete",
            })
    return runs


def work(runs: list[dict]) -> tuple[int, int]:
    """Training images processed so far and in total."""
    return (sum(run["images"] * sum(run["done"]) for run in runs),
            sum(run["images"] * run["epochs"] * len(run["done"]) for run in runs))


def bar(fraction: float, width: int = WIDTH) -> str:
    filled = min(width, int(fraction * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def render(runs: list[dict], baseline: int | None = None, elapsed: float | None = None) -> str:
    done, total = work(runs)
    students = sum(len(run["done"]) for run in runs)
    finished = sum(epochs == run["epochs"] for run in runs for epochs in run["done"])
    scored = sum(run["scored"] for run in runs)
    eta = "--:--:--"
    if baseline is not None and elapsed and done > baseline:
        eta = clock((total - done) * elapsed / (done - baseline))
    fraction = done / total if total else 1.0
    lines = [f"{bar(fraction)} {100 * fraction:5.1f}%  students {finished}/{students}  "
             f"runs scored {scored}/{len(runs)}  ETA {eta}"]
    for run in runs:
        if run["scored"] or not any(run["done"]):
            continue
        pending = [index for index, epochs in enumerate(run["done"]) if epochs < run["epochs"]]
        if pending:
            current = pending[0]
            where = (f"student {current + 1}/{len(run['done'])}  "
                     f"epoch {run['done'][current]}/{run['epochs']}")
        else:
            where = "scoring on the test split"
        own = sum(run["done"]) / (run["epochs"] * len(run["done"]))
        lines.append(f"  {run['job']:<22} {run['method']:<4} {bar(own, 12)}  {where}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matrix", type=Path, default=Path("runs/experiment1-multidataset"),
                        help="Matrix root with matrix_summary.json")
    parser.add_argument("--jobs", default="", help="Comma-separated study ids (default: all)")
    parser.add_argument("--methods", default=",".join(METHODS), help="Comma-separated methods")
    parser.add_argument("--watch", type=float, default=0,
                        help="Redraw every N seconds, estimating the ETA from observed progress")
    parser.add_argument("--count", action="store_true",
                        help="Print only 'images_done<TAB>images_total' (for the launcher)")
    parser.add_argument("--baseline-work", type=int, default=None,
                        help="Images already done when timing started (for the ETA)")
    parser.add_argument("--elapsed", type=float, default=None,
                        help="Seconds since --baseline-work was measured (for the ETA)")
    args = parser.parse_args()
    jobs = [job for job in args.jobs.split(",") if job] or None
    methods = [method for method in args.methods.split(",") if method]

    def snapshot() -> list[dict]:
        return collect(args.matrix, jobs, methods)

    if args.count:
        print("\t".join(map(str, work(snapshot()))))
        return
    if not args.watch:
        print(render(snapshot(), args.baseline_work, args.elapsed))
        return
    start, baseline = time.monotonic(), work(snapshot())[0]
    redraw = sys.stdout.isatty()
    while True:
        runs = snapshot()
        text = render(runs, baseline, time.monotonic() - start)
        print(("\033[H\033[J" if redraw else "") + time.strftime("%H:%M:%S ") + text, flush=True)
        done, total = work(runs)
        if done >= total and all(run["scored"] for run in runs):
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
