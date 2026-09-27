"""Alternative distillation losses for any teacher-student pairing and dataset setting.

For each cell of a pairing (see pairing_cells.py) and each seed, students are retrained from the
cell's classical-KD control with only the distillation loss changed: LoCa, DKD and RLD at their
published settings, and DKD and RLD at a lower response weight that restores classical KD's
balance between the teacher term and the cross-entropy. Every student must reproduce its
control's initial weights and data and be distilled from the cell's original teacher. The
control itself is scored on the same test split, so cells whose repair was rejected, whose
students were never scored, still get a complete comparison.

The lower response weight is fixed per pairing from training-loss histories only, never from
test results: weight = (KD control's median teacher-term/cross-entropy ratio) / (the same ratio
for the published-setting run), medians over epochs after warm-up and over every cell and seed.
Run the published variants first, print the weights with --suggest-weights, then run the
lower-response variants with --matched-weights. CNN-T -> CNN-S keeps the weights already fixed
for it (dkd=0.15, rld=0.2), so its earlier runs stay valid.

--link-existing reuses finished CNN-T -> CNN-S runs of the same recipe (the distillation
baselines under the matrix and runs/matched-balance) by linking them into the output, after
checking that each one's config equals the config this script would train, apart from paths,
run names and loader settings.

    python scripts/alternative_losses.py --pairing r18-cnn --variants loca dkd rld --device cuda:0
    python scripts/alternative_losses.py --pairing r18-cnn --suggest-weights
    python scripts/alternative_losses.py --pairing r18-cnn --variants dkd_matched rld_matched \\
        --matched-weights dkd=0.14 rld=0.19 --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kd.checkpoints import fingerprint, write_json  # noqa: E402
from kd.config import from_dict  # noqa: E402
from kd.data import build_data  # noqa: E402
from kd.engine import run_experiment  # noqa: E402
from kd.neuron_surgery_study import _json, _model, _quality_report, _read, _save_predictions  # noqa: E402

from pairing_cells import PAIRINGS, Cell, cells  # noqa: E402

VARIANTS = ("loca", "dkd", "rld", "dkd_matched", "rld_matched")
WARMUP = 5  # epochs excluded when measuring the loss balance, as in the distillation warm-up
# Settings that locate a run or feed its loader, not part of what is trained.
EXECUTION = {("name",), ("output_dir",), ("data", "root"), ("teacher", "checkpoint"), ("train", "device"),
             ("train", "workers"), ("train", "threads"), ("train", "persistent_workers"),
             ("train", "prefetch_factor")}


def _spec(variant: str, control, weights: dict) -> tuple[str, float, float]:
    """(method, response weight, beta) for a variant; LoCa keeps the control's own weight."""
    if variant == "loca":
        return "loca", control.distillation.weight, control.distillation.beta
    method = variant.removesuffix("_matched")
    if variant.endswith("_matched"):
        if method not in weights:
            raise ValueError(f"{variant} needs --matched-weights {method}=W")
        return method, weights[method], 8.0
    return method, 1.0, 8.0


def _balance(history_path: Path, ce_weight: float, response_weight: float) -> float:
    history = _read(history_path)[WARMUP:]
    return statistics.median(response_weight * e["train"]["response"] / (ce_weight * e["train"]["ce"])
                             for e in history if e["train"]["ce"] > 0)


def suggest_weights(pairing_cells: list[Cell], output: Path, seeds: list[int]) -> dict:
    """Lower-response weights for DKD and RLD from training histories alone."""
    kd, published = [], {"dkd": [], "rld": []}
    for cell in pairing_cells:
        for seed in seeds:
            control = cell.control(seed)
            config = from_dict(_read(control / "config.json"))
            kd.append(_balance(control / "history.json", 1 - config.distillation.weight,
                               config.distillation.weight))
            for method in published:
                history = output / cell.job / f"{method}_{cell.job}_seed{seed}" / "history.json"
                if not history.is_file():
                    raise FileNotFoundError(f"Train the published {method} variant first: {history}")
                published[method].append(_balance(history, 1.0, 1.0))
    reference = statistics.median(kd)
    return {"kd_balance": reference,
            **{f"{m}_balance": statistics.median(v) for m, v in published.items()},
            **{m: round(reference / statistics.median(v), 3) for m, v in published.items()}}


def _config(cell: Cell, seed: int, variant: str, weights: dict, output: Path, device: str = "cpu",
            data_root: str = "data", workers: int | None = None):
    control = from_dict(_read(cell.control(seed) / "config.json"))
    method, weight, beta = _spec(variant, control, weights)
    return replace(
        control, name=f"{variant}_{cell.job}_seed{seed}", output_dir=str(output / cell.job),
        teacher=replace(control.teacher, checkpoint=str(cell.teacher)),
        data=replace(control.data, root=data_root),
        train=replace(control.train, device=device, **({} if workers is None else {"workers": workers})),
        distillation=replace(control.distillation, method=method, weight=weight, beta=beta))


def _recipe(config) -> dict:
    recipe = asdict(config)
    for path in EXECUTION:
        section = recipe
        for key in path[:-1]:
            section = section[key]
        section.pop(path[-1])
    return recipe


def _existing(cell: Cell, seed: int, variant: str, weights: dict, matrix: Path, matched: Path) -> Path | None:
    """A finished CNN-T -> CNN-S run of this variant from the earlier studies, if there is one."""
    method = variant.removesuffix("_matched")
    if variant == method:
        path = cell.root / "baselines" / "distillation" / method / f"{method}_{cell.job}_seed{seed}"
        return path if (path / "summary.json").is_file() else None
    found = sorted(matched.glob(f"*/{cell.job}/{method}_w{weights[method]:g}_{cell.job}_seed{seed}"))
    found = [path for path in found if (path / "summary.json").is_file()]
    if len(found) > 1:
        raise ValueError(f"Several finished {variant} runs for {cell.job} seed {seed}: {found}")
    return found[0] if found else None


def link_existing(pairing_cells: list[Cell], variants, seeds, weights: dict, output: Path,
                  matrix: Path, matched: Path, dry_run: bool = False) -> int:
    """Link finished runs whose recipe equals this script's; returns how many were (or would be) linked."""
    linked = 0
    for cell in pairing_cells:
        for seed in seeds:
            for variant in variants:
                target = output / cell.job / f"{variant}_{cell.job}_seed{seed}"
                if target.is_symlink() or target.exists():
                    continue
                source = _existing(cell, seed, variant, weights, matrix, matched)
                if source is None:
                    continue
                expected = _recipe(_config(cell, seed, variant, weights, output))
                if _recipe(from_dict(_read(source / "config.json"))) != expected:
                    raise ValueError(f"{source} was trained with a different recipe than {variant}")
                linked += 1
                if dry_run:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()),
                                  target_is_directory=True)
    return linked


def _train(cell: Cell, seed: int, variant: str, weights: dict, output: Path, device: str,
           data_root: str, workers: int | None) -> Path:
    control_dir = cell.control(seed)
    config = _config(cell, seed, variant, weights, output, device, data_root, workers)
    directory = output / cell.job / config.name
    if not (directory / "summary.json").is_file():
        last = directory / "last.pt"
        print(f"[train] {config.name}", flush=True)
        run_experiment(config, resume=str(last) if last.is_file() else None)
    summary, reference = _read(directory / "summary.json"), _read(control_dir / "summary.json")
    for key in ("initial_student_sha256", "data"):
        if summary[key] != reference[key]:
            raise ValueError(f"{config.name} differs from its classical-KD control in {key}")
    if summary["teacher"]["checkpoint_sha256"] != fingerprint(cell.teacher):
        raise ValueError(f"{config.name} was not distilled from the original teacher")
    return directory


def _score(checkpoint_dir: Path, name: str, seed: int, evaluation, cell: Cell, output: Path) -> dict:
    config = from_dict(_read(checkpoint_dir / "config.json"))
    checkpoint = checkpoint_dir / "student.pt"
    model, _ = _model(config, checkpoint, evaluation.classes)
    # Outside the run folder, which may be a link into an earlier study's results.
    directory = output / cell.job / "evaluation" / name
    identity = {"checkpoint_sha256": fingerprint(checkpoint), "seed": seed, "condition": name,
                "evaluation_split": cell.spec["evaluation_split"], "data": evaluation.provenance}
    split = getattr(evaluation, cell.spec["evaluation_split"])
    labels, probabilities = _save_predictions(directory / "predictions.npz", identity, model, split,
                                              evaluation.classes)
    metrics = _quality_report(directory, labels, probabilities, evaluation.classes, cell.target_classes)
    return {"accuracy": metrics["accuracy"], "target_recall": metrics["target_macro_recall"],
            "response_weight": config.distillation.weight, "method": config.distillation.method}


def run(args) -> None:
    teachers = args.teachers.resolve() if args.teachers else None
    students = args.students.resolve() if args.students else None
    pairing_cells = cells(args.pairing, args.matrix.resolve(), teachers=teachers, students=students,
                          jobs=args.jobs, verify_repaired=False)
    output = (args.output or Path("runs/alternative-losses") / args.pairing).resolve()
    if args.suggest_weights:
        print(suggest_weights(pairing_cells, output, args.seeds))
        return
    weights = dict(pair.split("=") for pair in args.matched_weights)
    weights = {k: float(v) for k, v in weights.items()}
    data_root = str(Path(args.data_root).resolve())
    linked = 0
    if args.link_existing:
        if args.pairing != "cnn-cnn":
            raise SystemExit("--link-existing applies to cnn-cnn, whose earlier runs share its controls")
        linked = link_existing(pairing_cells, args.variants, args.seeds, weights, output,
                               args.matrix.resolve(), args.matched_root.resolve(), dry_run=args.dry_run)
        print(f"[link] {linked} finished runs {'would be ' if args.dry_run else ''}reused", flush=True)
    planned = {"cells": len(pairing_cells), "to_train": -linked if args.dry_run else 0}
    for cell in pairing_cells:
        for seed in args.seeds:
            control = cell.control(seed)
            for path in (control / "config.json", control / "summary.json", control / "student.pt", cell.teacher):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing input for {cell.job}: {path}")
            for variant in args.variants:
                _spec(variant, from_dict(_read(control / "config.json")), weights)
                planned["to_train"] += not (output / cell.job / f"{variant}_{cell.job}_seed{seed}"
                                            / "summary.json").is_file()
    print(f"[plan] {args.pairing}: {planned['cells']} cells, {planned['to_train']} students still to train",
          flush=True)
    if args.dry_run:
        return
    report = _read(output / "report.json") if (output / "report.json").is_file() else {}
    for cell in pairing_cells:
        trained = {(v, s): _train(cell, s, v, weights, output, args.device, data_root, args.workers)
                   for s in args.seeds for v in args.variants}
        teacher_config = from_dict(_read(cell.root / "baselines" / "confirmatory" / cell.spec["teacher_run"]
                                         / "config.json"))
        teacher_config = replace(teacher_config, data=replace(teacher_config.data, root=data_root))
        split = cell.spec["evaluation_split"]
        evaluation = build_data(teacher_config.data, replace(teacher_config.train, workers=0, device="cpu"),
                                include_test=split == "test", include_confirmation=split == "confirmation",
                                diagnostic=True)
        reference = cell.reference_data(args.seeds[0])
        if reference is not None and _json(evaluation.provenance) != reference:
            raise ValueError(f"{cell.job}: evaluation data differs from the split the study was scored on")
        entry = report.setdefault(cell.job, {"repair": "selected" if cell.repaired else "none"})
        entry["kd"] = {seed: _score(cell.control(seed), f"kd_control_seed{seed}", seed, evaluation, cell, output)
                       for seed in args.seeds}
        for variant in args.variants:
            entry[variant] = {seed: _score(trained[(variant, seed)], f"{variant}_{cell.job}_seed{seed}", seed,
                                           evaluation, cell, output) for seed in args.seeds}
        write_json(output / "report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairing", required=True, choices=sorted(PAIRINGS))
    parser.add_argument("--matrix", type=Path, default=Path("runs/experiment1-multidataset"))
    parser.add_argument("--teachers", type=Path, default=Path("runs/teacher-architecture"),
                        help="teacher_architecture_study.py output (ResNet-teacher pairings)")
    parser.add_argument("--students", type=Path, default=Path("runs/student-architecture"),
                        help="student_architecture_study.py output root (cnn-resnet pairings)")
    parser.add_argument("--jobs", nargs="*", default=None, help="Setting ids (default: every decided cell)")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["loca", "dkd", "rld"])
    parser.add_argument("--matched-weights", nargs="*", default=[], metavar="METHOD=W",
                        help="Lower response weights, e.g. dkd=0.15 rld=0.2")
    parser.add_argument("--suggest-weights", action="store_true",
                        help="Print the lower response weights from training histories and exit")
    parser.add_argument("--link-existing", action="store_true",
                        help="cnn-cnn: reuse finished runs of the same recipe from the earlier studies")
    parser.add_argument("--matched-root", type=Path, default=Path("runs/matched-balance"),
                        help="Where the earlier lower-response DKD/RLD runs live (with --link-existing)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Default: runs/alternative-losses/PAIRING")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and count students without training")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
