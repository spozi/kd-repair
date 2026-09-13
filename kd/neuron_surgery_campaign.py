"""Sequential, resumable campaign for neuron-surgery confirmation and generalization."""

from __future__ import annotations

import json
from pathlib import Path

from .checkpoints import fingerprint, write_json
from .neuron_surgery_baselines import run_neuron_surgery_baselines
from .neuron_surgery_study import (NeuronSurgerySpec, load_neuron_surgery_spec,
                                   run_neuron_surgery_study)


CAMPAIGN_VERSION = 1
CONFIG_DIRECTORY = Path(__file__).resolve().parent.parent / "configs" / "neuron_surgery"


def _load_specs() -> dict[str, NeuronSurgerySpec]:
    names = {
        "factor10": "f10.json",
        "factor50": "f50.json",
        "confirmatory": "confirmatory_f100.json",
        "resnet": "resnet_f100.json",
        "no_preservation_kd": "ablation_no_kd.json",
        "no_companions": "ablation_no_companions.json",
        "no_causal_validation": "ablation_no_causal.json",
    }
    return {name: load_neuron_surgery_spec(CONFIG_DIRECTORY / filename)
            for name, filename in names.items()}


def campaign_plan(output: str | Path, data_root: str | Path,
                  legacy_baseline: str | Path, reference_study: str | Path,
                  device: str) -> dict:
    specs = _load_specs()
    directory = Path(output).resolve()
    return {
        "version": CAMPAIGN_VERSION,
        "output": str(directory),
        "data_root": str(Path(data_root).resolve()),
        "legacy_baseline": str(Path(legacy_baseline).resolve()),
        "reference_study": str(Path(reference_study).resolve()),
        "device": device,
        "execution": "Sequential and resumable; no concurrent accelerator jobs",
        "resnet_gate": ("Run the expensive ResNet baseline/study only if at least two of "
                        "factor10, factor50, and sealed factor100 pass the transfer rule"),
        "stages": [
            "factor10_existing_baseline",
            "sealed_factor100_baselines_and_study",
            "factor100_component_ablations_validation_only",
            "factor50_baselines_and_study",
            "resnet34_to_resnet18_baselines_and_study",
            "aggregate",
        ],
        "specs": {name: spec.to_dict() for name, spec in specs.items()},
        "config_sha256": {
            path.name: fingerprint(path) for path in sorted(CONFIG_DIRECTORY.glob("*.json"))
        },
        "source_sha256": {
            name: fingerprint(Path(__file__).resolve().parent / name)
            for name in ("config.py", "data.py", "models.py", "neuron_surgery.py",
                         "neuron_surgery_study.py", "neuron_surgery_baselines.py",
                         "neuron_surgery_campaign.py")
        },
    }


def _validate_reference(directory: Path) -> None:
    manifest_path = directory / "report_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing completed reference manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    for name, digest in manifest.get("files", {}).items():
        if fingerprint(directory / name) != digest:
            raise ValueError(f"Reference study artifact changed: {directory / name}")


def _best_candidates_by_budget(rows: list[dict]) -> dict[str, dict]:
    best = {}
    for budget in sorted({row["budget"] for row in rows}):
        options = [row for row in rows if row["budget"] == budget]
        options.sort(key=lambda row: (not row["eligible"], -row["tail_recall_delta"],
                                      row["tail_nll_delta"], row["learning_rate"]))
        best[str(budget)] = options[0]
    return best


def _budget_summary(reference: Path) -> dict:
    _validate_reference(reference)
    selection = json.loads((reference / "teacher_selection.json").read_text())
    baseline = selection["baseline_validation"]
    candidates = json.loads((reference / "candidate_results.json").read_text())
    rows = []
    for candidate in candidates:
        validation = candidate["validation"]
        rows.append({
            "budget": candidate["budget"],
            "learning_rate": candidate["learning_rate"],
            "eligible": (validation["accuracy"] >= baseline["accuracy"] - 0.005
                         and validation["tail_macro_recall"] > baseline["tail_macro_recall"]),
            "accuracy_delta": validation["accuracy"] - baseline["accuracy"],
            "tail_recall_delta": (validation["tail_macro_recall"]
                                  - baseline["tail_macro_recall"]),
            "tail_nll_delta": validation["tail_balanced_nll"] - baseline["tail_balanced_nll"],
        })
    best_by_budget = _best_candidates_by_budget(rows)
    return {"reference": str(reference), "baseline_validation": baseline,
            "candidates": rows, "best_by_budget": best_by_budget}


def _write_progress(directory: Path, completed: list[str], phase: str) -> None:
    write_json(directory / "progress.json", {"phase": phase, "completed": completed})


def _render_report(comparisons: dict, budget: dict) -> str:
    lines = ["# Neuron-surgery follow-up campaign", "",
             "| Study | Status | Transfer passed | Mean accuracy delta | Mean tail delta |",
             "|---|---|---:|---:|---:|"]
    for name, result in comparisons.items():
        aggregate = result.get("aggregate", {})
        accuracy = aggregate.get("accuracy_delta", {}).get("mean")
        tail = aggregate.get("tail_recall_delta", {}).get("mean")
        lines.append(
            f"| {name} | {result.get('status')} | {result.get('success', {}).get('passed', '')} | "
            f"{'' if accuracy is None else f'{accuracy:+.4f}'} | "
            f"{'' if tail is None else f'{tail:+.4f}'} |")
    lines.extend(["", "## Channel budgets", "",
                  "| Budget | Learning rate | Validation accuracy delta | Validation tail delta |",
                  "|---:|---:|---:|---:|"])
    for value in budget["best_by_budget"].values():
        lines.append(f"| {value['budget']} | {value['learning_rate']:.4g} | "
                     f"{value['accuracy_delta']:+.4f} | {value['tail_recall_delta']:+.4f} |")
    return "\n".join([*lines, ""])


def run_neuron_surgery_campaign(output="runs/cifar10-lt-neuron-campaign", *,
                                data_root="data",
                                legacy_baseline="runs/cifar10-lt-multiseed",
                                reference_study="runs/cifar10-lt-neuron-surgery",
                                device="auto", dry_run=False) -> dict:
    """Run the predeclared follow-up studies one at a time with stage reuse."""
    plan = campaign_plan(output, data_root, legacy_baseline, reference_study, device)
    if dry_run:
        return {"dry_run": True, "plan": plan}
    directory = Path(output).resolve()
    protocol_path = directory / "protocol.json"
    if directory.exists() and any(directory.iterdir()) and not protocol_path.exists():
        raise FileExistsError(f"Nonempty campaign directory has no protocol: {directory}")
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != plan:
        raise ValueError("Campaign protocol changed; use a new output directory")
    directory.mkdir(parents=True, exist_ok=True)
    if not protocol_path.exists():
        write_json(protocol_path, plan)
    completion_path = directory / "completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        if completion.get("protocol_sha256") != fingerprint(protocol_path):
            raise ValueError("Completed campaign belongs to another protocol")
        for name, digest in completion.get("files", {}).items():
            if fingerprint(directory / name) != digest:
                raise ValueError(f"Completed campaign artifact changed: {name}")
        return json.loads((directory / "comparison.json").read_text())

    specs = _load_specs()
    completed = []
    comparisons = {}

    comparisons["factor10"] = run_neuron_surgery_study(
        plan["legacy_baseline"], directory / "studies" / "factor10", device,
        specs["factor10"].student_seeds, spec=specs["factor10"])
    completed.append("factor10")
    _write_progress(directory, completed, "studies")

    confirmatory_baseline = directory / "baselines" / "confirmatory"
    run_neuron_surgery_baselines(
        str(confirmatory_baseline), plan["data_root"], device, specs["confirmatory"])
    comparisons["confirmatory"] = run_neuron_surgery_study(
        confirmatory_baseline, directory / "studies" / "confirmatory", device,
        specs["confirmatory"].student_seeds, spec=specs["confirmatory"])
    completed.append("confirmatory")
    _write_progress(directory, completed, "studies")

    for name in ("no_preservation_kd", "no_companions", "no_causal_validation"):
        spec = specs[name]
        comparisons[name] = run_neuron_surgery_study(
            plan["legacy_baseline"], directory / "studies" / name, device,
            spec.student_seeds, spec=spec)
        completed.append(name)
        _write_progress(directory, completed, "studies")

    factor50_baseline = directory / "baselines" / "factor50"
    run_neuron_surgery_baselines(
        str(factor50_baseline), plan["data_root"], device, specs["factor50"])
    comparisons["factor50"] = run_neuron_surgery_study(
        factor50_baseline, directory / "studies" / "factor50", device,
        specs["factor50"].student_seeds, spec=specs["factor50"])
    completed.append("factor50")
    _write_progress(directory, completed, "studies")

    transfer_passes = sum(
        comparisons[name].get("status") == "complete"
        and comparisons[name].get("success", {}).get("passed", False)
        for name in ("factor10", "factor50", "confirmatory"))
    if transfer_passes >= 2:
        resnet_baseline = directory / "baselines" / "resnet"
        run_neuron_surgery_baselines(
            str(resnet_baseline), plan["data_root"], device, specs["resnet"])
        comparisons["resnet"] = run_neuron_surgery_study(
            resnet_baseline, directory / "studies" / "resnet", device,
            specs["resnet"].student_seeds, spec=specs["resnet"])
    else:
        comparisons["resnet"] = {
            "status": "skipped_by_predeclared_gate", "transfer_passes": transfer_passes,
            "reason": "Fewer than two preceding CifarCNN factor studies passed transfer."
        }
    completed.append("resnet")

    budget = _budget_summary(Path(plan["reference_study"]))
    write_json(directory / "budget_comparison.json", budget)
    report = {"status": "complete", "protocol_sha256": fingerprint(protocol_path),
              "comparisons": comparisons, "budget_comparison": budget}
    write_json(directory / "comparison.json", report)
    (directory / "report.md").write_text(_render_report(comparisons, budget))
    write_json(completion_path, {
        "protocol_sha256": fingerprint(protocol_path),
        "files": {name: fingerprint(directory / name)
                  for name in ("comparison.json", "budget_comparison.json", "report.md")},
    })
    _write_progress(directory, completed, "complete")
    return report
