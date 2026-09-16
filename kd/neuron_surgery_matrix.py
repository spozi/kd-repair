"""Frozen multi-dataset Experiment 1 matrix and cross-study summary."""

from __future__ import annotations

import json
from pathlib import Path

from .checkpoints import fingerprint, write_json
from .dataset_registry import load_catalog
from .neuron_surgery_study import NeuronSurgerySpec


MATRIX_VERSION = 2
DATASET_ORDER = (
    "cinic10", "pathmnist", "svhn", "fashionmnist", "organamnist", "cifar100",
    "cifar10", "gtsrb", "eurosat", "bloodmnist", "caltech101", "dermamnist",
    "stl10",
)
_CATALOG = load_catalog()
DATASETS = {
    name: {"version": _CATALOG["datasets"][name]["version"],
           "classes": _CATALOG["datasets"][name]["classes"],
           "image_size": _CATALOG["datasets"][name]["image_size"]}
    for name in DATASET_ORDER
}
PROFILES = {
    "balanced": {"factor": 1.0, "epochs": 81, "target_policy": "all"},
    "lt-if10": {"factor": 10.0, "epochs": 49, "target_policy": "frequency_tail"},
    "lt-if50": {"factor": 50.0, "epochs": 71, "target_policy": "frequency_tail"},
    "lt-if100": {"factor": 100.0, "epochs": 81, "target_policy": "frequency_tail"},
}


def matrix_spec(dataset: str, profile: str) -> NeuronSurgerySpec:
    """Create one predeclared dataset/profile protocol."""
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported matrix dataset: {dataset}")
    if profile not in PROFILES:
        raise ValueError(f"Unsupported matrix profile: {profile}")
    data, profile_config = DATASETS[dataset], PROFILES[profile]
    slug = f"{dataset}-{profile}"
    spec = NeuronSurgerySpec(
        name=slug,
        imbalance_factor=profile_config["factor"],
        teacher_run=f"teacher_{slug}",
        student_run_template=f"kd_{slug}_seed{{seed}}",
        teacher_seed=41,
        student_seeds=(42, 43, 44),
        split_seed=2026,
        training_epochs=profile_config["epochs"],
        target_classes=None,
        target_class_policy=profile_config["target_policy"],
        expected_train_counts=None,
        expected_target_counts=None,
        dataset_source=dataset,
        dataset_num_classes=data["classes"],
        dataset_image_size=data["image_size"],
        dataset_version=data["version"],
        dataset_profile=profile,
        allow_sparse_class_fallback=True,
    )
    spec.validate()
    return spec


def matrix_jobs(datasets=None, profiles=None) -> list[dict]:
    """Return largest-dataset-first jobs for efficient dynamic scheduling."""
    datasets = tuple(DATASETS if datasets is None else datasets)
    profiles = tuple(PROFILES if profiles is None else profiles)
    jobs = []
    for dataset in datasets:
        for profile in profiles:
            spec = matrix_spec(dataset, profile)
            jobs.append({
                "id": spec.name,
                "dataset": dataset,
                "version": spec.dataset_version,
                "profile": profile,
                "output": f"{dataset}/{profile}",
                "study": json.loads(json.dumps(spec.to_dict(), allow_nan=False)),
            })
    return jobs


def _same_json(path: Path, value: dict) -> bool:
    return path.is_file() and json.loads(path.read_text()) == value


def materialize_matrix_plan(output: str | Path, datasets=None, profiles=None) -> dict:
    """Write immutable per-job configs plus a shell-readable queue."""
    output = Path(output).resolve()
    jobs = matrix_jobs(datasets, profiles)
    protocol = {
        "version": MATRIX_VERSION,
        "purpose": "Experiment 1 multi-dataset neuron repair and KD matrix",
        "catalog_version": _CATALOG["catalog_version"],
        "datasets": list(DATASETS if datasets is None else datasets),
        "profiles": list(PROFILES if profiles is None else profiles),
        "jobs": jobs,
        "source_sha256": fingerprint(Path(__file__)),
    }
    protocol = json.loads(json.dumps(protocol, allow_nan=False))
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and not _same_json(protocol_path, protocol):
        raise ValueError("Matrix protocol changed; use a new plan directory")
    if output.exists() and any(output.iterdir()) and not protocol_path.exists():
        raise FileExistsError(f"Nonempty matrix plan directory has no protocol: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not protocol_path.exists():
        write_json(protocol_path, protocol)

    config_directory = output / "configs"
    config_directory.mkdir(exist_ok=True)
    rows = []
    for job in jobs:
        config = config_directory / f"{job['id']}.json"
        if config.exists() and not _same_json(config, job["study"]):
            raise ValueError(f"Generated matrix config changed: {config}")
        if not config.exists():
            write_json(config, job["study"])
        rows.append("\t".join((job["id"], job["dataset"], job["version"],
                               job["profile"], str(config), job["output"])))
    queue_path = output / "jobs.tsv"
    queue_text = "\n".join(rows) + "\n"
    if queue_path.exists() and queue_path.read_text() != queue_text:
        raise ValueError("Matrix job queue changed; use a new plan directory")
    if not queue_path.exists():
        queue_path.write_text(queue_text)
    return {"status": "ready", "job_count": len(jobs), "output": str(output),
            "protocol": str(protocol_path), "jobs": str(queue_path)}


def summarize_matrix(plan: str | Path, output_root: str | Path) -> dict:
    """Summarize completed studies without pooling differently sized datasets."""
    plan, output_root = Path(plan).resolve(), Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((plan / "protocol.json").read_text())
    rows = []
    for job in protocol["jobs"]:
        comparison_path = (output_root / job["output"] / "studies" / "confirmatory" /
                           "comparison.json")
        if not comparison_path.is_file():
            rows.append({"id": job["id"], "dataset": job["dataset"],
                         "profile": job["profile"], "status": "missing"})
            continue
        comparison = json.loads(comparison_path.read_text())
        row = {"id": job["id"], "dataset": job["dataset"],
               "profile": job["profile"], "status": comparison.get("status", "unknown")}
        if comparison.get("status") == "complete":
            row.update(
                success=bool(comparison.get("success", {}).get("passed", False)),
                mean_accuracy_delta=comparison["aggregate"]["accuracy_delta"]["mean"],
                mean_target_recall_delta=comparison["aggregate"]
                .get("target_recall_delta", comparison["aggregate"]["tail_recall_delta"])["mean"],
            )
        else:
            row["reason"] = comparison.get("reason")
        rows.append(row)
    complete = [row for row in rows if row["status"] == "complete"]
    report = {
        "matrix_protocol_sha256": fingerprint(plan / "protocol.json"),
        "job_count": len(rows),
        "complete_count": len(complete),
        "successful_transfer_count": sum(bool(row.get("success")) for row in complete),
        "all_jobs_finished": all(row["status"] != "missing" for row in rows),
        "jobs": rows,
    }
    write_json(output_root / "matrix_summary.json", report)
    lines = ["# Experiment 1 multi-dataset summary", "",
             "| Dataset | Profile | Status | Transfer | Mean accuracy delta | Mean target-recall delta |",
             "|---|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['profile']} | {row['status']} | "
            f"{str(row.get('success', 'n/a')).lower()} | "
            f"{row.get('mean_accuracy_delta', float('nan')):+.4f} | "
            f"{row.get('mean_target_recall_delta', float('nan')):+.4f} |")
    (output_root / "matrix_summary.md").write_text("\n".join(lines) + "\n")
    return report
