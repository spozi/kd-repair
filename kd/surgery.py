"""Auditable sample/slice diagnostics and bounded training-set surgery."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .checkpoints import fingerprint, load_model_checkpoint, metadata, write_json
from .config import ExperimentConfig, SurgeryConfig
from .data import DataBundle, SyntheticImages, build_data, seed_worker
from .models import create_model


FORMAT_VERSION = 1
SCORES = {"high_loss", "high_confidence_error"}
STRATEGIES = {"sample", "slice", "hybrid"}


def _json_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _dataset_position_descriptor(dataset) -> dict:
    if isinstance(dataset, Subset):
        indices = [int(index) for index in dataset.indices]
        return {
            "kind": "subset",
            "size": len(indices),
            "indices_sha256": _json_sha256(indices),
            "base": _dataset_position_descriptor(dataset.dataset),
        }
    if isinstance(dataset, SyntheticImages):
        return {
            "kind": "synthetic",
            "count": dataset.count,
            "classes": dataset.classes,
            "image_size": dataset.size,
            "seed": dataset.seed,
        }
    if hasattr(dataset, "samples"):
        root = Path(dataset.root).resolve()
        samples = []
        for path, label in dataset.samples:
            resolved = Path(path).resolve()
            try:
                name = str(resolved.relative_to(root))
            except ValueError:
                name = str(resolved)
            samples.append([name, int(label)])
        return {"kind": type(dataset).__name__, "size": len(samples),
                "samples_sha256": _json_sha256(samples)}
    if hasattr(dataset, "targets"):
        targets = [int(label) for label in dataset.targets]
        return {"kind": type(dataset).__name__, "size": len(targets),
                "targets_sha256": _json_sha256(targets),
                "train": bool(getattr(dataset, "train", False))}
    return {"kind": type(dataset).__name__, "size": len(dataset)}


def dataset_contract(config: ExperimentConfig, data: DataBundle, split: str) -> dict:
    loader = getattr(data, split, None)
    if loader is None:
        raise ValueError(f"Dataset has no {split} split")
    contract = {
        "source": config.data.source,
        "split": split,
        "size": len(loader.dataset),
        "classes": list(data.classes),
        "num_classes": config.data.num_classes,
        "image_size": config.data.image_size,
        "positions": _dataset_position_descriptor(loader.dataset),
    }
    if config.data.source == "synthetic":
        contract["train_seed"] = config.train.seed
    else:
        contract["root"] = str(Path(config.data.root).resolve())
    if config.data.source == "cifar10":
        contract.update(validation_fraction=config.data.validation_fraction,
                        split_seed=config.data.split_seed,
                        imbalance_factor=config.data.imbalance_factor)
    return contract


@torch.inference_mode()
def collect_sample_diagnostics(model, loader, device: torch.device) -> list[dict]:
    """Collect stable dataset-local scores from a non-shuffled diagnostic loader."""
    previous_mode = model.training
    model.eval()
    records = []
    offset = 0
    try:
        for images, labels in loader:
            labels_device = labels.to(device)
            output = model(images.to(device))
            logits = output.logits if hasattr(output, "logits") else output
            logits = logits.float()
            if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
                raise ValueError("Diagnostic model must return [batch, classes] logits")
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite logits in surgery diagnostics")
            log_probabilities = F.log_softmax(logits, dim=1)
            probabilities = log_probabilities.exp()
            confidence, prediction = probabilities.max(dim=1)
            true_probability = probabilities.gather(1, labels_device[:, None]).squeeze(1)
            nll = -log_probabilities.gather(1, labels_device[:, None]).squeeze(1)
            top_two = probabilities.topk(2, dim=1).values
            margin = top_two[:, 0] - top_two[:, 1]
            batch = zip(labels.tolist(), prediction.cpu().tolist(), confidence.cpu().tolist(),
                        true_probability.cpu().tolist(), nll.cpu().tolist(), margin.cpu().tolist())
            for row, values in enumerate(batch):
                true_label, predicted_label, conf, true_prob, loss, prediction_margin = values
                records.append({
                    "index": offset + row,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "correct": predicted_label == true_label,
                    "confidence": conf,
                    "true_probability": true_prob,
                    "nll": loss,
                    "prediction_margin": prediction_margin,
                })
            offset += labels.shape[0]
    finally:
        model.train(previous_mode)
    if offset != len(loader.dataset):
        raise ValueError("Diagnostic loader must visit every sample exactly once")
    return records


def _summarize(records: list[dict]) -> dict:
    count = len(records)
    accuracy = sum(row["correct"] for row in records) / count
    confidence = sum(row["confidence"] for row in records) / count
    return {
        "support": count,
        "errors": count - sum(row["correct"] for row in records),
        "accuracy": accuracy,
        "mean_nll": sum(row["nll"] for row in records) / count,
        "mean_confidence": confidence,
        "overconfidence_gap": confidence - accuracy,
    }


def slice_report(records: list[dict], classes: list[str]) -> dict:
    true_groups: dict[int, list[dict]] = {}
    confusion_groups: dict[tuple[int, int], list[dict]] = {}
    for row in records:
        true_groups.setdefault(row["true_label"], []).append(row)
        if not row["correct"]:
            key = (row["true_label"], row["predicted_label"])
            confusion_groups.setdefault(key, []).append(row)
    true_slices = [
        {"key": f"true:{label}", "true_label": label, "true_class": classes[label],
         **_summarize(group)}
        for label, group in true_groups.items()
    ]
    confusion_slices = [
        {"key": f"confusion:{true_label}->{predicted_label}",
         "true_label": true_label, "true_class": classes[true_label],
         "predicted_label": predicted_label, "predicted_class": classes[predicted_label],
         **_summarize(group)}
        for (true_label, predicted_label), group in confusion_groups.items()
    ]
    true_slices.sort(key=lambda row: (-row["mean_nll"], row["true_label"]))
    confusion_slices.sort(
        key=lambda row: (-row["support"], -row["mean_confidence"], row["key"]))
    return {"true_class": true_slices, "confusion": confusion_slices}


def _record_score(row: dict, score: str) -> float:
    return row["nll"] if score == "high_loss" else row["confidence"]


def _record_slice(row: dict, score: str) -> str:
    if score == "high_confidence_error":
        return f"confusion:{row['true_label']}->{row['predicted_label']}"
    return f"true:{row['true_label']}"


def select_bad_samples(records: list[dict], budget: int, *, score: str,
                       strategy: str, min_slice_size: int) -> tuple[list[dict], list[str]]:
    if score not in SCORES:
        raise ValueError(f"score must be one of {sorted(SCORES)}")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(STRATEGIES)}")
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("budget must be a positive integer")
    if not isinstance(min_slice_size, int) or isinstance(min_slice_size, bool) or min_slice_size < 1:
        raise ValueError("min_slice_size must be a positive integer")
    candidates = ([row for row in records if not row["correct"]]
                  if score == "high_confidence_error" else list(records))
    candidates.sort(key=lambda row: (-_record_score(row, score), -row["nll"], row["index"]))
    budget = min(budget, len(candidates))
    if not budget:
        return [], []

    groups: dict[str, list[dict]] = {}
    for row in candidates:
        groups.setdefault(_record_slice(row, score), []).append(row)
    groups = {key: value for key, value in groups.items() if len(value) >= min_slice_size}
    slice_keys = sorted(
        groups,
        key=lambda key: (-sum(_record_score(row, score) for row in groups[key]) / len(groups[key]),
                         -len(groups[key]), key),
    )

    selected: list[dict] = []
    selected_indices: set[int] = set()
    slice_budget = budget if strategy == "slice" else math.ceil(budget / 2)
    if strategy != "sample" and slice_keys:
        positions = {key: 0 for key in slice_keys}
        while len(selected) < slice_budget:
            progressed = False
            for key in slice_keys:
                position = positions[key]
                if position >= len(groups[key]) or len(selected) >= slice_budget:
                    continue
                row = groups[key][position]
                positions[key] += 1
                selected.append({**row, "score": _record_score(row, score),
                                 "selection_reason": "slice", "slice_key": key})
                selected_indices.add(row["index"])
                progressed = True
            if not progressed:
                break

    if strategy != "slice" or not selected:
        for row in candidates:
            if len(selected) >= budget:
                break
            if row["index"] in selected_indices:
                continue
            selected.append({**row, "score": _record_score(row, score),
                             "selection_reason": "sample",
                             "slice_key": _record_slice(row, score)})
            selected_indices.add(row["index"])
    return selected, slice_keys


def create_surgery_plan(config: ExperimentConfig, checkpoint: str | Path, output: str | Path,
                        device: torch.device, *, split: str = "train", fraction: float = 0.05,
                        max_samples: int | None = None, score: str = "high_confidence_error",
                        strategy: str = "hybrid", min_slice_size: int = 2) -> dict:
    config.validate(require_teacher=False)
    if split not in {"train", "val", "test"}:
        raise ValueError("Surgery diagnostics split must be train, val, or test")
    if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("fraction must be finite and in (0, 1)")
    if max_samples is not None and (not isinstance(max_samples, int)
                                    or isinstance(max_samples, bool) or max_samples < 1):
        raise ValueError("max_samples must be a positive integer")

    data = build_data(config.data, config.train, include_test=split == "test", diagnostic=True)
    loader = getattr(data, split, None)
    if loader is None:
        raise ValueError(f"Dataset has no {split} split")
    model = create_model(config.student.name, config.data.num_classes)
    expected = metadata(config.student.name, data.classes, config.data.image_size, config.data.source)
    load_model_checkpoint(model, checkpoint, expected)
    model.to(device)
    records = collect_sample_diagnostics(model, loader, device)
    budget = max(1, math.ceil(len(records) * fraction))
    if max_samples is not None:
        budget = min(budget, max_samples)
    selected, selected_slice_keys = select_bad_samples(
        records, budget, score=score, strategy=strategy, min_slice_size=min_slice_size)
    contract = dataset_contract(config, data, split)
    baseline = _summarize(records)
    plan = {
        "format_version": FORMAT_VERSION,
        "kind": "sample_slice_surgery",
        "dataset": {"contract": contract, "sha256": _json_sha256(contract)},
        "model": {"name": config.student.name, "checkpoint": str(Path(checkpoint).resolve()),
                  "checkpoint_sha256": fingerprint(checkpoint)},
        "selection": {"score": score, "strategy": strategy, "fraction": fraction,
                      "budget": budget, "selected_count": len(selected),
                      "min_slice_size": min_slice_size,
                      "eligible_slice_keys": selected_slice_keys},
        "baseline": {"samples": len(records), **baseline},
        "slices": slice_report(records, data.classes),
        "selected_samples": selected,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, plan)
    return plan


def _load_plan(path: str | Path) -> dict:
    with Path(path).open() as handle:
        plan = json.load(handle)
    if not isinstance(plan, dict) or plan.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported surgery plan format_version")
    if plan.get("kind") != "sample_slice_surgery":
        raise ValueError("Expected a sample_slice_surgery plan")
    if not isinstance(plan.get("selected_samples"), list):
        raise ValueError("Surgery plan selected_samples must be a list")
    return plan


def apply_training_surgery(data: DataBundle, config: ExperimentConfig) -> tuple[DataBundle, dict]:
    surgery: SurgeryConfig = config.surgery
    if surgery.action == "none":
        return data, {"enabled": False, "action": "none"}
    plan = _load_plan(surgery.plan)
    contract = dataset_contract(config, data, "train")
    expected_dataset = plan.get("dataset", {})
    planned_contract = expected_dataset.get("contract")
    if not isinstance(planned_contract, dict) or expected_dataset.get("sha256") != _json_sha256(planned_contract):
        raise ValueError("Surgery plan dataset checksum is inconsistent")
    if planned_contract.get("split") != "train":
        raise ValueError("Only surgery plans diagnosed on the train split can modify training")
    if expected_dataset.get("sha256") != _json_sha256(contract):
        raise ValueError("Surgery plan dataset contract does not match this training split")
    if plan.get("selection", {}).get("selected_count") != len(plan["selected_samples"]):
        raise ValueError("Surgery plan selected_count is inconsistent")

    indices = []
    for row in plan["selected_samples"]:
        index = row.get("index") if isinstance(row, dict) else None
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("Every selected surgery sample needs an integer index")
        indices.append(index)
    if not indices:
        raise ValueError("Surgery plan selected no samples; refusing an empty intervention")
    if len(set(indices)) != len(indices):
        raise ValueError("Surgery plan contains duplicate sample indices")
    original_size = len(data.train.dataset)
    if min(indices) < 0 or max(indices) >= original_size:
        raise ValueError("Surgery plan contains an out-of-range sample index")
    drop_fraction = len(indices) / original_size
    if drop_fraction > surgery.max_drop_fraction:
        raise ValueError(
            f"Surgery would drop {drop_fraction:.1%}, above max_drop_fraction={surgery.max_drop_fraction:.1%}")

    dropped = set(indices)
    kept = [index for index in range(original_size) if index not in dropped]
    loader = DataLoader(Subset(data.train.dataset, kept), batch_size=config.train.batch_size,
                        shuffle=True, num_workers=config.train.workers, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(config.train.seed))
    plan_path = Path(surgery.plan)
    info = {
        "enabled": True,
        "action": surgery.action,
        "plan": str(plan_path.resolve()),
        "plan_sha256": fingerprint(plan_path),
        "diagnostic_checkpoint_sha256": plan.get("model", {}).get("checkpoint_sha256"),
        "score": plan.get("selection", {}).get("score"),
        "strategy": plan.get("selection", {}).get("strategy"),
        "original_train_size": original_size,
        "dropped_samples": len(indices),
        "drop_fraction": drop_fraction,
        "final_train_size": len(kept),
    }
    provenance = dict(data.provenance)
    provenance["split_sizes"] = {**provenance.get("split_sizes", {}), "train": len(kept)}
    provenance["surgery"] = info
    return DataBundle(loader, data.val, data.classes, data.test, provenance), info
