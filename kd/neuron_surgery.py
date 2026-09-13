"""AI-Lancet-inspired channel localization and constrained CIFAR-CNN repair."""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from .losses import LogitsKD
from .models import CifarCNN, ModelOutput


FORMAT_VERSION = 1
LOCALIZATION_STAGES = ("stage2", "stage3")
TAIL_CLASSES = (5, 6, 7, 8, 9)


def collect_teacher_diagnostics(model: CifarCNN, loader, device: torch.device) -> tuple[list[dict], np.ndarray]:
    """Collect canonical predictions and normalized stage-3 embeddings in dataset order."""
    if not isinstance(model, CifarCNN):
        raise ValueError("Neuron surgery version 1 supports CifarCNN only")
    previous_mode = model.training
    model.eval()
    records: list[dict] = []
    embeddings = []
    offset = 0
    try:
        with torch.inference_mode():
            for images, labels in loader:
                labels_device = labels.to(device)
                output = model(images.to(device), return_features=True)
                logits = output.logits.float()
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Non-finite logits in neuron-surgery diagnostics")
                log_probability = F.log_softmax(logits, dim=1)
                probability = log_probability.exp()
                confidence, prediction = probability.max(1)
                nll = -log_probability.gather(1, labels_device[:, None]).squeeze(1)
                embedding = F.adaptive_avg_pool2d(output.features["stage3"].float(), 1).flatten(1)
                embeddings.append(F.normalize(embedding, dim=1).cpu().numpy())
                rows = zip(labels.tolist(), prediction.cpu().tolist(), confidence.cpu().tolist(),
                           nll.cpu().tolist())
                for row, (label, predicted, conf, loss) in enumerate(rows):
                    records.append({"index": offset + row, "true_label": int(label),
                                    "predicted_label": int(predicted), "correct": predicted == label,
                                    "confidence": float(conf), "nll": float(loss)})
                offset += labels.shape[0]
    finally:
        model.train(previous_mode)
    if offset != len(loader.dataset):
        raise ValueError("Diagnostic loader must visit every sample exactly once")
    return records, np.concatenate(embeddings).astype(np.float32, copy=False)


def select_hard_tail_targets(records: list[dict], target_classes: Iterable[int] = TAIL_CLASSES,
                             fraction: float = 0.2) -> list[dict]:
    """Keep every mistake, then fill each class quota with its hardest correct rows."""
    if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Hard-target fraction must be finite and in (0, 1]")
    targets = []
    seen = set()
    for label in tuple(target_classes):
        rows = [dict(row) for row in records if row["true_label"] == label]
        if not rows:
            raise ValueError(f"Target class {label} has no samples")
        quota = math.ceil(len(rows) * fraction)
        mistakes = sorted((row for row in rows if not row["correct"]),
                           key=lambda row: (-row["nll"], row["index"]))
        correct = sorted((row for row in rows if row["correct"]),
                         key=lambda row: (-row["nll"], row["index"]))
        chosen = mistakes + correct[:max(0, quota - len(mistakes))]
        for row in chosen:
            if row["index"] in seen:
                raise ValueError("Diagnostic indices must be unique")
            seen.add(row["index"])
            row["selection_reason"] = "misclassified" if not row["correct"] else "high_loss_correct"
            targets.append(row)
    return sorted(targets, key=lambda row: (row["true_label"], row["index"]))


def select_preservation_indices(records: list[dict], target_indices: Iterable[int], *,
                                per_class_cap: int = 64, seed: int = 2026) -> list[int]:
    """Choose an equal-size, correctly classified preservation set across all classes."""
    if per_class_cap < 1:
        raise ValueError("per_class_cap must be positive")
    excluded = set(target_indices)
    labels = sorted({int(row["true_label"]) for row in records})
    pools = {label: [int(row["index"]) for row in records
                     if row["true_label"] == label and row["correct"] and row["index"] not in excluded]
             for label in labels}
    count = min(per_class_cap, *(len(pool) for pool in pools.values()))
    if count < 1:
        raise ValueError("Every class needs a correctly classified preservation sample")
    rng = np.random.default_rng(seed)
    chosen = []
    for label in labels:
        values = np.asarray(sorted(pools[label]), dtype=np.int64)
        chosen.extend(rng.permutation(values)[:count].tolist())
    return sorted(chosen)


def build_companion_records(records: list[dict], embeddings: np.ndarray, targets: list[dict],
                            *, companions: int = 5) -> list[dict]:
    """Find deterministic same-class correct nearest neighbours for each target."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise ValueError("Embeddings must be [samples, features] and align with diagnostics")
    if companions < 1:
        raise ValueError("companions must be positive")
    target_indices = {int(row["index"]) for row in targets}
    by_label: dict[int, list[int]] = {}
    for row in records:
        if row["correct"] and row["index"] not in target_indices:
            by_label.setdefault(int(row["true_label"]), []).append(int(row["index"]))
    result = []
    for target in targets:
        index, label = int(target["index"]), int(target["true_label"])
        candidates = np.asarray(sorted(by_label.get(label, [])), dtype=np.int64)
        if len(candidates) < companions:
            raise ValueError(f"Class {label} has fewer than {companions} eligible companions")
        similarities = embeddings[candidates] @ embeddings[index]
        order = np.lexsort((candidates, -similarities))[:companions]
        neighbours = []
        for position in order:
            candidate_index = int(candidates[position])
            source = records[candidate_index]
            neighbours.append({"index": candidate_index,
                               "cosine_distance": float(1.0 - similarities[position]),
                               "true_label": int(source["true_label"]),
                               "predicted_label": int(source["predicted_label"]),
                               "confidence": float(source["confidence"])})
        result.append({"target": dict(target), "companions": neighbours})
    return result


class CompanionDataset(Dataset):
    """Materialize one target and its fixed companion image group."""

    def __init__(self, dataset: Dataset, companion_records: list[dict]):
        self.dataset = dataset
        self.records = companion_records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, position):
        record = self.records[position]
        target_index = int(record["target"]["index"])
        target_image, target_label = self.dataset[target_index]
        images = [self.dataset[int(row["index"])][0] for row in record["companions"]]
        return target_image, int(target_label), target_index, torch.stack(images)


def differential_channel_scores(model: CifarCNN, dataset: Dataset, companion_records: list[dict],
                                device: torch.device, *, batch_size: int = 16,
                                stages: tuple[str, ...] = LOCALIZATION_STAGES) -> dict[str, np.ndarray]:
    """Score target channels by companion difference times error-margin gradient."""
    if not isinstance(model, CifarCNN):
        raise ValueError("Neuron surgery version 1 supports CifarCNN only")
    if any(stage not in LOCALIZATION_STAGES for stage in stages):
        raise ValueError(f"Localization stages must be drawn from {LOCALIZATION_STAGES}")
    loader = DataLoader(CompanionDataset(dataset, companion_records), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    previous_mode = model.training
    requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    model.requires_grad_(False).eval()
    collected = {stage: [] for stage in stages}
    labels_seen, indices_seen = [], []
    try:
        for target_images, labels, target_indices, companion_images in loader:
            target_images = target_images.to(device).requires_grad_(True)
            labels_device = labels.to(device)
            target_output = model(target_images, return_features=True)
            for stage in stages:
                target_output.features[stage].retain_grad()
            competing = target_output.logits.float().clone()
            competing.scatter_(1, labels_device[:, None], -torch.inf)
            margin = competing.max(1).values - target_output.logits.float().gather(
                1, labels_device[:, None]).squeeze(1)
            margin.sum().backward()
            batch, count = companion_images.shape[:2]
            with torch.no_grad():
                companion_output = model(companion_images.flatten(0, 1).to(device),
                                         return_features=True)
            for stage in stages:
                target_feature = target_output.features[stage].detach()
                companion_feature = companion_output.features[stage].reshape(
                    batch, count, *target_feature.shape[1:])
                difference = (target_feature[:, None] - companion_feature).abs().mean((1, 3, 4))
                sensitivity = target_output.features[stage].grad.abs().mean((2, 3))
                score = difference * sensitivity
                score = score / score.mean(1, keepdim=True).clamp_min(torch.finfo(score.dtype).eps)
                if not torch.isfinite(score).all():
                    raise FloatingPointError("Non-finite differential channel score")
                collected[stage].append(score.cpu().numpy())
            labels_seen.append(labels.numpy())
            indices_seen.append(target_indices.numpy())
    finally:
        for parameter, enabled in zip(model.parameters(), requires_grad):
            parameter.requires_grad_(enabled)
        model.train(previous_mode)
    return {"target_indices": np.concatenate(indices_seen).astype(np.int64),
            "labels": np.concatenate(labels_seen).astype(np.int64),
            **{stage: np.concatenate(values).astype(np.float32) for stage, values in collected.items()}}


def consensus_channel_ranking(scores: dict[str, np.ndarray], *, target_classes: Iterable[int] = TAIL_CLASSES,
                              repetitions: int = 20, seed: int = 2026,
                              top_fraction: float = 0.25, stability_threshold: float = 0.7) -> dict:
    """Aggregate per-target scores with equal class weight and bootstrap stability."""
    labels = np.asarray(scores["labels"], dtype=np.int64)
    stages = [stage for stage in LOCALIZATION_STAGES if stage in scores]
    if not stages or repetitions < 1 or not 0 < top_fraction <= 1 or not 0 <= stability_threshold <= 1:
        raise ValueError("Invalid consensus configuration")
    matrices = [np.asarray(scores[stage], dtype=np.float64) for stage in stages]
    if any(matrix.ndim != 2 or matrix.shape[0] != len(labels) for matrix in matrices):
        raise ValueError("Every stage score must be [targets, channels]")
    matrix = np.concatenate(matrices, axis=1)
    channels = [(stage, index) for stage, values in zip(stages, matrices)
                for index in range(values.shape[1])]
    classes = tuple(int(label) for label in target_classes)
    positions = {label: np.flatnonzero(labels == label) for label in classes}
    if any(not len(value) for value in positions.values()):
        raise ValueError("Every target class must be represented in consensus scores")

    def aggregate(sampled):
        per_class = np.stack([matrix[sampled[label]].mean(0) for label in classes])
        return per_class.mean(0), per_class

    consensus, class_scores = aggregate(positions)
    rng = np.random.default_rng(seed)
    votes = np.zeros(len(channels), dtype=np.int64)
    top_count = max(1, math.ceil(len(channels) * top_fraction))
    for _ in range(repetitions):
        sampled = {label: rng.choice(value, size=len(value), replace=True)
                   for label, value in positions.items()}
        bootstrap, _ = aggregate(sampled)
        order = np.lexsort((np.arange(len(channels)), -bootstrap))
        votes[order[:top_count]] += 1
    records = []
    for position, (stage, index) in enumerate(channels):
        stability = float(votes[position] / repetitions)
        records.append({"stage": stage, "channel": index,
                        "consensus_score": float(consensus[position]),
                        "bootstrap_stability": stability,
                        "eligible": stability >= stability_threshold,
                        "per_class_score": {str(label): float(class_scores[i, position])
                                            for i, label in enumerate(classes)}})
    records.sort(key=lambda row: (-row["consensus_score"], row["stage"], row["channel"]))
    return {"repetitions": repetitions, "seed": seed, "top_fraction": top_fraction,
            "stability_threshold": stability_threshold,
            "eligible_count": sum(row["eligible"] for row in records), "channels": records}


@contextmanager
def ablate_channel(model: CifarCNN, stage: str, channel: int):
    """Temporarily zero one stage output channel and always remove the hook."""
    if not isinstance(model, CifarCNN) or stage not in LOCALIZATION_STAGES:
        raise ValueError("Ablation supports CifarCNN stage2/stage3 only")
    if not 0 <= channel < model.feature_channels[stage]:
        raise ValueError(f"Channel {channel} is outside {stage}")

    def hook(_module, _inputs, output):
        changed = output.clone()
        changed[:, channel] = 0
        return changed

    handle = model.stages[stage].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def measure_repair_set(model: CifarCNN, loader, device: torch.device,
                       balanced_classes: Iterable[int] | None = None) -> dict:
    """Measure accuracy, NLL, and an equally weighted true-versus-rival margin."""
    previous_mode = model.training
    model.eval()
    count = correct = 0
    nll_sum = 0.0
    margins: dict[int, list[float]] = {}
    try:
        for images, labels in loader:
            labels_device = labels.to(device)
            logits = model(images.to(device)).logits.float()
            competitors = logits.clone()
            competitors.scatter_(1, labels_device[:, None], -torch.inf)
            true_logits = logits.gather(1, labels_device[:, None]).squeeze(1)
            margin = true_logits - competitors.max(1).values
            predictions = logits.argmax(1)
            count += labels.numel()
            correct += predictions.eq(labels_device).sum().item()
            nll_sum += F.cross_entropy(logits, labels_device, reduction="sum").item()
            for label, value in zip(labels.tolist(), margin.cpu().tolist()):
                margins.setdefault(int(label), []).append(float(value))
    finally:
        model.train(previous_mode)
    if not count:
        raise ValueError("Cannot measure an empty repair set")
    classes = tuple(sorted(margins) if balanced_classes is None else balanced_classes)
    if any(label not in margins for label in classes):
        raise ValueError("Every balanced class must occur in the measured set")
    return {"samples": count, "accuracy": correct / count, "nll": nll_sum / count,
            "class_balanced_margin": float(np.mean([np.mean(margins[label]) for label in classes]))}


def causal_channel_validation(model: CifarCNN, target_loader, preservation_loader,
                              consensus: dict, device: torch.device, *,
                              target_classes: Iterable[int] = TAIL_CLASSES,
                              accuracy_guardrail: float = 0.005) -> dict:
    """Retain consensus channels whose ablation helps targets without broad damage."""
    baseline_target = measure_repair_set(model, target_loader, device, target_classes)
    baseline_preservation = measure_repair_set(model, preservation_loader, device)
    rows = []
    for channel in (row for row in consensus["channels"] if row["eligible"]):
        with ablate_channel(model, channel["stage"], int(channel["channel"])):
            target = measure_repair_set(model, target_loader, device, target_classes)
            preservation = measure_repair_set(model, preservation_loader, device)
        margin_delta = target["class_balanced_margin"] - baseline_target["class_balanced_margin"]
        accuracy_delta = preservation["accuracy"] - baseline_preservation["accuracy"]
        nll_delta = preservation["nll"] - baseline_preservation["nll"]
        rows.append({"stage": channel["stage"], "channel": int(channel["channel"]),
                     "consensus_score": channel["consensus_score"],
                     "bootstrap_stability": channel["bootstrap_stability"],
                     "target_margin_delta": margin_delta,
                     "preservation_accuracy_delta": accuracy_delta,
                     "preservation_nll_delta": nll_delta,
                     "retained": margin_delta > 0 and accuracy_delta >= -accuracy_guardrail})
    retained = sorted((row for row in rows if row["retained"]),
                      key=lambda row: (-row["target_margin_delta"], row["preservation_nll_delta"],
                                       row["stage"], row["channel"]))
    return {"accuracy_guardrail": accuracy_guardrail, "baseline_target": baseline_target,
            "baseline_preservation": baseline_preservation, "channels": rows,
            "ranking": [{"stage": row["stage"], "channel": row["channel"]} for row in retained]}


def cifar_channel_parameter_masks(model: CifarCNN, channels: Iterable[dict]) -> dict[str, Tensor]:
    """Map stage channels to producer, BN-affine, and downstream consumer weights."""
    if not isinstance(model, CifarCNN):
        raise ValueError("Masked channel repair supports CifarCNN only")
    masks = {name: torch.zeros_like(parameter, dtype=torch.bool)
             for name, parameter in model.named_parameters()}
    for row in channels:
        stage, channel = str(row["stage"]), int(row["channel"])
        if stage not in LOCALIZATION_STAGES or not 0 <= channel < model.feature_channels[stage]:
            raise ValueError(f"Invalid repair channel {stage}:{channel}")
        masks[f"stages.{stage}.3.weight"][channel] = True
        masks[f"stages.{stage}.4.weight"][channel] = True
        masks[f"stages.{stage}.4.bias"][channel] = True
        if stage == "stage2":
            masks["stages.stage3.0.weight"][:, channel] = True
        else:
            masks["classifier.weight"][:, channel * 4:(channel + 1) * 4] = True
    if not any(mask.any().item() for mask in masks.values()):
        raise ValueError("At least one channel must be selected for repair")
    return masks


@contextmanager
def masked_parameters(model: nn.Module, masks: dict[str, Tensor]):
    """Expose masked tensors to an optimizer while zeroing every other gradient entry."""
    original = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    handles = []
    trainable = []
    try:
        for name, parameter in model.named_parameters():
            mask = masks[name].to(parameter.device)
            enabled = bool(mask.any().item())
            parameter.requires_grad_(enabled)
            if enabled:
                handles.append(parameter.register_hook(lambda gradient, keep=mask: gradient * keep))
                trainable.append(parameter)
        yield trainable
    finally:
        for handle in handles:
            handle.remove()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original[name])


def same_model_feature_loss(candidate: dict[str, Tensor], anchor: dict[str, Tensor],
                            stages: tuple[str, ...] = LOCALIZATION_STAGES) -> Tensor:
    losses = []
    for stage in stages:
        losses.append((F.normalize(candidate[stage].float(), dim=1)
                       - F.normalize(anchor[stage].detach().float(), dim=1)).square().sum(1).mean())
    return torch.stack(losses).mean()


def repair_loss(candidate_target: ModelOutput, target_labels: Tensor,
                candidate_preservation: ModelOutput, anchor_preservation: ModelOutput,
                preservation_labels: Tensor, *, temperature: float = 4.0,
                feature_weight: float = 0.25) -> dict[str, Tensor]:
    ce = F.cross_entropy(candidate_target.logits, target_labels)
    kd = LogitsKD(temperature)(candidate_preservation.logits, anchor_preservation.logits,
                               preservation_labels)
    features = same_model_feature_loss(candidate_preservation.features,
                                       anchor_preservation.features)
    return {"total": ce + kd + feature_weight * features, "ce": ce, "kd": kd,
            "features": features}


def assert_only_masked_changes(model: nn.Module, before: dict[str, Tensor],
                               masks: dict[str, Tensor]) -> dict:
    """Fail if an unselected parameter coordinate or any buffer changed."""
    parameter_names = dict(model.named_parameters())
    selected = changed = 0
    for name, value in model.state_dict().items():
        prior = before[name].to(value.device)
        difference = value.ne(prior)
        if name in parameter_names:
            mask = masks[name].to(value.device)
            selected += int(mask.sum().item())
            if bool((difference & ~mask).any().item()):
                raise RuntimeError(f"Unselected parameter coordinates changed: {name}")
            changed += int((difference & mask).sum().item())
        elif not torch.equal(value, prior):
            raise RuntimeError(f"Frozen model buffer changed: {name}")
    return {"selected_parameter_coordinates": selected, "changed_parameter_coordinates": changed,
            "all_unselected_coordinates_unchanged": True, "all_buffers_unchanged": True}


def _balanced_weights(indices: list[int], labels: np.ndarray) -> Tensor:
    selected_labels = labels[np.asarray(indices, dtype=np.int64)]
    counts = np.bincount(selected_labels, minlength=int(labels.max()) + 1)
    return torch.tensor([1.0 / counts[label] for label in selected_labels], dtype=torch.double)


def train_repair_candidate(candidate: CifarCNN, anchor: CifarCNN, dataset: Dataset,
                           labels: np.ndarray, target_indices: list[int],
                           preservation_indices: list[int], channels: list[dict],
                           device: torch.device, *, epochs: int = 20, samples_per_epoch: int = 640,
                           batch_size: int = 64, learning_rate: float = 0.001,
                           momentum: float = 0.9, temperature: float = 4.0,
                           feature_weight: float = 0.25, seed: int = 2026) -> tuple[list[dict], dict]:
    """Fine-tune only selected channel-connected weights under anchor-teacher KD."""
    if min(epochs, samples_per_epoch, batch_size) < 1:
        raise ValueError("Repair epochs, samples_per_epoch, and batch_size must be positive")
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) != len(dataset):
        raise ValueError("Repair labels must align with the dataset")
    torch.manual_seed(seed)
    candidate.to(device).eval()
    anchor.to(device).requires_grad_(False).eval()
    masks = cifar_channel_parameter_masks(candidate, channels)
    before = {name: value.detach().cpu().clone() for name, value in candidate.state_dict().items()}
    target_sampler = WeightedRandomSampler(_balanced_weights(target_indices, labels), samples_per_epoch,
                                           replacement=True, generator=torch.Generator().manual_seed(seed))
    preserve_sampler = WeightedRandomSampler(_balanced_weights(preservation_indices, labels), samples_per_epoch,
                                             replacement=True,
                                             generator=torch.Generator().manual_seed(seed + 1))
    target_loader = DataLoader(Subset(dataset, target_indices), batch_size=batch_size,
                               sampler=target_sampler, num_workers=0)
    preservation_loader = DataLoader(Subset(dataset, preservation_indices), batch_size=batch_size,
                                     sampler=preserve_sampler, num_workers=0)
    history = []
    with masked_parameters(candidate, masks) as parameters:
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        for epoch in range(epochs):
            candidate.eval()
            totals = {"total": 0.0, "ce": 0.0, "kd": 0.0, "features": 0.0}
            count = 0
            for (target_images, target_labels), (preserve_images, preserve_labels) in zip(
                    target_loader, preservation_loader):
                target_images, target_labels = target_images.to(device), target_labels.to(device)
                preserve_images, preserve_labels = preserve_images.to(device), preserve_labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                target_output = candidate(target_images)
                preserve_output = candidate(preserve_images, return_features=True)
                with torch.no_grad():
                    anchor_output = anchor(preserve_images, return_features=True)
                losses = repair_loss(target_output, target_labels, preserve_output, anchor_output,
                                     preserve_labels, temperature=temperature,
                                     feature_weight=feature_weight)
                if not torch.isfinite(losses["total"]):
                    raise FloatingPointError("Non-finite neuron-repair loss")
                losses["total"].backward()
                nn.utils.clip_grad_norm_(parameters, float("inf"), error_if_nonfinite=True)
                optimizer.step()
                count += target_labels.numel()
                for name, value in losses.items():
                    totals[name] += value.detach().item() * target_labels.numel()
            history.append({"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"],
                            **{name: value / count for name, value in totals.items()}})
            scheduler.step()
    candidate.requires_grad_(False).eval()
    verification = assert_only_masked_changes(candidate, before, masks)
    return history, verification


def validate_repair_provenance(state: dict, *, base_checkpoint_sha256: str,
                               localization_sha256: str) -> dict:
    repair = state.get("repair")
    if state.get("kind") != "inference" or not isinstance(repair, dict):
        raise ValueError("Checkpoint is not a repaired inference teacher")
    if repair.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported neuron-repair checkpoint format")
    if repair.get("base_checkpoint_sha256") != base_checkpoint_sha256:
        raise ValueError("Repaired teacher base checkpoint changed")
    if repair.get("localization_sha256") != localization_sha256:
        raise ValueError("Repaired teacher localization plan changed")
    return repair
