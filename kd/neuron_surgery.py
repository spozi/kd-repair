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
from .models import CifarCNN, ModelOutput, ResNetAdapter, VisionModel
from .runtime import (autocast_context, loader_performance_kwargs, make_grad_scaler,
                      move_images, move_labels, prepare_model, resolve_precision)


FORMAT_VERSION = 1
LOCALIZATION_STAGES = ("stage2", "stage3")
TAIL_CLASSES = (5, 6, 7, 8, 9)


def localization_stages(model: VisionModel) -> tuple[str, ...]:
    """Return the bounded late-stage search space for a supported teacher."""
    if isinstance(model, CifarCNN):
        return LOCALIZATION_STAGES
    if isinstance(model, ResNetAdapter):
        return ("stage3", "stage4")
    raise ValueError("Neuron surgery supports CifarCNN and torchvision ResNet adapters")


def _stage_module(model: VisionModel, stage: str) -> nn.Module:
    if stage not in localization_stages(model):
        raise ValueError(f"Stage {stage!r} is not repairable for {type(model).__name__}")
    if isinstance(model, CifarCNN):
        return model.stages[stage]
    return getattr(model.backbone, f"layer{int(stage.removeprefix('stage'))}")


def collect_teacher_diagnostics(model: VisionModel, loader,
                                device: torch.device, *, precision="float32",
                                channels_last=True) -> tuple[list[dict], np.ndarray]:
    """Collect canonical predictions and normalized final-stage embeddings in dataset order."""
    embedding_stage = localization_stages(model)[-1]
    previous_mode = model.training
    model.eval()
    records: list[dict] = []
    embeddings = []
    offset = 0
    try:
        with torch.inference_mode():
            for images, labels in loader:
                labels_device = move_labels(labels, device)
                images = move_images(images, device, channels_last)
                with autocast_context(device, precision):
                    output = model(images, return_features=True)
                logits = output.logits.float()
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Non-finite logits in neuron-surgery diagnostics")
                log_probability = F.log_softmax(logits, dim=1)
                probability = log_probability.exp()
                confidence, prediction = probability.max(1)
                nll = -log_probability.gather(1, labels_device[:, None]).squeeze(1)
                embedding = F.adaptive_avg_pool2d(
                    output.features[embedding_stage].float(), 1).flatten(1)
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
                                per_class_cap: int = 64, seed: int = 2026,
                                allow_sparse_fallback: bool = False) -> list[int]:
    """Choose an equal-size, correctly classified preservation set across usable classes."""
    if per_class_cap < 1:
        raise ValueError("per_class_cap must be positive")
    excluded = set(target_indices)
    labels = sorted({int(row["true_label"]) for row in records})
    pools = {label: [int(row["index"]) for row in records
                     if row["true_label"] == label and row["correct"] and row["index"] not in excluded]
             for label in labels}
    if allow_sparse_fallback:
        pools = {label: pool for label, pool in pools.items() if pool}
    if not pools:
        raise ValueError("No correctly classified preservation samples are available")
    count = min(per_class_cap, *(len(pool) for pool in pools.values()))
    if count < 1:
        raise ValueError("Every class needs a correctly classified preservation sample")
    rng = np.random.default_rng(seed)
    chosen = []
    for label in sorted(pools):
        values = np.asarray(sorted(pools[label]), dtype=np.int64)
        chosen.extend(rng.permutation(values)[:count].tolist())
    return sorted(chosen)


def build_companion_records(records: list[dict], embeddings: np.ndarray, targets: list[dict],
                            *, companions: int = 5,
                            allow_sparse_fallback: bool = False) -> list[dict]:
    """Find deterministic same-class correct nearest neighbours for each target."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(records):
        raise ValueError("Embeddings must be [samples, features] and align with diagnostics")
    if companions < 1:
        raise ValueError("companions must be positive")
    target_indices = {int(row["index"]) for row in targets}
    by_label: dict[int, list[int]] = {}
    all_by_label: dict[int, list[int]] = {}
    for row in records:
        all_by_label.setdefault(int(row["true_label"]), []).append(int(row["index"]))
        if row["correct"] and row["index"] not in target_indices:
            by_label.setdefault(int(row["true_label"]), []).append(int(row["index"]))
    result = []
    for target in targets:
        index, label = int(target["index"]), int(target["true_label"])
        candidate_values = sorted(by_label.get(label, []))
        policy = "correct_non_target"
        gradient_only = False
        if len(candidate_values) < companions and allow_sparse_fallback:
            candidate_values = sorted(value for value in all_by_label.get(label, [])
                                      if value != index)
            policy = "same_class_fallback"
        if not candidate_values and allow_sparse_fallback:
            candidate_values = [index]
            policy = "self_gradient_fallback"
            gradient_only = True
        candidates = np.asarray(candidate_values, dtype=np.int64)
        if len(candidates) < companions and not allow_sparse_fallback:
            raise ValueError(f"Class {label} has fewer than {companions} eligible companions")
        similarities = embeddings[candidates] @ embeddings[index]
        order = np.lexsort((candidates, -similarities))
        if len(order) < companions:
            order = np.resize(order, companions)
        else:
            order = order[:companions]
        neighbours = []
        for position in order:
            candidate_index = int(candidates[position])
            source = records[candidate_index]
            neighbours.append({"index": candidate_index,
                               "cosine_distance": float(1.0 - similarities[position]),
                               "true_label": int(source["true_label"]),
                               "predicted_label": int(source["predicted_label"]),
                               "correct": bool(source["correct"]),
                               "confidence": float(source["confidence"])})
        result.append({"target": dict(target), "companions": neighbours,
                       "companion_policy": policy, "gradient_only": gradient_only})
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
        return (target_image, int(target_label), target_index, torch.stack(images),
                bool(record.get("gradient_only", False)))


def differential_channel_scores(model: VisionModel, dataset: Dataset,
                                companion_records: list[dict], device: torch.device, *,
                                batch_size: int = 16, stages: tuple[str, ...] | None = None,
                                score_mode: str = "differential") -> dict[str, np.ndarray]:
    """Score target channels with differential or gradient-only attribution."""
    stages = localization_stages(model) if stages is None else tuple(stages)
    if not stages or any(stage not in localization_stages(model) for stage in stages):
        raise ValueError(f"Localization stages must be drawn from {localization_stages(model)}")
    if score_mode not in {"differential", "gradient_only"}:
        raise ValueError("score_mode must be differential or gradient_only")
    loader = DataLoader(CompanionDataset(dataset, companion_records), batch_size=batch_size,
                        shuffle=False, num_workers=0,
                        **loader_performance_kwargs(str(device), 0))
    previous_mode = model.training
    requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    model.requires_grad_(False).eval()
    collected = {stage: [] for stage in stages}
    labels_seen, indices_seen = [], []
    try:
        for target_images, labels, target_indices, companion_images, gradient_only in loader:
            target_images = move_images(target_images, device).requires_grad_(True)
            labels_device = move_labels(labels, device)
            target_output = model(target_images, return_features=True)
            for stage in stages:
                target_output.features[stage].retain_grad()
            competing = target_output.logits.float().clone()
            competing.scatter_(1, labels_device[:, None], -torch.inf)
            margin = competing.max(1).values - target_output.logits.float().gather(
                1, labels_device[:, None]).squeeze(1)
            margin.sum().backward()
            batch, count = companion_images.shape[:2]
            companion_output = None
            if score_mode == "differential":
                with torch.no_grad():
                    companion_batch = move_images(companion_images.flatten(0, 1), device)
                    companion_output = model(companion_batch, return_features=True)
            for stage in stages:
                target_feature = target_output.features[stage].detach()
                sensitivity = target_output.features[stage].grad.abs().mean((2, 3))
                if companion_output is None:
                    score = sensitivity
                else:
                    companion_feature = companion_output.features[stage].reshape(
                        batch, count, *target_feature.shape[1:])
                    difference = (target_feature[:, None] - companion_feature).abs().mean((1, 3, 4))
                    score = difference * sensitivity
                    score = torch.where(gradient_only.to(device)[:, None], sensitivity, score)
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
                              top_fraction: float = 0.25, stability_threshold: float = 0.7,
                              stages: tuple[str, ...] | None = None) -> dict:
    """Aggregate per-target scores with equal class weight and bootstrap stability."""
    labels = np.asarray(scores["labels"], dtype=np.int64)
    if stages is None:
        stages = tuple(stage for stage, values in scores.items()
                       if stage not in {"labels", "target_indices"}
                       and np.asarray(values).ndim == 2)
    else:
        stages = tuple(stages)
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
def ablate_channel(model: VisionModel, stage: str, channel: int):
    """Temporarily zero one stage output channel and always remove the hook."""
    if stage not in localization_stages(model):
        raise ValueError(f"Ablation stages must be drawn from {localization_stages(model)}")
    if not 0 <= channel < model.feature_channels[stage]:
        raise ValueError(f"Channel {channel} is outside {stage}")

    def hook(_module, _inputs, output):
        changed = output.clone()
        changed[:, channel] = 0
        return changed

    handle = _stage_module(model, stage).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def measure_repair_set(model: VisionModel, loader, device: torch.device,
                       balanced_classes: Iterable[int] | None = None) -> dict:
    """Measure accuracy, NLL, and an equally weighted true-versus-rival margin."""
    previous_mode = model.training
    model.eval()
    count = 0
    correct = torch.zeros((), dtype=torch.long, device=device)
    nll_sum = torch.zeros((), device=device)
    labels_seen, margins_seen = [], []
    try:
        for images, labels in loader:
            labels_device = move_labels(labels, device)
            logits = model(move_images(images, device)).logits.float()
            competitors = logits.clone()
            competitors.scatter_(1, labels_device[:, None], -torch.inf)
            true_logits = logits.gather(1, labels_device[:, None]).squeeze(1)
            margin = true_logits - competitors.max(1).values
            predictions = logits.argmax(1)
            count += labels.numel()
            correct += predictions.eq(labels_device).sum()
            nll_sum += F.cross_entropy(logits, labels_device, reduction="sum")
            labels_seen.append(labels_device)
            margins_seen.append(margin)
    finally:
        model.train(previous_mode)
    if not count:
        raise ValueError("Cannot measure an empty repair set")
    labels_array = torch.cat(labels_seen).cpu().numpy()
    margin_array = torch.cat(margins_seen).cpu().numpy()
    margins = {int(label): margin_array[labels_array == label]
               for label in np.unique(labels_array)}
    classes = tuple(sorted(margins) if balanced_classes is None else balanced_classes)
    if any(label not in margins for label in classes):
        raise ValueError("Every balanced class must occur in the measured set")
    return {"samples": count, "accuracy": correct.item() / count,
            "nll": nll_sum.item() / count,
            "class_balanced_margin": float(np.mean([np.mean(margins[label]) for label in classes]))}


def causal_channel_validation(model: VisionModel, target_loader, preservation_loader,
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


def _empty_parameter_masks(model: nn.Module) -> dict[str, Tensor]:
    return {name: torch.zeros_like(parameter, dtype=torch.bool)
            for name, parameter in model.named_parameters()}


def _cifar_parameter_masks(model: CifarCNN, channels: Iterable[dict]) -> dict[str, Tensor]:
    masks = _empty_parameter_masks(model)
    for row in channels:
        stage, channel = str(row["stage"]), int(row["channel"])
        if stage not in localization_stages(model) or not 0 <= channel < model.feature_channels[stage]:
            raise ValueError(f"Invalid repair channel {stage}:{channel}")
        masks[f"stages.{stage}.3.weight"][channel] = True
        masks[f"stages.{stage}.4.weight"][channel] = True
        masks[f"stages.{stage}.4.bias"][channel] = True
        if stage == "stage2":
            masks["stages.stage3.0.weight"][:, channel] = True
        else:
            masks["classifier.weight"][:, channel * 4:(channel + 1) * 4] = True
    return masks


def _resnet_parameter_masks(model: ResNetAdapter, channels: Iterable[dict]) -> dict[str, Tensor]:
    """Map a late ResNet stage channel through its terminal block and consumers."""
    masks = _empty_parameter_masks(model)
    for row in channels:
        stage, channel = str(row["stage"]), int(row["channel"])
        if stage not in localization_stages(model) or not 0 <= channel < model.feature_channels[stage]:
            raise ValueError(f"Invalid repair channel {stage}:{channel}")
        number = int(stage.removeprefix("stage"))
        layer = getattr(model.backbone, f"layer{number}")
        block_index = len(layer) - 1
        block = layer[block_index]
        suffix = "3" if hasattr(block, "conv3") else "2"
        prefix = f"backbone.layer{number}.{block_index}"
        masks[f"{prefix}.conv{suffix}.weight"][channel] = True
        masks[f"{prefix}.bn{suffix}.weight"][channel] = True
        masks[f"{prefix}.bn{suffix}.bias"][channel] = True
        if number == 4:
            masks["backbone.fc.weight"][:, channel] = True
            continue
        consumer = f"backbone.layer{number + 1}.0"
        masks[f"{consumer}.conv1.weight"][:, channel] = True
        downsample_name = f"{consumer}.downsample.0.weight"
        if downsample_name in masks:
            masks[downsample_name][:, channel] = True
    return masks


def channel_parameter_masks(model: VisionModel, channels: Iterable[dict]) -> dict[str, Tensor]:
    """Map channels to producer rows, BN affine entries, and direct consumers."""
    channels = list(channels)
    if isinstance(model, CifarCNN):
        masks = _cifar_parameter_masks(model, channels)
    elif isinstance(model, ResNetAdapter):
        masks = _resnet_parameter_masks(model, channels)
    else:
        raise ValueError("Masked channel repair supports CifarCNN and ResNet adapters")
    if not any(mask.any().item() for mask in masks.values()):
        raise ValueError("At least one channel must be selected for repair")
    return masks


def cifar_channel_parameter_masks(model: CifarCNN, channels: Iterable[dict]) -> dict[str, Tensor]:
    """Backward-compatible strict CifarCNN channel map."""
    if not isinstance(model, CifarCNN):
        raise ValueError("Expected a CifarCNN repair teacher")
    return channel_parameter_masks(model, channels)


def resnet_channel_parameter_masks(model: ResNetAdapter,
                                   channels: Iterable[dict]) -> dict[str, Tensor]:
    """Strict torchvision ResNet late-stage channel map."""
    if not isinstance(model, ResNetAdapter):
        raise ValueError("Expected a ResNetAdapter repair teacher")
    return channel_parameter_masks(model, channels)


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
                            stages: tuple[str, ...] | None = None) -> Tensor:
    if stages is None:
        stages = tuple(stage for stage in candidate if stage in anchor)
    if not stages:
        raise ValueError("Feature preservation requires at least one shared stage")
    losses = []
    for stage in stages:
        losses.append((F.normalize(candidate[stage].float(), dim=1)
                       - F.normalize(anchor[stage].detach().float(), dim=1)).square().sum(1).mean())
    return torch.stack(losses).mean()


def repair_loss(candidate_target: ModelOutput, target_labels: Tensor,
                candidate_preservation: ModelOutput, anchor_preservation: ModelOutput,
                preservation_labels: Tensor, *, temperature: float = 4.0,
                kd_weight: float = 1.0, feature_weight: float = 0.25,
                preservation_ce_weight: float = 0.0,
                feature_stages: tuple[str, ...] | None = None) -> dict[str, Tensor]:
    if kd_weight < 0 or feature_weight < 0 or preservation_ce_weight < 0:
        raise ValueError("Repair preservation weights must be nonnegative")
    ce = F.cross_entropy(candidate_target.logits, target_labels)
    zero = ce.new_zeros(())
    preservation_ce = (F.cross_entropy(candidate_preservation.logits, preservation_labels)
                       if preservation_ce_weight else zero)
    kd = (LogitsKD(temperature)(candidate_preservation.logits, anchor_preservation.logits,
                                preservation_labels) if kd_weight else zero)
    features = (same_model_feature_loss(candidate_preservation.features,
                                        anchor_preservation.features, feature_stages)
                if feature_weight else zero)
    total = (ce + preservation_ce_weight * preservation_ce
             + kd_weight * kd + feature_weight * features)
    return {"total": total, "ce": ce, "preservation_ce": preservation_ce,
            "kd": kd, "features": features}


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


def train_repair_candidate(candidate: VisionModel, anchor: VisionModel, dataset: Dataset,
                           labels: np.ndarray, target_indices: list[int],
                           preservation_indices: list[int], channels: list[dict],
                           device: torch.device, *, epochs: int = 20, samples_per_epoch: int = 640,
                           batch_size: int = 64, learning_rate: float = 0.001,
                           momentum: float = 0.9, temperature: float = 4.0,
                           kd_weight: float = 1.0, feature_weight: float = 0.25,
                           preservation_ce_weight: float = 0.0,
                           feature_stages: tuple[str, ...] | None = None,
                           seed: int = 2026, precision: str = "auto",
                           channels_last: bool = True,
                           fail_fast: bool = True) -> tuple[list[dict], dict]:
    """Fine-tune only selected channel-connected weights under preservation losses."""
    if min(epochs, samples_per_epoch, batch_size) < 1:
        raise ValueError("Repair epochs, samples_per_epoch, and batch_size must be positive")
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) != len(dataset):
        raise ValueError("Repair labels must align with the dataset")
    torch.manual_seed(seed)
    prepare_model(candidate, device, channels_last).eval()
    prepare_model(anchor, device, channels_last).requires_grad_(False).eval()
    precision = resolve_precision(precision, device)
    if type(candidate) is not type(anchor) or candidate.feature_channels != anchor.feature_channels:
        raise ValueError("Candidate and anchor must use the same repair architecture")
    feature_stages = localization_stages(candidate) if feature_stages is None else tuple(feature_stages)
    masks = channel_parameter_masks(candidate, channels)
    before = {name: value.detach().cpu().clone() for name, value in candidate.state_dict().items()}
    target_sampler = WeightedRandomSampler(_balanced_weights(target_indices, labels), samples_per_epoch,
                                           replacement=True, generator=torch.Generator().manual_seed(seed))
    preserve_sampler = WeightedRandomSampler(_balanced_weights(preservation_indices, labels), samples_per_epoch,
                                             replacement=True,
                                             generator=torch.Generator().manual_seed(seed + 1))
    target_loader = DataLoader(Subset(dataset, target_indices), batch_size=batch_size,
                               sampler=target_sampler, num_workers=0,
                               **loader_performance_kwargs(str(device), 0))
    preservation_loader = DataLoader(Subset(dataset, preservation_indices), batch_size=batch_size,
                                     sampler=preserve_sampler, num_workers=0,
                                     **loader_performance_kwargs(str(device), 0))
    history = []
    with masked_parameters(candidate, masks) as parameters:
        optimizer = torch.optim.SGD(
            parameters, lr=learning_rate, momentum=momentum, weight_decay=0,
            fused=device.type == "cuda" and not fail_fast)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = make_grad_scaler(device, precision)
        for epoch in range(epochs):
            candidate.eval()
            totals = {name: torch.zeros((), device=device) for name in
                      ("total", "ce", "preservation_ce", "kd", "features")}
            count = 0
            for (target_images, target_labels), (preserve_images, preserve_labels) in zip(
                    target_loader, preservation_loader):
                target_images = move_images(target_images, device, channels_last)
                target_labels = move_labels(target_labels, device)
                preserve_images = move_images(preserve_images, device, channels_last)
                preserve_labels = move_labels(preserve_labels, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, precision):
                    target_output = candidate(target_images)
                    preserve_output = candidate(preserve_images, return_features=bool(feature_weight))
                    if kd_weight or feature_weight:
                        with torch.no_grad():
                            anchor_output = anchor(preserve_images, return_features=bool(feature_weight))
                    else:
                        anchor_output = preserve_output
                    losses = repair_loss(target_output, target_labels, preserve_output, anchor_output,
                                         preserve_labels, temperature=temperature,
                                         kd_weight=kd_weight, feature_weight=feature_weight,
                                         preservation_ce_weight=preservation_ce_weight,
                                         feature_stages=feature_stages)
                if fail_fast and not torch.isfinite(losses["total"]):
                    raise FloatingPointError("Non-finite neuron-repair loss")
                if scaler.is_enabled():
                    scaler.scale(losses["total"]).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["total"].backward()
                    if fail_fast:
                        nn.utils.clip_grad_norm_(parameters, float("inf"), error_if_nonfinite=True)
                    optimizer.step()
                count += target_labels.numel()
                for name, value in losses.items():
                    totals[name] += value.detach() * target_labels.numel()
            finite = torch.stack([torch.isfinite(value) for value in totals.values()]).all()
            parameter_finite = torch.stack([
                torch.isfinite(parameter).all() for parameter in parameters]).all()
            if not bool((finite & parameter_finite).cpu()):
                raise FloatingPointError(f"Non-finite neuron-repair state at epoch {epoch + 1}")
            history.append({"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"],
                            **{name: value.item() / count for name, value in totals.items()},
                            "precision": precision,
                            "grad_scaler_enabled": scaler.is_enabled()})
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
