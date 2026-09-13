"""Training orchestration depends on model/loss contracts, not architectures."""

from pathlib import Path
import random
import statistics
import time
import warnings

import numpy as np
import torch
from torch import nn

from .checkpoints import (capture_rng, fingerprint, load_model_checkpoint, metadata,
                          restore_rng, save_checkpoint, write_json)
from .checkpoints import tensor_state_fingerprint
from .config import ExperimentConfig, from_dict
from .calibration import CalibrationController, MMCELoss
from .adaptive_focal import AdaptiveFocalLoss
from .cpc import PairwiseCalibrationLoss
from .data import build_data
from .losses import DistillationObjective, StageFeatureLoss
from .metrics import benchmark, check_budget, evaluate, synchronize
from .models import VisionModel, create_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    return torch.device(requested)


class DistillationSystem(nn.Module):
    """Composition root for training; freezes the teacher even after train()."""

    def __init__(self, student: VisionModel, teacher: VisionModel | None,
                 objective: DistillationObjective):
        super().__init__()
        self.student, self.teacher, self.objective = student, teacher, objective
        if teacher is not None:
            teacher.requires_grad_(False)
            teacher.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    def forward(self, images, labels, epoch: int):
        features = self.objective.feature_loss is not None
        student = self.student(images, return_features=features)
        teacher = None
        if self.teacher is not None:
            with torch.no_grad():
                teacher = self.teacher(images, return_features=features)
        return student, self.objective(student, teacher, labels, epoch)


class Trainer:
    def __init__(self, system: DistillationSystem, optimizer, device: torch.device):
        self.system, self.optimizer, self.device = system, optimizer, device

    def train_epoch(self, loader, epoch: int) -> dict:
        self.system.train()
        sums = {"total": 0.0, "ce": 0.0, "supervised": 0.0, "response": 0.0, "features": 0.0, "calibration": 0.0}
        if self.system.objective.cpc_loss is not None:
            sums.update(cpc_discrimination=0.0, cpc_exclusion=0.0, cpc_weighted=0.0)
        count = correct = 0
        synchronize(self.device)
        start = time.perf_counter()
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            output, losses = self.system(images, labels, epoch)
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}")
            losses["total"].backward()
            # Fail before updating weights if gradients overflow.
            nn.utils.clip_grad_norm_([p for p in self.system.parameters() if p.requires_grad],
                                     max_norm=float("inf"), error_if_nonfinite=True)
            self.optimizer.step()
            count += labels.numel()
            correct += output.logits.detach().argmax(1).eq(labels).sum().item()
            for name, value in losses.items():
                sums[name] += value.detach().item() * labels.numel()
        synchronize(self.device)
        if not count:
            raise ValueError("Cannot train on an empty dataset")
        seconds = time.perf_counter() - start
        return {**{name: value / count for name, value in sums.items()}, "accuracy": correct / count,
                "samples": count, "seconds": seconds, "samples_per_second": count / seconds}


def _resume_signature(config: dict) -> dict:
    # Fill new optional defaults so pre-calibration checkpoints stay resumable.
    normalized = from_dict(config).to_dict()
    return {key: value for key, value in normalized.items() if key not in {"name", "output_dir", "benchmark"}}


def run_experiment(config: ExperimentConfig, *, resume: str | None = None) -> dict:
    config.validate()
    seed_everything(config.train.seed)
    torch.set_num_threads(config.train.threads)
    device = resolve_device(config.train.device)
    data = build_data(config.data, config.train, include_test=False)
    student = create_model(config.student.name, config.data.num_classes)
    student_metadata = metadata(config.student.name, data.classes, config.data.image_size, config.data.source)
    if config.student.checkpoint:
        load_model_checkpoint(student, config.student.checkpoint, student_metadata)
    initial_student_sha256 = tensor_state_fingerprint(student.state_dict())
    teacher = None
    teacher_info = None
    teacher_hash = None
    teacher_tensor_hash = None
    if config.distillation.method != "supervised":
        teacher = create_model(config.teacher.name, config.data.num_classes)
        teacher_state = load_model_checkpoint(teacher, config.teacher.checkpoint,
                                               metadata(config.teacher.name, data.classes, config.data.image_size, config.data.source))
        if teacher_state.get("epoch", -1) < 0:
            raise ValueError("Teacher checkpoint must come from a completed training epoch")
        teacher_hash = fingerprint(config.teacher.checkpoint)
        teacher.to(device).eval()
        teacher_tensor_hash = tensor_state_fingerprint(teacher.state_dict())
        teacher_info = evaluate(teacher, data.val, device)
        teacher_count = sum(p.numel() for p in teacher.parameters())
        ratio = teacher_count / sum(p.numel() for p in student.parameters())
        teacher_info.update(parameters=teacher_count, capacity_ratio=ratio, checkpoint_sha256=teacher_hash)
        if ratio > 10:
            warnings.warn(f"Teacher/student parameter ratio is {ratio:.1f}. Consider an intermediate teacher assistant; 10x is a heuristic.", stacklevel=2)
    feature_loss = None
    if config.distillation.feature_weight:
        feature_loss = StageFeatureLoss(student.feature_channels, teacher.feature_channels, config.distillation.stage_pairs)
    calibration_loss = MMCELoss(config.calibration.kernel_bandwidth) if config.calibration.enabled else None
    adaptive_loss = (AdaptiveFocalLoss(config.supervised_loss)
                     if config.supervised_loss.method != "cross_entropy" else None)
    cpc_loss = (PairwiseCalibrationLoss(config.cpc.discrimination_weight, config.cpc.exclusion_weight,
                                        config.cpc.warmup_epochs)
                if config.cpc.enabled else None)
    objective = DistillationObjective(config.distillation, feature_loss, calibration_loss, adaptive_loss,
                                      cpc_loss=cpc_loss)
    controller = CalibrationController(config.calibration)
    system = DistillationSystem(student, teacher, objective).to(device)
    optimizer = torch.optim.SGD([p for p in system.parameters() if p.requires_grad],
                                 lr=config.train.learning_rate, momentum=config.train.momentum,
                                 weight_decay=config.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs)
    trainer = Trainer(system, optimizer, device)
    directory = Path(config.output_dir) / config.name
    history, best_accuracy, start_epoch = [], -1.0, 0
    loaders = {"train": data.train, "val": data.val}
    if data.test is not None:
        loaders["test"] = data.test
    # Reset augmentation randomness after constructing different teacher/loss
    # variants, so ablations start from identical students and image streams.
    seed_everything(config.train.seed)
    if resume:
        state = load_model_checkpoint(student, resume, student_metadata)
        if state.get("kind") != "training":
            raise ValueError("Resume requires a training checkpoint (last.pt or best.pt)")
        if _resume_signature(state["config"]) != _resume_signature(config.to_dict()):
            raise ValueError("Resume configuration differs from the original training configuration")
        if state["teacher_sha256"] != teacher_hash:
            raise ValueError("Teacher checkpoint changed since the saved training epoch")
        objective.load_state_dict(state["objective"])
        if adaptive_loss is not None and adaptive_loss.controller.last_epoch.item() != state["epoch"] + 1:
            raise ValueError("Adaptive focal controller state does not match the resumed epoch")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history, best_accuracy = state["history"], state["best_accuracy"]
        if config.calibration.enabled and "calibration_controller" not in state:
            raise ValueError("Calibration resume requires saved controller state")
        if "calibration_controller" in state:
            controller.load_state_dict(state["calibration_controller"])
        start_epoch = state["epoch"] + 1
        # Resume in the same run to retain its previously selected best model.
        if Path(resume).resolve().parent != directory.resolve() or not (directory / "best.pt").exists():
            raise ValueError("Resume in the original run directory, which must contain best.pt")
        restore_rng(state["rng"], loaders)
    else:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Run directory is not empty: {directory}; choose a new name or resume last.pt")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", config.to_dict())
    write_json(directory / "data.json", data.provenance)
    write_json(directory / "teacher.json", teacher_info or {"used": False})
    for epoch in range(start_epoch, config.train.epochs):
        objective.calibration_weight = controller.weight_for_epoch(epoch + 1)
        training = trainer.train_epoch(data.train, epoch)
        adaptive_decision = None
        if adaptive_loss is None:
            validation = evaluate(student, data.val, device)
        else:
            # Reuse the student validation pass; the teacher/test never drive gamma.
            feedback = []
            validation = evaluate(student, data.val, device,
                                  confidence_observer=lambda confidence, correct: feedback.append((confidence, correct)))
            adaptive_decision = adaptive_loss.controller.observe(
                epoch + 1, torch.cat([v[0] for v in feedback]), torch.cat([v[1] for v in feedback]))
        decision = controller.observe(epoch + 1, validation)
        history.append({"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"],
                        "train": training, "val": validation,
                        "calibration_weight": objective.calibration_weight, "calibration_controller": decision})
        if adaptive_decision is not None:
            history[-1]["adaptive_focal"] = adaptive_decision
        scheduler.step()
        improved = validation["accuracy"] > best_accuracy
        best_accuracy = max(best_accuracy, validation["accuracy"])
        state = {**student_metadata, "kind": "training", "epoch": epoch,
                 "student": student.state_dict(), "objective": objective.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "best_accuracy": best_accuracy, "history": history, "config": config.to_dict(),
                 "teacher_sha256": teacher_hash, "rng": capture_rng(loaders),
                 "calibration_controller": controller.state_dict()}
        if improved:
            save_checkpoint(directory / "best.pt", state)
        save_checkpoint(directory / "last.pt", state)
        write_json(directory / "history.json", history)
        print(f"{config.name} epoch={epoch + 1}/{config.train.epochs} loss={training['total']:.4f} val_accuracy={validation['accuracy']:.4f} val_ece={validation['ece_15_bins']:.4f} cal_weight={objective.calibration_weight:.3f}", flush=True)
    best = load_model_checkpoint(student, directory / "best.pt", student_metadata)
    if teacher is not None and tensor_state_fingerprint(teacher.state_dict()) != teacher_tensor_hash:
        raise RuntimeError("Teacher tensors changed during student training")
    quality = evaluate(student, data.val, device)
    cost = benchmark(student, config.data.image_size, device, config.benchmark)
    summary = {"name": config.name, "method": config.distillation.method, "seed": config.train.seed,
               "initial_student_sha256": initial_student_sha256,
               "supervised_loss": config.supervised_loss.method,
               "data_source": config.data.source, "data": data.provenance,
               "synthetic_smoke_only": config.data.source == "synthetic",
               "best_epoch": best["epoch"] + 1, "validation": quality, "teacher": teacher_info,
               "calibration_controller": controller.state_dict(),
               "inference": cost, "deployment_budget": check_budget(cost, config.benchmark),
               "training": {"total_epoch_seconds": sum(h["train"]["seconds"] for h in history),
                            "mean_epoch_seconds": statistics.mean(h["train"]["seconds"] for h in history),
                            "extra_trainable_parameters": sum(p.numel() for p in objective.parameters()),
                            "timing_scope": "training epochs including data loading; excludes validation, checkpointing and profiling"},
               "checkpoint": str(directory / "student.pt")}
    if adaptive_loss is not None:
        summary["adaptive_focal_final_state"] = adaptive_loss.controller.snapshot()
    if config.cpc.enabled:
        summary["cpc"] = {"discrimination_weight": config.cpc.discrimination_weight,
                          "exclusion_weight": config.cpc.exclusion_weight,
                          "schedule": "fixed_from_epoch_1", "logits_temperature": 1.0}
    # Deployment artifact contains only student weights and their contract.
    save_checkpoint(directory / "student.pt", {**student_metadata, "kind": "inference",
                                               "epoch": best["epoch"], "student": student.cpu().state_dict()})
    write_json(directory / "summary.json", summary)
    return summary


def evaluate_checkpoint(config: ExperimentConfig, checkpoint: str, split: str = "test") -> dict:
    config.validate(require_teacher=False)
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    seed_everything(config.train.seed)
    torch.set_num_threads(config.train.threads)
    device = resolve_device(config.train.device)
    data = build_data(config.data, config.train, include_test=split == "test")
    loader = getattr(data, split)
    if loader is None:
        raise ValueError(f"Dataset has no {split} split")
    model = create_model(config.student.name, config.data.num_classes)
    load_model_checkpoint(model, checkpoint, metadata(config.student.name, data.classes, config.data.image_size, config.data.source))
    model.to(device)
    cost = benchmark(model, config.data.image_size, device, config.benchmark)
    return {"split": split, "quality": evaluate(model, loader, device), "inference": cost,
            "deployment_budget": check_budget(cost, config.benchmark),
            "synthetic_smoke_only": config.data.source == "synthetic"}
