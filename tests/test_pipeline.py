from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from kd.checkpoints import load_model_checkpoint, metadata
from kd.config import (BenchmarkConfig, DataConfig, DistillationConfig, ExperimentConfig,
                       ModelConfig, TrainConfig, from_dict)
from kd.data import build_data
from kd.engine import DistillationSystem, Trainer, evaluate_checkpoint, run_experiment, seed_everything
from kd.experiments import grid_configs, run_ablation
from kd.losses import DistillationObjective, StageFeatureLoss
from kd.metrics import benchmark, check_budget, evaluate
from kd.models import ModelOutput, create_model


class ConfigurationTests(unittest.TestCase):
    def test_missing_teacher_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "trained teacher"):
            ExperimentConfig().validate()

    def test_unsupported_tasks_are_explicit(self):
        for task in ("detection", "segmentation"):
            with self.assertRaisesRegex(ValueError, "task-specific"):
                from_dict({"task": task})

    def test_unknown_options_and_invalid_values_fail(self):
        with self.assertRaises(TypeError):
            from_dict({"distillation": {"temprature": 4}})
        for raw in ({"distillation": {"temperature": 0}}, {"distillation": {"weight": float("nan")}},
                    {"train": {"epochs": 0}}, {"name": "../escape"}, {"benchmark": {"iterations": 0}},
                    {"distillation": {"method": "supervised", "feature_weight": 1}}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                from_dict(raw)

    def test_grid_has_requested_baseline_and_nine_settings_per_method(self):
        runs = grid_configs(ExperimentConfig())
        self.assertEqual(len(runs), 19)
        self.assertEqual(sum(r.distillation.method == "kd" for r in runs), 9)
        self.assertEqual(sum(r.distillation.method == "dkd" for r in runs), 9)
        self.assertEqual({r.train.seed for r in runs}, {42})
        self.assertTrue(all(r.distillation.feature_weight == 0 for r in runs))
        with self.assertRaises(ValueError):
            grid_configs(ExperimentConfig(), temperatures=(2, 2))


class DataTests(unittest.TestCase):
    def test_synthetic_data_and_augmentation_are_reproducible(self):
        def first():
            seed_everything(123)
            return next(iter(build_data(DataConfig(augmentation="strong"), TrainConfig()).train))
        first_images, first_labels = first()
        images, labels = first()
        torch.testing.assert_close(first_images, images, rtol=0, atol=0)
        torch.testing.assert_close(first_labels, labels, rtol=0, atol=0)

    def test_imagefolder_class_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for split, names in (("train", ["left", "right"]), ("val", ["left", "stop"])):
                for name in names:
                    directory = Path(root) / split / name
                    directory.mkdir(parents=True)
                    Image.new("RGB", (20, 20)).save(directory / "image.png")
            with self.assertRaisesRegex(ValueError, "Class names/order"):
                build_data(DataConfig(source="imagefolder", root=root, num_classes=2), TrainConfig())


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_teacher_frozen_shared_images_and_projection_updates(self):
        student, teacher = create_model("tiny_small", 4), create_model("tiny_medium", 4)
        alignment = StageFeatureLoss(student.feature_channels, teacher.feature_channels, (("stage2", "stage2"),))
        system = DistillationSystem(student, teacher,
                                    DistillationObjective(DistillationConfig(feature_weight=1), alignment))
        before = {k: v.clone() for k, v in teacher.state_dict().items()}
        before_projection = alignment.projections[0].weight.clone()
        seen = []
        handles = [model.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].data_ptr()))
                   for model in (student, teacher)]
        optimizer = torch.optim.SGD([p for p in system.parameters() if p.requires_grad], lr=0.1)
        images, labels = torch.randn(4, 3, 32, 32), torch.arange(4)
        Trainer(system, optimizer, torch.device("cpu")).train_epoch(DataLoader(TensorDataset(images, labels), batch_size=4), 0)
        for handle in handles:
            handle.remove()
        self.assertEqual(seen[0], seen[1])
        self.assertFalse(teacher.training)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        for name, value in teacher.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(before_projection, alignment.projections[0].weight))

    def test_resnet_adapter_contract(self):
        model = create_model("resnet18", 3).eval()
        with torch.inference_mode():
            result = model(torch.randn(2, 3, 32, 32), return_features=True)
        self.assertEqual(result.logits.shape, (2, 3))
        self.assertEqual({name: f.shape[1] for name, f in result.features.items()}, model.feature_channels)

    def test_benchmark_and_budget_use_student_inference_only(self):
        model = create_model("tiny_small", 4)
        result = benchmark(model, 32, torch.device("cpu"), BenchmarkConfig(warmup=0, iterations=2))
        self.assertTrue(model.training)
        self.assertGreater(result["estimated_flops"], 0)
        self.assertEqual(result["estimated_flops"], 2 * result["conv_linear_macs"])
        self.assertFalse(check_budget(result, BenchmarkConfig(max_parameters=1))["passed"])

    def test_confidence_metrics_include_wrong_predictions(self):
        class Constant(torch.nn.Module):
            def forward(self, images):
                return ModelOutput(torch.tensor([[2.0, 0.0]]).expand(images.shape[0], -1))
        result = evaluate(Constant(), DataLoader(TensorDataset(torch.zeros(4, 3, 16, 16), torch.tensor([0, 0, 1, 1])), batch_size=4), torch.device("cpu"))
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["correct_confidence"], result["incorrect_confidence"], places=6)
        self.assertAlmostEqual(result["ece_15_bins"], result["mean_confidence"] - 0.5, places=6)


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = ExperimentConfig(name="teacher", output_dir=self.temporary.name,
                                        data=DataConfig(train_samples=16, val_samples=8, test_samples=8),
                                        student=ModelConfig("tiny_medium"),
                                        distillation=DistillationConfig(method="supervised"),
                                        train=TrainConfig(epochs=2, batch_size=8, device="cpu"),
                                        benchmark=BenchmarkConfig(warmup=0, iterations=2))
        self.output = io.StringIO()

    def test_distillation_export_evaluation_and_checkpoint_validation(self):
        with redirect_stdout(self.output):
            teacher = run_experiment(self.config)
            config = replace(self.config, name="student", student=ModelConfig("tiny_small"),
                             teacher=ModelConfig("tiny_medium", teacher["checkpoint"]),
                             distillation=DistillationConfig(feature_weight=1))
            result = run_experiment(config)
        self.assertGreater(result["training"]["extra_trainable_parameters"], 0)
        self.assertIsNotNone(result["teacher"]["mean_confidence"])
        artifact = torch.load(result["checkpoint"], weights_only=True)
        self.assertNotIn("optimizer", artifact)
        self.assertTrue(all(not key.startswith("teacher") for key in artifact["student"]))
        evaluated = evaluate_checkpoint(config, result["checkpoint"])
        self.assertEqual(evaluated["split"], "test")
        expected = metadata("tiny_small", ["wrong"] * 4, 32)
        with self.assertRaisesRegex(ValueError, "classes"):
            load_model_checkpoint(create_model("tiny_small", 4), result["checkpoint"], expected)
        with self.assertRaises(FileExistsError):
            run_experiment(config)

    def test_interrupted_training_resumes_exactly_at_epoch_boundary(self):
        with redirect_stdout(self.output):
            uninterrupted = run_experiment(replace(self.config, name="uninterrupted"))
        original = Trainer.train_epoch

        def interrupt_after_first(trainer, loader, epoch):
            if epoch == 1:
                raise InterruptedError("simulated interruption")
            return original(trainer, loader, epoch)

        with patch.object(Trainer, "train_epoch", interrupt_after_first), redirect_stdout(self.output):
            with self.assertRaises(InterruptedError):
                run_experiment(self.config)
        path = str(Path(self.config.output_dir) / self.config.name / "last.pt")
        with redirect_stdout(self.output):
            resumed = run_experiment(self.config, resume=path)
        # Compare final (not only best) weights to verify optimizer/RNG recovery.
        expected = torch.load(Path(uninterrupted["checkpoint"]).with_name("last.pt"), weights_only=True)
        actual = torch.load(Path(resumed["checkpoint"]).with_name("last.pt"), weights_only=True)
        for name, value in expected["student"].items():
            torch.testing.assert_close(value, actual["student"][name], rtol=0, atol=0)
        self.assertEqual(expected["history"][-1]["train"]["total"], actual["history"][-1]["train"]["total"])
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            run_experiment(replace(self.config, train=replace(self.config.train, learning_rate=0.1)), resume=path)

    def test_dkd_resume_restores_feature_optimizer_and_teacher_identity(self):
        with redirect_stdout(self.output):
            teacher = run_experiment(self.config)
            config = replace(self.config, name="dkd_full", student=ModelConfig("tiny_small"),
                             teacher=ModelConfig("tiny_medium", teacher["checkpoint"]),
                             distillation=DistillationConfig(feature_weight=1))
            full = run_experiment(config)
        resumed_config = replace(config, name="dkd_resumed")
        original = Trainer.train_epoch

        def interrupted(trainer, loader, epoch):
            if epoch == 1:
                raise InterruptedError("simulated interruption")
            return original(trainer, loader, epoch)

        with patch.object(Trainer, "train_epoch", interrupted), redirect_stdout(self.output):
            with self.assertRaises(InterruptedError):
                run_experiment(resumed_config)
        path = Path(self.config.output_dir) / resumed_config.name / "last.pt"
        with redirect_stdout(self.output):
            run_experiment(resumed_config, resume=str(path))
        expected = torch.load(Path(full["checkpoint"]).with_name("last.pt"), weights_only=True)
        actual = torch.load(path, weights_only=True)
        for section in ("student", "objective"):
            for name, value in expected[section].items():
                torch.testing.assert_close(value, actual[section][name], rtol=0, atol=0)
        artifact = torch.load(teacher["checkpoint"], weights_only=True)
        artifact["student"]["classifier.bias"].add_(0.1)
        torch.save(artifact, teacher["checkpoint"])
        with self.assertRaisesRegex(ValueError, "Teacher checkpoint changed"):
            run_experiment(resumed_config, resume=str(path))

    def test_real_imagefolder_pipeline_and_class_order(self):
        root = Path(self.temporary.name) / "images"
        for split in ("train", "val", "test"):
            for label, color in (("stop", "red"), ("yield", "yellow")):
                directory = root / split / label
                directory.mkdir(parents=True)
                for index in range(2):
                    Image.new("RGB", (24, 24), color).save(directory / f"{index}.png")
        config = replace(self.config, data=DataConfig(source="imagefolder", root=str(root), num_classes=2),
                         train=replace(self.config.train, epochs=1))
        with redirect_stdout(self.output):
            report = run_experiment(config)
        artifact = torch.load(report["checkpoint"], weights_only=True)
        self.assertEqual(artifact["classes"], ["stop", "yield"])
        self.assertFalse(report["synthetic_smoke_only"])
        self.assertEqual(evaluate_checkpoint(config, report["checkpoint"])["quality"]["samples"], 4)

    def test_ablation_compares_five_variants_and_respects_budget(self):
        with redirect_stdout(self.output):
            teacher = run_experiment(self.config)
            config = replace(self.config, name="sweep", student=ModelConfig("tiny_small"),
                             teacher=ModelConfig("tiny_medium", teacher["checkpoint"]),
                             train=replace(self.config.train, epochs=1),
                             distillation=DistillationConfig(),
                             benchmark=BenchmarkConfig(warmup=0, iterations=1, max_parameters=1))
            report = run_ablation(config, temperatures=(4,), weights=(0.5,))
        self.assertEqual(len(report["runs"]), 5)
        self.assertIsNone(report["best"])
        self.assertEqual(report["status"], "no_model_within_budget")
        self.assertEqual(report["selection_split"], "val")
        comparison = Path(self.config.output_dir) / "sweep" / "comparison.json"
        self.assertEqual(len(json.loads(comparison.read_text())["runs"]), 5)
        self.assertEqual(report["runs"][0]["training_overhead_ratio"], 1)
