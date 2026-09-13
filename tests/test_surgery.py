from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest

import torch

from kd.config import (BenchmarkConfig, DataConfig, DistillationConfig, ExperimentConfig,
                       ModelConfig, SurgeryConfig, TrainConfig, from_dict)
from kd.data import build_data
from kd.engine import run_experiment, seed_everything
from kd.surgery import (apply_training_surgery, create_surgery_plan, select_bad_samples,
                        slice_report)


class SurgerySelectionTests(unittest.TestCase):
    def test_hybrid_selection_combines_bad_slices_and_global_samples(self):
        records = [
            {"index": 0, "true_label": 0, "predicted_label": 1, "correct": False,
             "confidence": .9, "true_probability": .01, "nll": 5., "prediction_margin": .8},
            {"index": 1, "true_label": 0, "predicted_label": 0, "correct": True,
             "confidence": .6, "true_probability": .6, "nll": 4., "prediction_margin": .2},
            {"index": 2, "true_label": 1, "predicted_label": 0, "correct": False,
             "confidence": .95, "true_probability": .02, "nll": 3., "prediction_margin": .9},
            {"index": 3, "true_label": 1, "predicted_label": 0, "correct": False,
             "confidence": .8, "true_probability": .05, "nll": 2., "prediction_margin": .7},
            {"index": 4, "true_label": 2, "predicted_label": 2, "correct": True,
             "confidence": .7, "true_probability": .7, "nll": 1., "prediction_margin": .4},
            {"index": 5, "true_label": 2, "predicted_label": 2, "correct": True,
             "confidence": .8, "true_probability": .8, "nll": .5, "prediction_margin": .6},
        ]
        selected, slices = select_bad_samples(
            records, 4, score="high_loss", strategy="hybrid", min_slice_size=2)
        self.assertEqual([row["index"] for row in selected], [0, 2, 1, 3])
        self.assertEqual(slices, ["true:0", "true:1", "true:2"])
        self.assertEqual([row["selection_reason"] for row in selected],
                         ["slice", "slice", "sample", "sample"])

    def test_high_confidence_error_does_not_remove_correct_samples(self):
        records = [
            {"index": 0, "true_label": 0, "predicted_label": 0, "correct": True,
             "confidence": .99, "true_probability": .99, "nll": .01, "prediction_margin": .9}
        ]
        selected, slices = select_bad_samples(
            records, 1, score="high_confidence_error", strategy="hybrid", min_slice_size=1)
        self.assertEqual(selected, [])
        self.assertEqual(slices, [])

    def test_slice_report_exposes_true_class_and_confusion_failures(self):
        records = [
            {"index": 0, "true_label": 0, "predicted_label": 1, "correct": False,
             "confidence": .9, "true_probability": .1, "nll": 2.3, "prediction_margin": .8},
            {"index": 1, "true_label": 0, "predicted_label": 0, "correct": True,
             "confidence": .7, "true_probability": .7, "nll": .35, "prediction_margin": .5},
        ]
        report = slice_report(records, ["left", "right"])
        self.assertEqual(report["true_class"][0]["key"], "true:0")
        self.assertEqual(report["true_class"][0]["support"], 2)
        self.assertAlmostEqual(report["true_class"][0]["accuracy"], .5)
        self.assertEqual(report["confusion"][0]["key"], "confusion:0->1")

    def test_surgery_configuration_is_explicit_and_bounded(self):
        self.assertEqual(from_dict({}).surgery.action, "none")
        for surgery in (SurgeryConfig(action="drop"),
                        SurgeryConfig(plan="plan.json"),
                        SurgeryConfig(action="prune", plan="plan.json"),
                        SurgeryConfig(max_drop_fraction=1.0)):
            with self.subTest(surgery=surgery), self.assertRaises(ValueError):
                surgery.validate()


class SurgeryIntegrationTests(unittest.TestCase):
    def test_plan_drops_bounded_training_samples_before_kd(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            teacher_config = ExperimentConfig(
                name="teacher", output_dir=root,
                data=DataConfig(num_classes=3, train_samples=12, val_samples=6, test_samples=6),
                student=ModelConfig("tiny_medium"),
                distillation=DistillationConfig(method="supervised"),
                train=TrainConfig(epochs=1, batch_size=6, device="cpu", threads=1),
                benchmark=BenchmarkConfig(warmup=0, iterations=1),
            )
            teacher = run_experiment(teacher_config)
            plan_path = Path(root) / "surgery-plan.json"
            seed_everything(teacher_config.train.seed)
            plan = create_surgery_plan(
                teacher_config, teacher["checkpoint"], plan_path, torch.device("cpu"),
                fraction=.25, score="high_loss", strategy="hybrid", min_slice_size=1)
            self.assertEqual(plan["selection"]["selected_count"], 3)

            student_config = replace(
                teacher_config, name="student", student=ModelConfig("tiny_small"),
                teacher=ModelConfig("tiny_medium", teacher["checkpoint"]),
                distillation=DistillationConfig(method="kd"),
                surgery=SurgeryConfig(action="drop", plan=str(plan_path), max_drop_fraction=.3),
            )
            summary = run_experiment(student_config)
            self.assertTrue(summary["surgery"]["enabled"])
            self.assertEqual(summary["surgery"]["dropped_samples"], 3)
            self.assertEqual(summary["surgery"]["final_train_size"], 9)
            self.assertEqual(summary["data"]["split_sizes"]["train"], 9)
            state = torch.load(Path(root) / "student" / "last.pt", weights_only=True)
            self.assertEqual(state["history"][0]["train"]["samples"], 9)
            self.assertEqual(state["surgery_plan_sha256"], summary["surgery"]["plan_sha256"])

            mismatched = replace(
                teacher_config, train=replace(teacher_config.train, seed=43),
                surgery=SurgeryConfig(action="drop", plan=str(plan_path), max_drop_fraction=.3),
            )
            with self.assertRaisesRegex(ValueError, "dataset contract"):
                apply_training_surgery(build_data(mismatched.data, mismatched.train), mismatched)


if __name__ == "__main__":
    unittest.main()
