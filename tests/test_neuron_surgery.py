from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from kd.checkpoints import metadata, save_checkpoint
from kd.cli import main as cli_main
from kd.config import (BenchmarkConfig, DataConfig, DistillationConfig, ExperimentConfig,
                       ModelConfig, TrainConfig)
from kd.engine import run_experiment
from kd.models import ModelOutput, create_model
from kd.neuron_surgery import (ablate_channel, build_companion_records,
                               causal_channel_validation, cifar_channel_parameter_masks,
                               consensus_channel_ranking,
                               differential_channel_scores, repair_loss,
                               select_hard_tail_targets, train_repair_candidate,
                               validate_repair_provenance)


def diagnostic_row(index, label, *, correct=True, nll=0.1):
    predicted = label if correct else (label + 1) % 10
    return {"index": index, "true_label": label, "predicted_label": predicted,
            "correct": correct, "confidence": 0.9, "nll": nll}


class SelectionAndConsensusTests(unittest.TestCase):
    def test_hard_targets_keep_all_mistakes_then_fill_class_quota(self):
        records = []
        for label in (5, 6):
            for position in range(10):
                mistakes = position < (3 if label == 5 else 1)
                records.append(diagnostic_row(len(records), label, correct=not mistakes,
                                              nll=10 - position))
        selected = select_hard_tail_targets(records, (5, 6), fraction=0.2)
        class_five = [row for row in selected if row["true_label"] == 5]
        class_six = [row for row in selected if row["true_label"] == 6]
        self.assertEqual(len(class_five), 3)
        self.assertTrue(all(row["selection_reason"] == "misclassified" for row in class_five))
        self.assertEqual(len(class_six), 2)
        self.assertEqual(sum(row["selection_reason"] == "misclassified" for row in class_six), 1)

    def test_companions_are_correct_same_class_neighbours_with_stable_ties(self):
        records = [diagnostic_row(i, 5, nll=float(i)) for i in range(7)]
        records[6] = diagnostic_row(6, 6)
        embeddings = np.asarray([[1, 0], [1, 0], [1, 0], [.9, .1], [.8, .2], [.7, .3], [0, 1]],
                                dtype=np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        targets = [{**records[0], "selection_reason": "high_loss_correct"}]
        result = build_companion_records(records, embeddings, targets, companions=2)
        self.assertEqual([row["index"] for row in result[0]["companions"]], [1, 2])
        self.assertTrue(all(row["true_label"] == 5 for row in result[0]["companions"]))

    def test_consensus_is_class_balanced_and_bootstrap_stable(self):
        labels = np.repeat(np.arange(5, 10), 2)
        stage2 = np.ones((10, 4), dtype=np.float32)
        stage3 = np.ones((10, 4), dtype=np.float32)
        stage2[:, 0] = 100
        report = consensus_channel_ranking(
            {"labels": labels, "target_indices": np.arange(10), "stage2": stage2, "stage3": stage3},
            repetitions=20, seed=7, top_fraction=0.25, stability_threshold=0.7)
        winner = next(row for row in report["channels"]
                      if row["stage"] == "stage2" and row["channel"] == 0)
        self.assertEqual(winner["bootstrap_stability"], 1.0)
        self.assertTrue(winner["eligible"])


class LocalizationAndMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_differential_scores_have_one_finite_value_per_stage_channel(self):
        model = create_model("cifar_teacher", 10).eval()
        dataset = TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([5, 5, 6, 6]))
        pairs = [
            {"target": {"index": 0}, "companions": [{"index": 1}]},
            {"target": {"index": 2}, "companions": [{"index": 3}]},
        ]
        result = differential_channel_scores(model, dataset, pairs, torch.device("cpu"), batch_size=2)
        self.assertEqual(result["stage2"].shape, (2, 64))
        self.assertEqual(result["stage3"].shape, (2, 128))
        self.assertTrue(np.isfinite(result["stage2"]).all())
        self.assertEqual(result["labels"].tolist(), [5, 6])

    def test_ablation_hook_is_removed_even_after_an_exception(self):
        model = create_model("cifar_teacher", 10).eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.inference_mode():
            baseline = model(images).logits.clone()
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with ablate_channel(model, "stage3", 0):
                raise RuntimeError("stop")
        with torch.inference_mode():
            restored = model(images).logits
        torch.testing.assert_close(restored, baseline, rtol=0, atol=0)

    def test_channel_masks_cover_only_declared_producers_bn_and_consumers(self):
        model = create_model("cifar_teacher", 10)
        masks = cifar_channel_parameter_masks(
            model, [{"stage": "stage2", "channel": 3}, {"stage": "stage3", "channel": 7}])
        self.assertTrue(masks["stages.stage2.3.weight"][3].all())
        self.assertTrue(masks["stages.stage3.0.weight"][:, 3].all())
        self.assertTrue(masks["classifier.weight"][:, 28:32].all())
        self.assertFalse(masks["stages.stage1.0.weight"].any())
        self.assertFalse(masks["classifier.bias"].any())

    def test_causal_validation_reports_each_channel_and_leaves_model_clean(self):
        model = create_model("cifar_teacher", 10).eval()
        target_labels = torch.tensor(list(range(5, 10)) * 2)
        preservation_labels = torch.tensor(list(range(10)) * 2)
        target = DataLoader(TensorDataset(torch.randn(10, 3, 32, 32), target_labels), batch_size=5)
        preservation = DataLoader(
            TensorDataset(torch.randn(20, 3, 32, 32), preservation_labels), batch_size=10)
        consensus = {"channels": [{"stage": "stage3", "channel": 0,
                                    "eligible": True, "consensus_score": 1.0,
                                    "bootstrap_stability": 1.0}]}
        result = causal_channel_validation(model, target, preservation, consensus,
                                           torch.device("cpu"))
        self.assertEqual(len(result["channels"]), 1)
        self.assertEqual(result["channels"][0]["stage"], "stage3")
        self.assertEqual(len(model.stages["stage3"]._forward_hooks), 0)

    def test_masked_training_changes_no_unselected_coordinates_or_bn_buffers(self):
        anchor = create_model("cifar_teacher", 10).eval()
        candidate = deepcopy(anchor)
        images = torch.randn(20, 3, 32, 32)
        labels = np.tile(np.arange(10), 2)
        dataset = TensorDataset(images, torch.tensor(labels))
        anchor_before = {name: value.clone() for name, value in anchor.state_dict().items()}
        history, verification = train_repair_candidate(
            candidate, anchor, dataset, labels, list(range(10, 20)), list(range(10)),
            [{"stage": "stage3", "channel": 0}], torch.device("cpu"), epochs=1,
            samples_per_epoch=10, batch_size=5, learning_rate=0.01, seed=3)
        self.assertEqual(len(history), 1)
        self.assertTrue(verification["all_unselected_coordinates_unchanged"])
        self.assertTrue(verification["all_buffers_unchanged"])
        self.assertGreater(verification["changed_parameter_coordinates"], 0)
        for name, value in anchor.state_dict().items():
            torch.testing.assert_close(value, anchor_before[name], rtol=0, atol=0)

    def test_repair_preservation_terms_are_zero_for_identical_outputs(self):
        logits = torch.randn(3, 10)
        features = {"stage2": torch.randn(3, 4, 4, 4), "stage3": torch.randn(3, 8, 2, 2)}
        output = ModelOutput(logits, features)
        labels = torch.tensor([0, 1, 2])
        losses = repair_loss(output, labels, output, output, labels)
        self.assertLess(abs(losses["kd"].item()), 2e-6)
        self.assertAlmostEqual(losses["features"].item(), 0.0, places=6)
        self.assertLess(abs(losses["total"].item() - losses["ce"].item()), 2e-6)

    def test_direct_student_repair_uses_labeled_preservation_without_kd(self):
        target = ModelOutput(torch.randn(3, 10), {})
        preservation = ModelOutput(torch.randn(3, 10), {})
        anchor = ModelOutput(torch.randn(3, 10), {})
        labels = torch.tensor([0, 1, 2])
        losses = repair_loss(
            target, labels, preservation, anchor, labels,
            kd_weight=0.0, feature_weight=0.0, preservation_ce_weight=1.0)
        expected = (torch.nn.functional.cross_entropy(target.logits, labels)
                    + torch.nn.functional.cross_entropy(preservation.logits, labels))
        torch.testing.assert_close(losses["total"], expected)
        self.assertEqual(losses["kd"].item(), 0.0)
        self.assertEqual(losses["features"].item(), 0.0)

    def test_direct_student_masked_step_needs_no_distillation_terms(self):
        anchor = create_model("cifar_student", 10).eval()
        candidate = deepcopy(anchor)
        labels = np.tile(np.arange(10), 2)
        dataset = TensorDataset(torch.randn(20, 3, 32, 32), torch.tensor(labels))
        with patch.object(anchor, "forward", side_effect=AssertionError("anchor was used")):
            _, verification = train_repair_candidate(
                candidate, anchor, dataset, labels, list(range(10, 20)), list(range(10)),
                [{"stage": "stage3", "channel": 0}], torch.device("cpu"), epochs=1,
                samples_per_epoch=10, batch_size=5, learning_rate=0.01,
                kd_weight=0.0, feature_weight=0.0, preservation_ce_weight=1.0, seed=8)
        self.assertTrue(verification["all_unselected_coordinates_unchanged"])
        self.assertTrue(verification["all_buffers_unchanged"])


class RepairIntegrationTests(unittest.TestCase):
    def test_repaired_teacher_checkpoint_runs_through_existing_kd_engine(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            root = Path(root)
            teacher = create_model("cifar_teacher", 10)
            state = {**metadata("cifar_teacher", [f"class_{i}" for i in range(10)], 32, "synthetic"),
                     "kind": "inference", "epoch": 0, "student": teacher.state_dict(),
                     "repair": {"format_version": 1, "base_checkpoint_sha256": "base",
                                "localization_sha256": "plan", "channels": []}}
            checkpoint = root / "teacher_repaired.pt"
            save_checkpoint(checkpoint, state)
            validate_repair_provenance(state, base_checkpoint_sha256="base",
                                       localization_sha256="plan")
            config = ExperimentConfig(
                name="student", output_dir=str(root),
                data=DataConfig(num_classes=10, train_samples=20, val_samples=10, test_samples=10),
                student=ModelConfig("cifar_student"),
                teacher=ModelConfig("cifar_teacher", str(checkpoint)),
                distillation=DistillationConfig(method="kd"),
                train=TrainConfig(epochs=1, batch_size=10, workers=0, threads=1, device="cpu"),
                benchmark=BenchmarkConfig(warmup=0, iterations=1))
            result = run_experiment(config)
            self.assertTrue(Path(result["checkpoint"]).is_file())
            self.assertEqual(result["teacher"]["parameters"], 292586)

    def test_cli_exposes_read_only_study_dry_run(self):
        result = {"dry_run": True}
        with patch("kd.neuron_surgery_study.run_neuron_surgery_study", return_value=result) as run, \
             redirect_stdout(io.StringIO()) as output:
            cli_main(["neuron-surgery-study", "--baseline", "baseline", "--output", "output",
                      "--device", "cpu", "--dry-run"])
        run.assert_called_once_with("baseline", "output", "cpu", (42, 43, 44), dry_run=True)
        self.assertIn('"dry_run": true', output.getvalue())

    def test_repair_provenance_rejects_changed_inputs(self):
        state = {"kind": "inference", "repair": {"format_version": 1,
                 "base_checkpoint_sha256": "base", "localization_sha256": "plan"}}
        with self.assertRaisesRegex(ValueError, "base checkpoint changed"):
            validate_repair_provenance(state, base_checkpoint_sha256="other",
                                       localization_sha256="plan")


if __name__ == "__main__":
    unittest.main()
