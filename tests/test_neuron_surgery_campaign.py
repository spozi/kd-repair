from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import TensorDataset

from kd.config import DataConfig, ExperimentConfig, TrainConfig
from kd.data import stratified_confirmation_split
from kd.models import create_model
from kd.neuron_surgery import (differential_channel_scores, localization_stages,
                               resnet_channel_parameter_masks, train_repair_candidate)
from kd.neuron_surgery_baselines import baseline_configs
from kd.neuron_surgery_campaign import (_best_candidates_by_budget, campaign_plan,
                                        run_neuron_surgery_campaign)
from kd.neuron_surgery_study import (NeuronSurgerySpec, load_neuron_surgery_spec,
                                     neuron_surgery_spec_from_dict)


class ConfirmationSplitTests(unittest.TestCase):
    def test_confirmation_partition_is_balanced_disjoint_and_deterministic(self):
        labels = np.repeat(np.arange(10), 20)
        first = stratified_confirmation_split(labels, 0.2, 0.1, 73)
        second = stratified_confirmation_split(labels, 0.2, 0.1, 73)
        self.assertEqual(first, second)
        train, validation, confirmation = (set(values) for values in first)
        self.assertFalse(train & validation or train & confirmation or validation & confirmation)
        self.assertEqual(train | validation | confirmation, set(range(200)))
        self.assertEqual(np.bincount(labels[list(confirmation)]).tolist(), [2] * 10)

    def test_confirmation_configuration_rejects_leakage_prone_values(self):
        for value in (-0.1, 1.0):
            with self.assertRaisesRegex(ValueError, "confirmation_fraction"):
                ExperimentConfig(data=DataConfig(confirmation_fraction=value)).validate(
                    require_teacher=False)
        with self.assertRaisesRegex(ValueError, "below one"):
            ExperimentConfig(
                data=DataConfig(source="cifar10", num_classes=10, image_size=32,
                                validation_fraction=0.6, confirmation_fraction=0.4)).validate(
                                    require_teacher=False)


class GeneralizedLocalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gradient_only_attribution_does_not_forward_companion_images(self):
        model = create_model("cifar_teacher", 10).eval()
        images = torch.randn(2, 3, 32, 32)
        images[1].fill_(float("nan"))
        dataset = TensorDataset(images, torch.tensor([5, 5]))
        pairs = [{"target": {"index": 0}, "companions": [{"index": 1}]}]
        result = differential_channel_scores(
            model, dataset, pairs, torch.device("cpu"), batch_size=1,
            score_mode="gradient_only")
        self.assertTrue(np.isfinite(result["stage2"]).all())
        self.assertTrue(np.isfinite(result["stage3"]).all())

    def test_resnet18_map_covers_terminal_basic_block_and_both_consumers(self):
        model = create_model("resnet18", 10)
        masks = resnet_channel_parameter_masks(
            model, [{"stage": "stage3", "channel": 3},
                    {"stage": "stage4", "channel": 7}])
        self.assertEqual(localization_stages(model), ("stage3", "stage4"))
        self.assertTrue(masks["backbone.layer3.1.conv2.weight"][3].all())
        self.assertTrue(masks["backbone.layer3.1.bn2.weight"][3])
        self.assertTrue(masks["backbone.layer4.0.conv1.weight"][:, 3].all())
        self.assertTrue(masks["backbone.layer4.0.downsample.0.weight"][:, 3].all())
        self.assertTrue(masks["backbone.fc.weight"][:, 7].all())
        self.assertFalse(masks["backbone.conv1.weight"].any())

    def test_resnet50_map_uses_terminal_bottleneck_convolution(self):
        model = create_model("resnet50", 10)
        masks = resnet_channel_parameter_masks(
            model, [{"stage": "stage4", "channel": 11}])
        self.assertTrue(masks["backbone.layer4.2.conv3.weight"][11].all())
        self.assertTrue(masks["backbone.layer4.2.bn3.bias"][11])
        self.assertTrue(masks["backbone.fc.weight"][:, 11].all())
        self.assertFalse(masks["backbone.layer4.2.conv2.weight"].any())

    def test_resnet_masked_optimizer_step_preserves_all_other_coordinates(self):
        anchor = create_model("resnet18", 10).eval()
        candidate = deepcopy(anchor)
        labels = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
        images = torch.randn(8, 3, 32, 32)
        with torch.inference_mode():
            activity = anchor(images, return_features=True).features["stage4"].abs().mean((0, 2, 3))
        channel = int(activity.argmax())
        dataset = TensorDataset(images, torch.tensor(labels))
        history, verification = train_repair_candidate(
            candidate, anchor, dataset, labels, [4, 5, 6, 7], [0, 1, 2, 3],
            [{"stage": "stage4", "channel": channel}], torch.device("cpu"), epochs=1,
            samples_per_epoch=4, batch_size=2, learning_rate=0.001, seed=19)
        self.assertEqual(len(history), 1)
        self.assertTrue(verification["all_unselected_coordinates_unchanged"])
        self.assertTrue(verification["all_buffers_unchanged"])
        self.assertGreater(verification["changed_parameter_coordinates"], 0)


class StudyConfigurationTests(unittest.TestCase):
    def test_spec_json_round_trip_and_confirmation_guard(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "spec.json"
            path.write_text(json.dumps({
                "name": "confirmation", "student_seeds": [1, 2, 3],
                "confirmation_fraction": 0.1, "evaluation_split": "confirmation",
                "expected_train_counts": None, "expected_target_counts": None,
            }))
            spec = load_neuron_surgery_spec(path)
        self.assertEqual(spec.student_seeds, (1, 2, 3))
        self.assertEqual(spec.evaluation_split, "confirmation")
        with self.assertRaisesRegex(ValueError, "requires confirmation_fraction"):
            neuron_surgery_spec_from_dict({"evaluation_split": "confirmation"})

    def test_baseline_builder_matches_spec_models_seeds_and_holdout(self):
        spec = NeuronSurgerySpec(
            name="resnet-confirmation", teacher_run="teacher_resnet34",
            student_run_template="student_resnet18_seed{seed}",
            teacher_model="resnet34", student_model="resnet18", teacher_seed=10,
            student_seeds=(11, 12), split_seed=99, confirmation_fraction=0.1,
            evaluation_split="confirmation", training_epochs=2,
            expected_train_counts=None, expected_target_counts=None)
        teacher, students = baseline_configs("runs/example", "data", "cpu", spec)
        self.assertEqual(teacher.student.name, "resnet34")
        self.assertEqual([row.student.name for row in students], ["resnet18", "resnet18"])
        self.assertEqual([row.train.seed for row in students], [11, 12])
        self.assertEqual(teacher.data.confirmation_fraction, 0.1)

    def test_campaign_dry_plan_is_sequential_and_contains_every_requested_arm(self):
        plan = campaign_plan("runs/campaign", "data", "runs/baseline", "runs/reference", "cpu")
        self.assertIn("sealed_factor100_baselines_and_study", plan["stages"])
        self.assertIn("factor50_baselines_and_study", plan["stages"])
        self.assertIn("resnet34_to_resnet18_baselines_and_study", plan["stages"])
        self.assertFalse(plan["specs"]["no_companions"]["downstream_kd"])

    def test_campaign_runs_serial_stages_and_applies_resnet_gate(self):
        failed = {"status": "no_repair_selected", "reason": "gate test"}
        ablation = {"status": "repair_only_complete"}
        outcomes = [failed, failed, ablation, ablation, ablation, failed]
        budget = {"best_by_budget": {}, "candidates": [], "baseline_validation": {}}
        with tempfile.TemporaryDirectory() as root, \
             patch("kd.neuron_surgery_campaign.run_neuron_surgery_study",
                   side_effect=outcomes) as study, \
             patch("kd.neuron_surgery_campaign.run_neuron_surgery_baselines") as baselines, \
             patch("kd.neuron_surgery_campaign._budget_summary", return_value=budget):
            result = run_neuron_surgery_campaign(
                Path(root) / "campaign", data_root="data", legacy_baseline="baseline",
                reference_study="reference", device="cpu")
        self.assertEqual(study.call_count, 6)
        self.assertEqual(baselines.call_count, 2)
        self.assertEqual(result["comparisons"]["resnet"]["status"],
                         "skipped_by_predeclared_gate")

    def test_budget_summary_prefers_guardrail_eligible_candidate(self):
        rows = [
            {"budget": 8, "learning_rate": 0.005, "eligible": False,
             "tail_recall_delta": 0.03, "tail_nll_delta": -0.3},
            {"budget": 8, "learning_rate": 0.001, "eligible": True,
             "tail_recall_delta": 0.02, "tail_nll_delta": -0.1},
        ]
        self.assertEqual(_best_candidates_by_budget(rows)["8"]["learning_rate"], 0.001)


if __name__ == "__main__":
    unittest.main()
