import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import TensorDataset

from kd.neuron_surgery import (build_companion_records, differential_channel_scores,
                               select_preservation_indices)
from kd.neuron_surgery_baselines import baseline_configs
from kd.neuron_surgery_matrix import (matrix_jobs, matrix_spec, materialize_matrix_plan,
                                      summarize_matrix)
from kd.neuron_surgery_study import (NeuronSurgerySpec, load_neuron_surgery_spec,
                                     resolve_target_classes, resolved_dataset_profile)
from kd.models import create_model


def record(index, label, *, correct=True):
    return {"index": index, "true_label": label,
            "predicted_label": label if correct else (label + 1) % 10,
            "correct": correct, "confidence": 0.8, "nll": 0.2}


class TargetPolicyTests(unittest.TestCase):
    def test_frequency_targets_use_lowest_half_with_label_tie_break(self):
        spec = NeuronSurgerySpec(
            target_classes=None, target_class_policy="frequency_tail",
            imbalance_factor=10, dataset_num_classes=5,
            expected_train_counts=None, expected_target_counts=None)
        self.assertEqual(resolve_target_classes(spec, [100, 20, 20, 5, 5]), (3, 4, 1))

    def test_balanced_policy_targets_every_class(self):
        spec = NeuronSurgerySpec(
            target_classes=None, target_class_policy="all", imbalance_factor=1,
            dataset_num_classes=4, expected_train_counts=None,
            expected_target_counts=None)
        self.assertEqual(resolve_target_classes(spec, [10, 10, 10, 10]), (0, 1, 2, 3))

    def test_profile_and_factor_must_agree(self):
        with self.assertRaisesRegex(ValueError, "profile and imbalance factor"):
            NeuronSurgerySpec(dataset_profile="lt-if10", imbalance_factor=50).validate()

    def test_legacy_factor_spec_keeps_balanced_profile_representation(self):
        spec = NeuronSurgerySpec(imbalance_factor=100)
        self.assertEqual(resolved_dataset_profile(spec), "balanced")


class SparseClassFallbackTests(unittest.TestCase):
    def test_companion_fallback_is_deterministic_and_fixed_width(self):
        records = [record(0, 0, correct=False), record(1, 0, correct=False)]
        embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        result = build_companion_records(
            records, embeddings, [dict(records[0])], companions=5,
            allow_sparse_fallback=True)
        self.assertEqual(result[0]["companion_policy"], "same_class_fallback")
        self.assertEqual([row["index"] for row in result[0]["companions"]], [1] * 5)

    def test_singleton_target_uses_gradient_only_fallback(self):
        model = create_model("cifar_teacher", 2).eval()
        records = [record(0, 0, correct=False)]
        pairs = build_companion_records(
            records, np.asarray([[1.0, 0.0]], dtype=np.float32),
            [dict(records[0])], companions=5, allow_sparse_fallback=True)
        dataset = TensorDataset(torch.randn(1, 3, 32, 32), torch.tensor([0]))
        scores = differential_channel_scores(model, dataset, pairs, torch.device("cpu"),
                                             batch_size=1)
        self.assertEqual(pairs[0]["companion_policy"], "self_gradient_fallback")
        self.assertTrue(np.isfinite(scores["stage2"]).all())
        self.assertGreater(float(scores["stage2"].sum()), 0)

    def test_sparse_preservation_omits_classes_without_correct_examples(self):
        records = [record(0, 0), record(1, 0), record(2, 1, correct=False)]
        selected = select_preservation_indices(
            records, [], per_class_cap=1, seed=3, allow_sparse_fallback=True)
        self.assertEqual(len(selected), 1)
        self.assertIn(selected[0], {0, 1})


class MatrixPlanTests(unittest.TestCase):
    def test_default_matrix_has_every_dataset_profile_pair(self):
        jobs = matrix_jobs()
        self.assertEqual(len(jobs), 20)
        self.assertEqual(len({job["id"] for job in jobs}), 20)
        self.assertEqual({job["dataset"] for job in jobs},
                         {"cifar10", "cifar100", "svhn", "cinic10", "gtsrb"})
        self.assertEqual({job["profile"] for job in jobs},
                         {"balanced", "lt-if10", "lt-if50", "lt-if100"})

    def test_specs_carry_catalog_profile_and_baselines_preserve_it(self):
        spec = matrix_spec("cifar100", "lt-if50")
        teacher, students = baseline_configs("runs/example", "data", "cuda:0", spec)
        self.assertEqual(teacher.data.dataset_profile, "lt-if50")
        self.assertEqual(teacher.data.imbalance_factor, 1.0)
        self.assertEqual(teacher.data.num_classes, 100)
        self.assertEqual(len(students), 3)

    def test_plan_is_idempotent_and_configs_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "plan"
            first = materialize_matrix_plan(
                output, datasets=["svhn"], profiles=["balanced", "lt-if100"])
            second = materialize_matrix_plan(
                output, datasets=["svhn"], profiles=["balanced", "lt-if100"])
            self.assertEqual(first, second)
            rows = (output / "jobs.tsv").read_text().splitlines()
            self.assertEqual(len(rows), 2)
            for row in rows:
                spec = load_neuron_surgery_spec(row.split("\t")[4])
                self.assertEqual(spec.dataset_source, "svhn")

    def test_summary_keeps_dataset_jobs_separate(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            plan = root / "plan"
            output = root / "runs"
            materialize_matrix_plan(
                plan, datasets=["cifar10"], profiles=["balanced", "lt-if10"])
            complete = output / "cifar10" / "balanced" / "studies" / "confirmatory"
            complete.mkdir(parents=True)
            (complete / "comparison.json").write_text(json.dumps({
                "status": "complete", "success": {"passed": True},
                "aggregate": {
                    "accuracy_delta": {"mean": -0.001},
                    "tail_recall_delta": {"mean": 0.02},
                },
            }))
            report = summarize_matrix(plan, output)
            self.assertEqual(report["job_count"], 2)
            self.assertEqual(report["complete_count"], 1)
            self.assertEqual(report["successful_transfer_count"], 1)
            self.assertFalse(report["all_jobs_finished"])
            self.assertTrue((output / "matrix_summary.md").is_file())


if __name__ == "__main__":
    unittest.main()
