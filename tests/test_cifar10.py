from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from kd.checkpoints import fingerprint, metadata
from kd.cifar_study import run_cifar_study, study_configs
from kd.config import DataConfig, ExperimentConfig, TrainConfig
from kd.data import build_data, image_transform, stratified_split
from kd.evaluation import paired_comparison, prediction_metrics, wilson_interval
from kd.models import create_model


class FakeCIFAR10(Dataset):
    calls = []

    def __init__(self, root, train=True, download=False, transform=None):
        self.calls.append(train)
        self.targets = np.repeat(np.arange(10), 5000 if train else 1000).tolist()
        self.classes = [f"class_{i}" for i in range(10)]
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = Image.fromarray(np.full((32, 32, 3), index % 256, dtype=np.uint8))
        return self.transform(image), self.targets[index]


class CIFARDataTests(unittest.TestCase):
    def test_stratified_split_exhaustive_disjoint_and_reproducible(self):
        labels = np.repeat(np.arange(10), 5000)
        train, val = stratified_split(labels, 0.1, 2026)
        self.assertEqual((len(train), len(val)), (45000, 5000))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(range(50000)))
        self.assertEqual(np.bincount(labels[val]).tolist(), [500] * 10)
        self.assertEqual((train, val), stratified_split(labels, 0.1, 2026))
        self.assertNotEqual(val, stratified_split(labels, 0.1, 2027)[1])

    @patch("kd.data.datasets.CIFAR10", FakeCIFAR10)
    def test_training_never_opens_official_test_and_seed_keeps_split(self):
        FakeCIFAR10.calls = []
        data = DataConfig(source="cifar10", num_classes=10)
        first = build_data(data, TrainConfig(seed=42), include_test=False)
        second = build_data(data, TrainConfig(seed=43), include_test=False)
        self.assertIsNone(first.test)
        self.assertTrue(all(FakeCIFAR10.calls))
        self.assertEqual(first.train.dataset.indices, second.train.dataset.indices)
        self.assertEqual(first.val.dataset.indices, second.val.dataset.indices)
        self.assertEqual(first.provenance["train_per_class"], [4500] * 10)
        self.assertEqual(first.provenance["split_sizes"], {"train": 45000, "val": 5000})
        test = build_data(data, TrainConfig(), include_test=True)
        self.assertEqual(len(test.test.dataset), 10000)
        self.assertFalse(FakeCIFAR10.calls[-1])

    def test_evaluation_preserves_all_32x32_pixels_and_fixed_normalization(self):
        data = DataConfig(source="cifar10", num_classes=10)
        array = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        actual = image_transform(data, training=False)(Image.fromarray(array))
        expected = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 127.5 - 1
        torch.testing.assert_close(actual, expected)
        self.assertEqual(metadata("cifar_student", [], 32, "cifar10")["preprocessing"]["mean"], [0.5] * 3)

    def test_rejects_wrong_cifar_shape_and_too_small_split(self):
        with self.assertRaisesRegex(ValueError, "CIFAR-10"):
            replace(ExperimentConfig(), data=DataConfig(source="cifar10", num_classes=4)).validate(require_teacher=False)
        with self.assertRaises(ValueError):
            stratified_split([0, 1], 0.1, 0)

    def test_models_have_distinct_capacity_and_compatible_stage_names(self):
        student, teacher = create_model("cifar_student", 10), create_model("cifar_teacher", 10)
        self.assertLess(sum(p.numel() for p in student.parameters()), sum(p.numel() for p in teacher.parameters()))
        self.assertEqual(set(student.feature_channels), set(teacher.feature_channels))
        for model in (student, teacher):
            model.eval()
            with torch.no_grad():
                result = model(torch.randn(2, 3, 32, 32), return_features=True)
            self.assertEqual(result.logits.shape, (2, 10))
            self.assertEqual(result.features["stage3"].shape[-2:], (4, 4))

    def test_study_has_equal_student_budget_and_fixed_teacher(self):
        teacher, students = study_configs("runs/example", "data", "cpu", 30, 20, (42, 43, 44))
        self.assertEqual(len(students), 9)
        self.assertEqual({run.train.epochs for run in students}, {20})
        self.assertEqual({run.data.split_seed for run in [teacher, *students]}, {2026})
        self.assertEqual({run.teacher.checkpoint for run in students}, {str(Path("runs/example/teacher/student.pt"))})
        self.assertEqual({run.distillation.temperature for run in students}, {4})


class DetailedEvaluationTests(unittest.TestCase):
    def test_confusion_f1_brier_and_nll_against_manual_values(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]])
        result = prediction_metrics(labels, probabilities, ["a", "b"])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["macro_f1"], 0.5)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["nll"], -sum(map(math.log, [0.9, 0.4, 0.8, 0.3])) / 4)
        self.assertAlmostEqual(result["brier_score"], 2 * (0.1**2 + 0.6**2 + 0.2**2 + 0.7**2) / 4)

    def test_wilson_interval_boundaries(self):
        self.assertAlmostEqual(wilson_interval(0, 10)[0], 0)
        self.assertAlmostEqual(wilson_interval(10, 10)[1], 1)
        self.assertLess(wilson_interval(500, 1000)[0], 0.5)
        self.assertGreater(wilson_interval(500, 1000)[1], 0.5)
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)

    def test_paired_mcnemar_exact_and_identical_models(self):
        labels = np.zeros(4, dtype=int)
        baseline = np.array([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]])
        candidate = baseline[:, ::-1]
        result = paired_comparison(labels, baseline, candidate)
        self.assertEqual(result["accuracy_delta"], 0.5)
        self.assertAlmostEqual(result["mcnemar_exact_two_sided_p"], 0.625)
        identical = paired_comparison(labels, baseline, baseline)
        self.assertEqual(identical["paired_bootstrap_95"], [0, 0])
        self.assertEqual(identical["mcnemar_exact_two_sided_p"], 1)

    def test_invalid_probabilities_rejected(self):
        with self.assertRaises(ValueError):
            prediction_metrics(np.array([0]), np.array([[float("nan"), 1]]), ["a", "b"])


class StudyLifecycleTests(unittest.TestCase):
    def test_test_phase_requires_frozen_selection_and_reports_are_reproducible(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "study"

            class MiniCIFAR10(FakeCIFAR10):
                def __init__(self, directory, train=True, download=False, transform=None):
                    if not train:
                        assert (output / "selection.json").exists(), "Test opened before selection was frozen"
                    self.targets = np.repeat(np.arange(10), 10 if train else 2).tolist()
                    self.classes = [f"class_{i}" for i in range(10)]
                    self.transform = transform

            with patch("kd.data.datasets.CIFAR10", MiniCIFAR10), redirect_stdout(io.StringIO()):
                report = run_cifar_study(str(output), "unused", "cpu", 1, 1, seeds=(42, 43))
                self.assertEqual(len(report["runs"]), 7)
                self.assertEqual(report["runs"]["teacher"]["test"]["samples"], 20)
                self.assertEqual(set(report["aggregate"]), {"supervised", "kd", "dkd"})
                frozen_selection = (output / "selection.json").read_text()
                with patch("kd.cifar_study.run_experiment", side_effect=AssertionError("Completed run retrained")):
                    resumed = run_cifar_study(str(output), "unused", "cpu", 1, 1, seeds=(42, 43))
                self.assertEqual(resumed["aggregate"], report["aggregate"])
                self.assertEqual((output / "selection.json").read_text(), frozen_selection)
                # Emulate an archived study created before optional calibration existed.
                protocol_path = output / "protocol.json"
                legacy = json.loads(protocol_path.read_text())
                for config in legacy["configs"]:
                    config.pop("calibration")
                    config.pop("supervised_loss")
                    config_path = output / config["name"] / "config.json"
                    saved = json.loads(config_path.read_text())
                    saved.pop("calibration")
                    saved.pop("supervised_loss")
                    config_path.write_text(json.dumps(saved))
                protocol_path.write_text(json.dumps(legacy))
                selection_path = output / "selection.json"
                selection = json.loads(selection_path.read_text())
                selection["protocol_sha256"] = fingerprint(protocol_path)
                selection_path.write_text(json.dumps(selection, indent=2) + "\n")
                with patch("kd.cifar_study.run_experiment", side_effect=AssertionError("Legacy run retrained")):
                    archived = run_cifar_study(str(output), "unused", "cpu", 1, 1, seeds=(42, 43))
                self.assertEqual(archived["aggregate"], report["aggregate"])
                self.assertEqual(json.loads(protocol_path.read_text()), legacy)
                self.assertEqual(json.loads(selection_path.read_text()), selection)
                with self.assertRaisesRegex(ValueError, "protocol differs"):
                    run_cifar_study(str(output), "unused", "cpu", 2, 1, seeds=(42, 43))
                with patch.dict(os.environ, {"MPLCONFIGDIR": str(Path(root) / "matplotlib")}):
                    try:
                        from kd.reporting import render_report
                        import matplotlib  # Optional reporting dependency.
                    except ImportError:
                        return
                    self.assertTrue(render_report(output).exists())
                self.assertTrue((output / "figures" / "learning-and-accuracy.png").exists())
                self.assertTrue((output / "figures" / "confusion-and-calibration.png").exists())
                self.assertEqual(json.loads((output / "progress.json").read_text())["phase"], "complete")


class LongTailedSamplingTests(unittest.TestCase):
    """Exponential imbalance over an existing split, without touching test data."""

    def setUp(self):
        # 4 classes, 100 examples each, in an interleaved label order.
        self.targets = [i % 4 for i in range(400)]
        self.indices = list(range(400))

    def test_factor_one_is_identity_and_preserves_existing_splits(self):
        from kd.data import long_tailed_subset
        self.assertEqual(long_tailed_subset(self.targets, self.indices, 1.0, 2026), self.indices)
        # A real CIFAR-10 split must be byte-identical to the balanced default.
        config = DataConfig(source="cifar10", root="data", num_classes=10, image_size=32,
                            horizontal_flip=True, validation_fraction=0.1, split_seed=2026)
        self.assertEqual(config.imbalance_factor, 1.0)

    def test_exponential_profile_head_tail_and_determinism(self):
        from kd.data import long_tailed_subset
        kept = long_tailed_subset(self.targets, self.indices, 100.0, 2026)
        counts = np.bincount(np.asarray(self.targets)[kept], minlength=4)
        self.assertEqual(counts[0], 100)                      # head keeps everything
        self.assertEqual(counts[3], 1)                        # tail keeps 100/100
        self.assertTrue(all(counts[i] >= counts[i + 1] for i in range(3)))
        for i, expected in enumerate(counts):
            self.assertEqual(expected, round(100 * 100.0 ** (-i / 3)))
        self.assertEqual(kept, long_tailed_subset(self.targets, self.indices, 100.0, 2026))
        self.assertNotEqual(kept, long_tailed_subset(self.targets, self.indices, 100.0, 7))
        self.assertEqual(kept, sorted(kept))
        self.assertEqual(len(set(kept)), len(kept))
        self.assertTrue(set(kept).issubset(set(self.indices)))

    def test_invalid_factors_are_rejected(self):
        from kd.data import long_tailed_subset
        for bad in (0.5, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                long_tailed_subset(self.targets, self.indices, bad, 2026)
        # A factor that would empty a class must fail loudly rather than silently.
        with self.assertRaises(ValueError):
            long_tailed_subset(self.targets, self.indices, 1e6, 2026)
        for bad in (0.5, float("nan"), True):
            with self.assertRaises(ValueError):
                ExperimentConfig(data=DataConfig(source="cifar10", num_classes=10, image_size=32,
                                                 imbalance_factor=bad)).validate(require_teacher=False)
        with self.assertRaisesRegex(ValueError, "cifar10 source only"):
            ExperimentConfig(data=DataConfig(imbalance_factor=10.0)).validate(require_teacher=False)
