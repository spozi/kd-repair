from contextlib import redirect_stdout
from dataclasses import replace
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from kd.adaptive_focal import AdaptiveFocalController, AdaptiveFocalLoss
from kd.checkpoints import fingerprint
from kd.config import (BenchmarkConfig, DataConfig, DistillationConfig, ExperimentConfig,
                       ModelConfig, SupervisedLossConfig, TrainConfig, from_dict, load_config)
from kd.engine import DistillationSystem, Trainer, run_experiment
from kd.losses import DistillationObjective
from kd.metrics import evaluate
from kd.models import ModelOutput, create_model


class AdaptiveLossTests(unittest.TestCase):
    def test_independent_probability_formula_and_dual_selection(self):
        rows = [[.7, .2, .1], [.6, .3, .1], [.6, .3, .1], [.4, .4, .2]]
        labels = [0, 1, 2, 0]  # Top, middle, lowest, and tied target.
        for method in ("adafocal", "adadualfocal"):
            loss = AdaptiveFocalLoss(SupervisedLossConfig(method=method, bins=1))
            loss.controller.gamma.fill_(2)
            expected = []
            for row, label in zip(rows, labels):
                p = row[label]
                dual = max([v for v in row if v < p], default=0) if method == "adadualfocal" else 0
                expected.append(-(1-p+dual)**2 * math.log(p))
            actual = loss(torch.tensor(rows, dtype=torch.float64).log(), torch.tensor(labels))
            self.assertAlmostEqual(actual.item(), sum(expected)/4, places=12)

    def test_zero_gamma_is_cross_entropy_and_inverse_formula(self):
        logits = torch.tensor([[2., 0., -1.], [0., 1., 2.]], dtype=torch.float64)
        labels = torch.tensor([0, 1])
        for method in ("adafocal", "adadualfocal"):
            loss = AdaptiveFocalLoss(SupervisedLossConfig(method=method, bins=1))
            loss.controller.gamma.zero_()
            torch.testing.assert_close(loss(logits, labels), F.cross_entropy(logits, labels))
            loss.controller.gamma.fill_(-1.5)
            p = logits.softmax(1).gather(1, labels[:, None]).flatten()
            torch.testing.assert_close(loss(logits, labels), -((1+p)**1.5*p.log()).mean())

    def test_gradients_and_extreme_confidence_for_both_branches(self):
        for method in ("adafocal", "adadualfocal"):
            loss = AdaptiveFocalLoss(SupervisedLossConfig(method=method, bins=1))
            for gamma in (.2, 2., -1.5):
                loss.controller.gamma.fill_(gamma)
                logits = torch.tensor([[1.7, .1, -.5], [.9, 1.3, -.2]], dtype=torch.float64, requires_grad=True)
                labels = torch.tensor([0, 1])
                self.assertTrue(torch.autograd.gradcheck(lambda x: loss(x, labels), (logits,)))
                for target in (0, 1):
                    extreme = torch.tensor([[10000., -10000.]], requires_grad=True)
                    value = loss(extreme, torch.tensor([target]))
                    value.backward()
                    self.assertTrue(torch.isfinite(value))
                    self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_assignment_uses_true_class_probability(self):
        loss = AdaptiveFocalLoss(SupervisedLossConfig(method="adafocal", bins=2))
        loss.controller.gamma.copy_(torch.tensor([1., 3.]))
        p = torch.tensor([[.8, .2]], dtype=torch.float64)
        self.assertAlmostEqual(loss(p.log(), torch.tensor([1])).item(), -.8*math.log(.2))
        self.assertAlmostEqual(loss(p.log(), torch.tensor([0])).item(), -.2**3*math.log(.8))

    def test_kd_composition_warmup_and_teacher_isolation(self):
        for method in ("adafocal", "adadualfocal"):
            for kd in ("kd", "dkd"):
                loss = AdaptiveFocalLoss(SupervisedLossConfig(method=method, bins=2))
                config = DistillationConfig(method=kd, weight=.4, warmup_epochs=4)
                objective = DistillationObjective(config, supervised_loss=loss)
                s = ModelOutput(torch.tensor([[2., 0.], [.4, 1.]], requires_grad=True))
                t = ModelOutput(torch.tensor([[3., 0.], [1., 2.]], requires_grad=True))
                labels = torch.tensor([0, 1])
                result = objective(s, t, labels, 0)
                weight = .6 if kd == "kd" else 1.
                torch.testing.assert_close(result["total"], weight*loss(s.logits, labels)+.1*result["response"])
                torch.testing.assert_close(result["ce"], F.cross_entropy(s.logits, labels))
                result["total"].backward()
                self.assertIsNone(t.logits.grad)
                self.assertGreater(s.logits.grad.abs().sum().item(), 0)


class AdaptiveControllerTests(unittest.TestCase):
    def test_signed_error_update_retains_memory_when_calibrated(self):
        c = AdaptiveFocalController(SupervisedLossConfig(method="adafocal", bins=1))
        c.gamma.fill_(2)
        conf = torch.full((4,), .9, dtype=torch.float64)
        decision = c.observe(1, conf, torch.tensor([1, 1, 1, 0]))
        self.assertAlmostEqual(c.gamma.item(), 2*math.exp(.15), places=6)
        self.assertEqual(decision["used"]["gamma"], [2])
        before = c.gamma.clone()
        c.observe(2, torch.full((4,), .75), torch.tensor([1, 1, 1, 0]))
        torch.testing.assert_close(c.gamma, before, rtol=0, atol=0)

    def test_switches_both_directions_and_clips_without_overflow(self):
        c = AdaptiveFocalController(SupervisedLossConfig(method="adafocal", bins=1, update_rate=10000))
        c.observe(1, torch.tensor([1.]), torch.tensor([0]))
        self.assertAlmostEqual(c.gamma.item(), 20, places=5)
        c.observe(2, torch.tensor([0.]), torch.tensor([1]))
        self.assertAlmostEqual(c.gamma.item(), -.2)
        c.observe(3, torch.tensor([0.]), torch.tensor([1]))
        self.assertAlmostEqual(c.gamma.item(), -2)
        c.observe(4, torch.tensor([1.]), torch.tensor([0]))
        self.assertAlmostEqual(c.gamma.item(), .2)

    def test_quantile_bins_ties_and_empty_bins_have_consistent_assignment(self):
        c = AdaptiveFocalController(SupervisedLossConfig(method="adafocal", bins=2))
        r = c.observe(1, torch.tensor([.55, .6, .8, .9]), torch.tensor([1, 1, 0, 0]))
        self.assertEqual(r["counts"], [2, 2])
        self.assertLess(c.gamma[0], 1)
        self.assertGreater(c.gamma[1], 1)
        before = c.gamma.clone()
        r = c.observe(2, torch.full((4,), .7), torch.tensor([1, 0, 1, 0]))
        self.assertEqual(r["counts"], [4, 0])
        self.assertEqual(c.gamma[1], before[1])
        torch.testing.assert_close(c.gamma_for(torch.tensor([.7, .71])), c.gamma)

    def test_feedback_rejects_bad_data_without_changing_state(self):
        c = AdaptiveFocalController(SupervisedLossConfig(method="adafocal", bins=2))
        before = c.snapshot()
        for confidence, correct in (([float("nan")], [1]), ([1.1], [1]), ([.5], [.2]), ([], [])):
            with self.assertRaises(ValueError):
                c.observe(1, torch.tensor(confidence), torch.tensor(correct))
            self.assertEqual(c.snapshot(), before)
        with self.assertRaises(ValueError):
            c.observe(2, torch.tensor([.5]), torch.tensor([1]))

    def test_validation_emits_top_confidence_and_correctness_and_restores_mode(self):
        class Fixed(torch.nn.Module):
            def forward(self, images):
                return ModelOutput(images)
        model = Fixed().train()
        loader = DataLoader(TensorDataset(torch.tensor([[2., 0.], [2., 0.]]), torch.tensor([0, 1])), batch_size=1)
        feedback = []
        actual = evaluate(model, loader, torch.device("cpu"),
                          confidence_observer=lambda p, y: feedback.append((p, y)))
        self.assertTrue(model.training)
        self.assertEqual(torch.cat([v[1] for v in feedback]).tolist(), [1, 0])
        self.assertAlmostEqual(torch.cat([v[0] for v in feedback]).mean().item(), actual["mean_confidence"])


class AdaptiveIntegrationTests(unittest.TestCase):
    def test_config_defaults_validation_and_examples(self):
        self.assertEqual(from_dict({}).supervised_loss.method, "cross_entropy")
        for raw in ({"method": "invalid"}, {"bins": 0}, {"bins": True}, {"gamma_min": 0},
                    {"gamma_max": .5}, {"switch_threshold": 0}, {"update_rate": float("nan")}):
            with self.assertRaises(ValueError):
                from_dict({"supervised_loss": raw})
        with self.assertRaises(ValueError):
            from_dict({"distillation": {"method": "kd", "weight": 1}, "supervised_loss": {"method": "adafocal"}})
        for method in ("adafocal", "adadualfocal"):
            config = load_config(f"configs/cifar10_kd_{method}.toml")
            self.assertEqual(config.supervised_loss.method, method)
            self.assertEqual(config.distillation.method, "kd")
            self.assertFalse(config.calibration.enabled)

    def test_both_methods_freeze_teacher_weights_and_batchnorm(self):
        for method in ("adafocal", "adadualfocal"):
            student, teacher = create_model("tiny_small", 4), create_model("tiny_medium", 4)
            loss = AdaptiveFocalLoss(SupervisedLossConfig(method=method))
            objective = DistillationObjective(DistillationConfig(method="kd"), supervised_loss=loss)
            system = DistillationSystem(student, teacher, objective)
            before = {k: v.clone() for k, v in teacher.state_dict().items()}
            student_before = student.classifier.weight.clone()
            optimizer = torch.optim.SGD([p for p in system.parameters() if p.requires_grad], lr=.1)
            loader = DataLoader(TensorDataset(torch.randn(4, 3, 32, 32), torch.arange(4)), batch_size=4)
            Trainer(system, optimizer, torch.device("cpu")).train_epoch(loader, 0)
            self.assertFalse(teacher.training)
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            for k, v in teacher.state_dict().items():
                torch.testing.assert_close(v, before[k], rtol=0, atol=0)
            self.assertFalse(torch.equal(student_before, student.classifier.weight))

    def test_real_feedback_kd_resume_and_student_only_export(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            config = ExperimentConfig(name="teacher", output_dir=root,
                data=DataConfig(train_samples=16, val_samples=8, test_samples=8),
                student=ModelConfig("tiny_medium"), distillation=DistillationConfig(method="supervised"),
                train=TrainConfig(epochs=3, batch_size=8, device="cpu"),
                benchmark=BenchmarkConfig(warmup=0, iterations=1))
            teacher = run_experiment(config)
            teacher_hash = fingerprint(teacher["checkpoint"])
            original = Trainer.train_epoch

            def interrupt(trainer, loader, epoch):
                if epoch == 1:
                    raise InterruptedError()
                return original(trainer, loader, epoch)

            for method in ("adafocal", "adadualfocal"):
                cfg = replace(config, name=method, student=ModelConfig("tiny_small"),
                    teacher=ModelConfig("tiny_medium", teacher["checkpoint"]),
                    distillation=DistillationConfig(method="kd"),
                    supervised_loss=SupervisedLossConfig(method=method, bins=3))
                full = run_experiment(cfg)
                resumed = replace(cfg, name=method+"_resumed")
                with patch.object(Trainer, "train_epoch", interrupt), self.assertRaises(InterruptedError):
                    run_experiment(resumed)
                path = Path(root)/resumed.name/"last.pt"
                run_experiment(resumed, resume=str(path))
                expected = torch.load(Path(root)/cfg.name/"last.pt", weights_only=True)
                actual = torch.load(path, weights_only=True)
                for section in ("student", "objective"):
                    for k, v in expected[section].items():
                        torch.testing.assert_close(v, actual[section][k], rtol=0, atol=0)
                for i, h in enumerate(actual["history"]):
                    self.assertEqual(h["adaptive_focal"], expected["history"][i]["adaptive_focal"])
                    self.assertEqual(h["train"]["total"], expected["history"][i]["train"]["total"])
                    self.assertEqual(sum(h["adaptive_focal"]["counts"]), 8)
                    if i:
                        self.assertEqual(h["adaptive_focal"]["used"], actual["history"][i-1]["adaptive_focal"]["next"])
                self.assertEqual(actual["objective"]["supervised_loss.controller.last_epoch"].item(), 3)
                artifact = torch.load(full["checkpoint"], weights_only=True)
                self.assertNotIn("objective", artifact)
                self.assertEqual(fingerprint(teacher["checkpoint"]), teacher_hash)
