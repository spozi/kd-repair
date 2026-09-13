import math
import unittest

import torch
from torch.nn import functional as F

from kd.config import DistillationConfig
from kd.losses import DecoupledKD, DistillationObjective, LogitsKD, StageFeatureLoss
from kd.models import ModelOutput


def softmax(values):
    exps = [math.exp(v - max(values)) for v in values]
    return [v / sum(exps) for v in exps]


def kl(teacher, student):
    return sum(p * math.log(p / q) for p, q in zip(teacher, student))


class ResponseLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.student = torch.tensor([[2.0, -1.0, 0.5], [0.1, 0.8, -0.5]], requires_grad=True)
        self.teacher = torch.tensor([[-0.5, 1.0, 2.0], [1.0, 0.0, -0.3]], requires_grad=True)
        self.target = torch.tensor([0, 1])

    def test_classical_matches_independent_probability_calculation(self):
        for temperature in (2, 4, 8):
            expected = sum(kl(softmax([v / temperature for v in t]),
                              softmax([v / temperature for v in s]))
                           for s, t in zip(self.student.tolist(), self.teacher.tolist())) / 2 * temperature**2
            actual = LogitsKD(temperature)(self.student, self.teacher, self.target)
            self.assertAlmostEqual(actual.item(), expected, places=5)

    def test_dkd_matches_independent_target_and_nontarget_calculation(self):
        temperature, alpha, beta = 4, 1.3, 5.0
        expected = 0.0
        for s, t, label in zip(self.student.tolist(), self.teacher.tolist(), self.target.tolist()):
            sp, tp = softmax([v / temperature for v in s]), softmax([v / temperature for v in t])
            target_part = kl([tp[label], 1 - tp[label]], [sp[label], 1 - sp[label]])
            other_part = kl([p / (1 - tp[label]) for i, p in enumerate(tp) if i != label],
                            [p / (1 - sp[label]) for i, p in enumerate(sp) if i != label])
            expected += (alpha * target_part + beta * other_part) * temperature**2 / 2
        actual = DecoupledKD(temperature, alpha, beta)(self.student, self.teacher, self.target)
        self.assertAlmostEqual(actual.item(), expected, places=4)

    def test_identity_is_zero_and_teacher_never_receives_gradient(self):
        for strategy in (LogitsKD(4), DecoupledKD(4)):
            with self.subTest(strategy=type(strategy).__name__):
                loss = strategy(self.student, self.teacher, self.target)
                loss.backward()
                self.assertIsNone(self.teacher.grad)
                self.assertTrue(torch.isfinite(self.student.grad).all())
                self.assertAlmostEqual(strategy(self.student, self.student, self.target).item(), 0, places=5)

    def test_extreme_confidence_and_two_classes_remain_finite(self):
        for classes in (2, 3):
            s = torch.tensor([[10000.0, -10000.0, 0.0][:classes]], requires_grad=True)
            t = -s.detach()
            loss = DecoupledKD(2)(s, t, torch.tensor([0]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(s.grad).all())

    def test_binary_nontarget_loss_is_zero(self):
        loss = DecoupledKD(4, alpha=0, beta=8)(self.student[:, :2], self.teacher[:, :2], self.target)
        self.assertEqual(loss.item(), 0)

    def test_dkd_keeps_incorrect_teacher_examples(self):
        loss = DecoupledKD(4)(self.student[:1], self.teacher[:1], self.target[:1])
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertGreater(self.student.grad[0].abs().sum().item(), 0)

    def test_global_loss_rejects_dense_prediction_logits(self):
        with self.assertRaisesRegex(ValueError, "matching"):
            DecoupledKD(4)(torch.zeros(2, 3, 4, 4), torch.zeros(2, 3, 4, 4), self.target)

    def test_weight_semantics_and_warmup(self):
        s, t = ModelOutput(self.student), ModelOutput(self.teacher)
        ce = F.cross_entropy(self.student, self.target)
        for method in ("kd", "dkd"):
            config = DistillationConfig(method=method, weight=0.25, warmup_epochs=4)
            objective = DistillationObjective(config)
            first = objective(s, t, self.target, epoch=0)
            full = objective(s, t, self.target, epoch=3)
            ce_weight = 0.75 if method == "kd" else 1
            torch.testing.assert_close(first["total"], ce_weight * ce + 0.25 * 0.25 * first["response"])
            torch.testing.assert_close(full["total"], ce_weight * ce + 0.25 * full["response"])


class FeatureLossTests(unittest.TestCase):
    def test_heterogeneous_channels_and_resolutions_receive_gradients(self):
        loss = StageFeatureLoss({"shallow": 4}, {"deep": 9}, (("shallow", "deep"),))
        student = torch.randn(2, 4, 8, 8, requires_grad=True)
        teacher = torch.randn(2, 9, 4, 4, requires_grad=True)
        loss({"shallow": student}, {"deep": teacher}).backward()
        self.assertGreater(student.grad.abs().sum().item(), 0)
        self.assertGreater(loss.projections[0].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(teacher.grad)

    def test_unknown_semantic_stage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown feature"):
            StageFeatureLoss({"stage2": 4}, {"stage3": 8}, (("stage1", "stage3"),))

