import itertools
import math
import unittest

import torch

from kd.cpc import PairwiseCalibrationLoss


def reference_components(logits, labels):
    """Independent scalar loops and log-add-exp identity, not tensor masks."""
    def softplus(value):
        return max(0.0, value) + math.log1p(math.exp(-abs(value)))

    discrimination, exclusion = [], []
    for row, label in zip(logits.tolist(), labels.tolist()):
        alternatives = [index for index in range(len(row)) if index != label]
        discrimination.append(sum(softplus(row[index] - row[label])
                                  for index in alternatives) / len(alternatives))
        pairs = list(itertools.combinations(alternatives, 2))
        exclusion.append(sum((softplus(row[left] - row[right])
                              + softplus(row[right] - row[left])) / 2
                             for left, right in pairs) / len(pairs) if pairs else 0.0)
    return {"discrimination": sum(discrimination) / len(labels),
            "exclusion": sum(exclusion) / len(labels)}


class PairwiseCalibrationLossTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([[2., -1., .5, 3.], [-2., 1., 4., 0.]], dtype=torch.float64)
        self.labels = torch.tensor([0, 2])
        self.loss = PairwiseCalibrationLoss(discrimination_weight=.2, exclusion_weight=.3)

    def test_components_and_weighting_match_independent_reference(self):
        expected = reference_components(self.logits, self.labels)
        actual = self.loss.components(self.logits, self.labels)
        for name in expected:
            self.assertAlmostEqual(actual[name].item(), expected[name], places=12)
        self.assertAlmostEqual(self.loss(self.logits, self.labels).item(),
                               .2 * expected["discrimination"] + .3 * expected["exclusion"], places=12)

    def test_cifar_pair_counts_masks_and_normalization(self):
        logits = torch.arange(100, dtype=torch.float64).reshape(10, 10) / 7
        labels = torch.arange(10)
        actual = self.loss.components(logits, labels)
        expected = reference_components(logits, labels)
        left, right = self.loss._pair_indices
        self.assertEqual(len(left), 45)
        self.assertTrue(bool((left < right).all()))
        for label in labels:
            self.assertEqual(int(((left == label) | (right == label)).sum()), 9)
            self.assertEqual(int(((left != label) & (right != label)).sum()), 36)
        for name in actual:
            self.assertAlmostEqual(actual[name].item(), expected[name], places=12)

    def test_exclusion_minimum_and_target_gradient_is_zero(self):
        logits = torch.tensor([[100., 2., 2., 2.], [-7., -7., -100., -7.]],
                              dtype=torch.float64, requires_grad=True)
        exclusion = self.loss.components(logits, self.labels)["exclusion"]
        self.assertAlmostEqual(exclusion.item(), math.log(2), places=12)
        exclusion.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
        unequal = self.logits.clone().requires_grad_()
        self.loss.components(unequal, self.labels)["exclusion"].backward()
        torch.testing.assert_close(unequal.grad.gather(1, self.labels[:, None]),
                                   torch.zeros((2, 1), dtype=torch.float64))

    def test_binary_exclusion_is_differentiable_zero(self):
        logits = torch.tensor([[9., -2.], [2., 8.]], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 1])
        actual = self.loss.components(logits, labels)
        self.assertEqual(actual["exclusion"].item(), 0)
        actual["exclusion"].backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
        self.assertAlmostEqual(actual["discrimination"].item(),
                               reference_components(logits, labels)["discrimination"], places=12)

    def test_float64_component_gradients(self):
        logits = self.logits.clone().requires_grad_()
        for name in ("discrimination", "exclusion"):
            with self.subTest(component=name):
                self.assertTrue(torch.autograd.gradcheck(
                    lambda values: self.loss.components(values, self.labels)[name], (logits,)))

    def test_large_logits_and_low_precision_have_finite_gradients(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                logits = torch.tensor([[10000., -10000., 0.], [-10000., 10000., 10000.]],
                                      dtype=dtype, requires_grad=True)
                value = self.loss(logits, torch.tensor([1, 0]))
                value.backward()
                self.assertTrue(bool(torch.isfinite(value)))
                self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_batch_duplication_class_permutation_and_common_shift_invariance(self):
        expected = self.loss.components(self.logits, self.labels)
        permutation = torch.tensor([3, 0, 2, 1])
        inverse = torch.argsort(permutation)
        inputs = [(self.logits.repeat(3, 1), self.labels.repeat(3)),
                  (self.logits[:, permutation], inverse[self.labels]),
                  (self.logits + torch.tensor([[100.], [-22.]]), self.labels)]
        for logits, labels in inputs:
            actual = self.loss.components(logits, labels)
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], rtol=1e-12, atol=1e-12)

    def test_cached_indices_and_checkpoint_restore_have_no_trainable_state(self):
        expected = self.loss(self.logits, self.labels)
        pair_buffer = self.loss._pair_indices
        self.loss(self.logits, self.labels)
        self.assertIs(self.loss._pair_indices, pair_buffer)
        self.assertEqual(list(self.loss.parameters()), [])
        self.assertEqual(dict(self.loss.state_dict()), {})
        restored = PairwiseCalibrationLoss(.2, .3)
        restored.load_state_dict(self.loss.state_dict(), strict=True)
        torch.testing.assert_close(restored(self.logits, self.labels), expected)
        restored(torch.zeros(1, 10), torch.tensor([0]))
        self.assertEqual(tuple(restored._pair_indices.shape), (2, 45))
        restored.to(dtype=torch.float64)
        self.assertEqual(restored._pair_indices.dtype, torch.long)

    def test_warmup_ramp_matches_project_convention_and_rejects_invalid_values(self):
        # Default (zero) reproduces the original always-on behavior exactly.
        always_on = PairwiseCalibrationLoss(.2, .3)
        self.assertEqual(always_on.warmup_epochs, 0)
        for epoch in (0, 1, 4, 19):
            self.assertEqual(always_on.ramp(epoch), 1.0)
        ramped = PairwiseCalibrationLoss(.2, .3, warmup_epochs=5)
        for epoch, expected in ((0, .2), (1, .4), (4, 1.0), (19, 1.0)):
            self.assertAlmostEqual(ramped.ramp(epoch), expected, places=12)
        for invalid in (-1, 1.5, True, False):
            with self.assertRaises(ValueError):
                PairwiseCalibrationLoss(warmup_epochs=invalid)

    def test_zero_weights_and_input_validation(self):
        self.assertEqual(PairwiseCalibrationLoss(0, 0)(self.logits, self.labels).item(), 0)
        for weight in (-1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                PairwiseCalibrationLoss(discrimination_weight=weight)
            with self.assertRaises(ValueError):
                PairwiseCalibrationLoss(exclusion_weight=weight)
        invalid = [(torch.zeros(0, 3), torch.zeros(0, dtype=torch.long)),
                   (torch.zeros(2, 1), torch.zeros(2, dtype=torch.long)),
                   (torch.zeros(2, 3, 4), self.labels),
                   (torch.zeros(2, 3, dtype=torch.long), self.labels),
                   (self.logits, self.labels.float()),
                   (self.logits, self.labels[:, None]),
                   (self.logits, torch.tensor([-1, 0])),
                   (self.logits, torch.tensor([0, 4]))]
        for logits, labels in invalid:
            with self.subTest(logits_shape=tuple(logits.shape), labels=labels):
                with self.assertRaises(ValueError):
                    self.loss(logits, labels)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_buffer_transfer_forward_backward_and_restore(self):
        # Populate the derived buffer on CPU, then transfer the module to MPS.
        self.loss(self.logits, self.labels)
        device_loss = self.loss.to("mps")
        self.assertEqual(device_loss._pair_indices.device.type, "mps")
        logits = self.logits.float().to("mps").requires_grad_()
        labels = self.labels.to("mps")
        actual = device_loss(logits, labels)
        actual.backward()
        self.assertTrue(bool(torch.isfinite(actual)))
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(logits.grad.abs().sum().item(), 0)
        expected = reference_components(self.logits, self.labels)
        self.assertAlmostEqual(actual.item(), .2 * expected["discrimination"]
                               + .3 * expected["exclusion"], places=5)
        # Check checkpoint loading before the first MPS call, and rebuilding
        # the pair indices on a changed class count without host-only kernels.
        restored = PairwiseCalibrationLoss(.2, .3).to("mps")
        restored.load_state_dict(device_loss.state_dict(), strict=True)
        torch.testing.assert_close(restored(logits.detach(), labels), actual.detach())
        ten_class = torch.randn(3, 10, device="mps", requires_grad=True)
        restored(ten_class, torch.tensor([0, 5, 9], device="mps")).backward()
        self.assertEqual(tuple(restored._pair_indices.shape), (2, 45))
        self.assertTrue(bool(torch.isfinite(ten_class.grad).all()))


if __name__ == "__main__":
    unittest.main()
