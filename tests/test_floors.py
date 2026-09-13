import unittest

import numpy as np
import torch

from kd.calibration_metrics import conditional_calibration_null, extended_prediction_metrics
from kd.floors import (FAMILIES, PREDICTION_PRESERVING, apply_family, fit_family,
                       oracle_bound, oracle_table)


def synthetic(samples=4000, classes=5, temperature=1.0, seed=0):
    """Logits whose softmax at `temperature` generates the labels, so the
    calibrated model is exactly correct and the true optimal temperature is known."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=2.0, size=(samples, classes))
    probabilities = np.exp(logits / temperature)
    probabilities /= probabilities.sum(1, keepdims=True)
    labels = np.array([rng.choice(classes, p=row) for row in probabilities])
    return labels, logits


class OracleBoundTests(unittest.TestCase):
    def setUp(self):
        self.labels, self.logits = synthetic()
        self.classes = [f"c{i}" for i in range(5)]

    def test_recovers_known_temperature_on_calibrated_data(self):
        labels, logits = synthetic(temperature=2.5, seed=1)
        (parameters,) = fit_family(logits, labels, "temperature")
        self.assertAlmostEqual(float(torch.exp(parameters)), 2.5, delta=0.15)

    def test_oracle_is_at_least_as_good_as_an_honest_fit(self):
        """Fitting on the scored split cannot be beaten by fitting elsewhere."""
        half = len(self.labels) // 2
        for family in ("temperature", "vector", "matrix"):
            with self.subTest(family=family):
                honest = fit_family(self.logits[:half], self.labels[:half], family)
                honest_nll = extended_prediction_metrics(
                    self.labels[half:], apply_family(self.logits[half:], honest, family),
                    self.classes, include_curve=False)["nll"]
                oracle = fit_family(self.logits[half:], self.labels[half:], family)
                oracle_nll = extended_prediction_metrics(
                    self.labels[half:], apply_family(self.logits[half:], oracle, family),
                    self.classes, include_curve=False)["nll"]
                self.assertLessEqual(oracle_nll, honest_nll + 1e-9)

    def test_ece_criterion_bounds_the_nll_criterion_on_ece(self):
        for family in PREDICTION_PRESERVING:
            with self.subTest(family=family):
                by_nll = oracle_bound(self.labels, self.logits, family, self.classes)
                by_ece = oracle_bound(self.labels, self.logits, family, self.classes,
                                      criterion="ece")
                self.assertLessEqual(by_ece["metrics"]["ece_15_bins"],
                                     by_nll["metrics"]["ece_15_bins"] + 1e-12)

    def test_scaling_families_preserve_predictions_and_others_may_not(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                record = oracle_bound(self.labels, self.logits, family, self.classes)
                if family in PREDICTION_PRESERVING:
                    self.assertTrue(record["predictions_preserved"])
                self.assertIn("never a reported result", record["interpretation"])

    def test_oracle_below_the_null_signals_an_exhausted_family(self):
        """On data that is already calibrated, no family can beat the null band."""
        probabilities = apply_family(self.logits, fit_family(self.logits, self.labels,
                                                             "temperature"), "temperature")
        null = conditional_calibration_null(self.labels, probabilities, self.classes,
                                            repetitions=200)
        oracle = oracle_bound(self.labels, self.logits, "temperature", self.classes,
                              criterion="ece")["metrics"]["ece_15_bins"]
        upper = null["metrics"]["ece_15_bins"]["null_central_95"][1]
        self.assertLessEqual(oracle, upper)

    def test_determinism_and_input_validation(self):
        first = oracle_bound(self.labels, self.logits, "matrix", self.classes)
        second = oracle_bound(self.labels, self.logits, "matrix", self.classes)
        self.assertEqual(first["metrics"]["nll"], second["metrics"]["nll"])
        self.assertEqual(len(oracle_table(self.labels, self.logits, self.classes)), len(FAMILIES))
        with self.assertRaises(ValueError):
            fit_family(self.logits, self.labels, "not_a_family")
        with self.assertRaises(ValueError):
            fit_family(self.logits, self.labels, "temperature", criterion="brier")
        with self.assertRaisesRegex(ValueError, "only defined for"):
            fit_family(self.logits, self.labels, "matrix", criterion="ece")
        with self.assertRaises(ValueError):
            fit_family(np.zeros((4, 1)), np.zeros(4, dtype=int), "temperature")
        with self.assertRaises(ValueError):
            fit_family(np.full((4, 3), np.nan), np.zeros(4, dtype=int), "temperature")


if __name__ == "__main__":
    unittest.main()
