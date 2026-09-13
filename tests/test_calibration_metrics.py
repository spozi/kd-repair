import itertools
import unittest

import numpy as np

from kd.calibration_metrics import (
    conditional_calibration_null,
    extended_prediction_metrics,
    paired_extended_comparison,
    paired_seed_comparison,
)


class CalibrationMetricsTests(unittest.TestCase):
    def test_hand_computed_ece_and_marginal_classwise(self):
        p = np.array([[.8, .2], [.6, .4], [.3, .7], [.5, .5]])
        labels = np.array([0, 1, 1, 0])
        metrics = extended_prediction_metrics(labels, p, ["a", "b"], bins=2)
        self.assertAlmostEqual(metrics["ece_2_bins"], .1)
        self.assertAlmostEqual(metrics["ece_l2_2_bins"], .1)
        self.assertAlmostEqual(metrics["classwise_calibration"][0]["ece"], .1)
        self.assertAlmostEqual(metrics["classwise_calibration"][1]["ece"], .15)
        self.assertAlmostEqual(metrics["classwise_ece_2_bins"], .125)
        group = metrics["predicted_class_calibration"][0]
        self.assertEqual(group["support"], 3)
        self.assertAlmostEqual(group["accuracy"], 2 / 3)
        self.assertAlmostEqual(group["mean_confidence"], 1.9 / 3)
        self.assertNotEqual(group["ece"], metrics["classwise_calibration"][0]["ece"])

    def test_l2_uses_squared_bin_gaps_with_count_weights(self):
        p = np.array([[.4, .35, .25], [.4, .35, .25], [.9, .05, .05]])
        labels = np.array([0, 1, 1])
        metrics = extended_prediction_metrics(labels, p, bins=2)
        self.assertAlmostEqual(metrics["ece_2_bins"], (2 * .1 + .9) / 3)
        self.assertAlmostEqual(metrics["ece_l2_2_bins"], np.sqrt((2 * .1**2 + .9**2) / 3))

    def test_equal_mass_keeps_ties_and_documents_empty_groups(self):
        p = np.tile([.8, .1, .1], (20, 1))
        labels = np.arange(20) % 3
        metrics = extended_prediction_metrics(labels, p, bins=15)
        self.assertEqual(metrics["equal_mass_effective_bins"], 1)
        self.assertEqual(metrics["equal_mass_bins"][0]["count"], 20)
        self.assertEqual(sum(row["count"] == 0 for row in metrics["equal_width_bins"]), 14)
        self.assertIsNone(metrics["predicted_class_calibration"][1]["accuracy"])
        self.assertIsNone(metrics["predicted_class_calibration"][1]["ece"])
        confidence = np.array([.9] * 7 + [.8] * 2 + [.6])
        more = extended_prediction_metrics(np.zeros(10, dtype=int), np.column_stack((confidence, 1 - confidence)), bins=10)
        self.assertEqual(more["equal_mass_effective_bins"], 3)
        self.assertEqual([row["count"] for row in more["equal_mass_bins"]], [7, 2, 1])
        self.assertAlmostEqual(more["ece_equal_mass_10_bins"], 1 - confidence.mean())

    def test_ranking_perfect_reversed_all_correct_and_all_wrong(self):
        p = np.array([[.9, .1], [.8, .2], [.7, .3], [.6, .4]])
        ordered = extended_prediction_metrics(np.array([0, 0, 1, 1]), p)
        reverse = extended_prediction_metrics(np.array([1, 1, 0, 0]), p)
        self.assertAlmostEqual(ordered["aurc"], (0 + 0 + 1 / 3 + .5) / 4)
        self.assertAlmostEqual(reverse["aurc"], (1 + 1 + 2 / 3 + .5) / 4)
        for label in (0, 1):
            result = extended_prediction_metrics(np.full(4, label), p)
            self.assertAlmostEqual(result["aurc"], label)
            self.assertAlmostEqual(result["selective_accuracy_80"], 1 - label)
            self.assertAlmostEqual(result["selective_accuracy_90"], 1 - label)
            np.testing.assert_allclose(result["risk_coverage_curve"]["risk"], label)

    def test_expected_tie_risk_equals_exhaustive_random_ordering(self):
        p = np.array([[.95, .05], [.8, .2], [.8, .2], [.8, .2], [.8, .2]])
        labels = np.array([1, 0, 1, 0, 0])
        metrics = extended_prediction_metrics(labels, p)
        risks, selective = [], []
        for order in itertools.permutations(range(1, 5)):
            errors = labels[np.array((0,) + order)]
            risks.append(np.cumsum(errors) / np.arange(1, 6))
            selective.append(1 - errors[:4].mean())
        np.testing.assert_allclose(metrics["risk_coverage_curve"]["risk"], np.mean(risks, axis=0))
        self.assertAlmostEqual(metrics["aurc"], np.mean(risks))
        self.assertAlmostEqual(metrics["selective_accuracy_80"], np.mean(selective))
        for permutation in ([4, 1, 3, 0, 2], [2, 4, 1, 3, 0]):
            reordered = extended_prediction_metrics(labels[permutation], p[permutation])
            for key in ("aurc", "selective_accuracy_80", "selective_accuracy_90"):
                self.assertAlmostEqual(metrics[key], reordered[key])

    def test_bootstrap_matches_explicit_resampling_with_ties_and_new_bins(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        b = np.array([[.9, .1], [.8, .2], [.8, .2], [.6, .4], [.55, .45], [.3, .7]])
        c = np.array([[.8, .2], [.3, .7], [.8, .2], [.3, .7], [.6, .4], [.1, .9]])
        indices = np.array([[0, 0, 0, 2, 5, 5], [1, 1, 3, 3, 4, 5], [0, 2, 2, 2, 3, 4]])
        result = paired_extended_comparison(labels, b, c, repetitions=3, bins=3, indices=indices)
        expected = []
        for index in indices:
            bm = extended_prediction_metrics(labels[index], b[index], bins=3, include_curve=False)
            cm = extended_prediction_metrics(labels[index], c[index], bins=3, include_curve=False)
            expected.append({key: cm[key] - bm[key] for key in result["metrics"]})
        for key, value in result["metrics"].items():
            np.testing.assert_allclose(value["paired_bootstrap_95"], np.quantile([row[key] for row in expected], [.025, .975]), atol=1e-14)

    def test_across_seed_pairing_uses_identical_images(self):
        p = np.array([[.9, .1], [.8, .2], [.6, .4], [.2, .8]])
        q = np.array([[.6, .4], [.1, .9], [.3, .7], [.9, .1]])
        labels = np.array([0, 1, 0, 1])
        # Opposite seed pairs cancel for EVERY shared-image replicate, whereas
        # separately bootstrapping each seed would create a spurious interval.
        result = paired_seed_comparison(labels, [p, q], [q, p], repetitions=40, seed=123)
        for row in result["metrics"].values():
            self.assertAlmostEqual(row["delta"], 0)
            np.testing.assert_allclose(row["paired_bootstrap_95"], [0, 0], atol=1e-14)
        repeated = paired_seed_comparison(labels, [p, q], [q, p], repetitions=40, seed=123)
        self.assertEqual(result, repeated)

    def test_null_is_deterministic_and_uses_coherent_categorical_labels(self):
        p = np.tile([.55, .3, .15], (30, 1))
        labels = np.arange(30) % 3
        repetitions, seed = 20, 734
        result = conditional_calibration_null(labels, p, repetitions=repetitions, seed=seed)
        self.assertEqual(result, conditional_calibration_null(labels, p, repetitions=repetitions, seed=seed))
        rng = np.random.default_rng(seed)
        expected = []
        for _ in range(repetitions):
            sampled = (rng.random((30, 1)) >= p.cumsum(1)).sum(1)
            expected.append(extended_prediction_metrics(sampled, p, include_curve=False))
        for key, row in result["metrics"].items():
            self.assertAlmostEqual(row["null_mean"], np.mean([value[key] for value in expected]))
        for group in range(3):
            self.assertAlmostEqual(result["classwise_calibration"][group]["ece"]["null_mean"],
                                   np.mean([value["classwise_calibration"][group]["ece"] for value in expected]))
        self.assertAlmostEqual(sum(row["target_support"]["null_mean"] for row in result["classwise_calibration"]), 30)
        self.assertIsNone(result["predicted_class_calibration"][1]["ece"]["null_mean"])

    def test_calibrated_degenerate_null_and_known_binomial_reference(self):
        p = np.array([[1., 0.], [0., 1.], [1., 0.]])
        labels = np.array([0, 1, 0])
        result = conditional_calibration_null(labels, p, repetitions=20)
        for row in result["metrics"].values():
            self.assertEqual(row["null_mean"], 0)
            self.assertEqual(row["null_central_95"], [0, 0])
        # Two fair binary observations: ECE is .5 with probability .5, else 0.
        result = conditional_calibration_null(np.array([0, 1]), np.full((2, 2), .5), repetitions=2000, seed=52)
        self.assertAlmostEqual(result["metrics"]["ece_15_bins"]["null_mean"], .25, delta=.025)

    def test_invalid_inputs(self):
        valid = np.array([[.7, .3], [.2, .8]])
        for labels, p in (([0.0, 1.0], valid), ([0, 2], valid), ([0], valid),
                          ([0, 1], [[.7, .4], [.2, .8]]), ([0, 1], [[np.nan, .3], [.2, .8]])):
            with self.assertRaises(ValueError):
                extended_prediction_metrics(labels, p)
        with self.assertRaises(ValueError):
            extended_prediction_metrics([0, 1], valid, bins=0)
        with self.assertRaises(ValueError):
            paired_extended_comparison([0, 1], valid, valid, repetitions=2, indices=[[0, 1]])
        with self.assertRaises(ValueError):
            paired_extended_comparison([0, 1], valid, valid, repetitions=1, indices=[[0, 1], [0, 1]])


if __name__ == "__main__":
    unittest.main()
