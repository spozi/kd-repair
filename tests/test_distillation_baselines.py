from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

from kd.checkpoints import fingerprint, write_json
from kd.config import (BenchmarkConfig, DataConfig, DistillationConfig, ExperimentConfig,
                       ModelConfig, TrainConfig)
from kd.data import build_data
from kd.distillation_baselines import (RESPONSE_WEIGHTS, run_distillation_baseline,
                                       run_matrix_distillation_baselines)
from kd.engine import run_experiment
from kd.neuron_surgery_study import _model, _quality_report, _save_predictions

JOB, SEEDS, TARGETS = "synthetic-lt", (42, 43), [2]


def _completed_study(root: Path) -> Path:
    """Lay out a trained teacher, matched KD controls and scored predictions as a study."""
    baseline = root / "synthetic" / "lt" / "baselines" / "confirmatory"
    study = root / "synthetic" / "lt" / "studies" / "confirmatory"
    teacher_config = ExperimentConfig(
        name=f"teacher_{JOB}", output_dir=str(baseline),
        data=DataConfig(num_classes=3, train_samples=12, val_samples=6, test_samples=6),
        student=ModelConfig("tiny_medium"), distillation=DistillationConfig(method="supervised"),
        train=TrainConfig(epochs=1, batch_size=6, device="cpu", threads=1),
        benchmark=BenchmarkConfig(warmup=0, iterations=1))
    with redirect_stdout(io.StringIO()):
        run_experiment(teacher_config)
        teacher = str(baseline / teacher_config.name / "student.pt")
        controls = {}
        for seed in SEEDS:
            controls[seed] = replace(
                teacher_config, name=f"kd_{JOB}_seed{seed}", student=ModelConfig("tiny_small"),
                teacher=ModelConfig("tiny_medium", teacher), distillation=DistillationConfig(method="kd"),
                train=replace(teacher_config.train, seed=seed))
            run_experiment(controls[seed])
    evaluation = build_data(teacher_config.data, replace(teacher_config.train, workers=0),
                            include_test=True, diagnostic=True)
    for seed, config in controls.items():
        checkpoint = baseline / config.name / "student.pt"
        model, _ = _model(config, checkpoint, evaluation.classes)
        # The repaired-teacher student is stood in by the control: it is only a reference.
        for condition in ("baseline", "candidate"):
            directory = study / "student_evaluation" / f"seed{seed}" / condition
            identity = {"checkpoint_sha256": fingerprint(checkpoint), "seed": seed,
                        "condition": condition, "evaluation_split": "test",
                        "data": evaluation.provenance}
            labels, probabilities = _save_predictions(directory / "predictions.npz", identity, model,
                                                      evaluation.test, evaluation.classes)
            _quality_report(directory, labels, probabilities, evaluation.classes, TARGETS)
    write_json(study / "comparison.json", {
        "status": "complete", "target_classes": TARGETS,
        "study": {"name": JOB, "teacher_run": teacher_config.name,
                  "student_run_template": f"kd_{JOB}_seed{{seed}}", "student_seeds": list(SEEDS),
                  "evaluation_split": "test"}})
    write_json(root / "matrix_summary.json", {"jobs": [
        {"id": JOB, "dataset": "synthetic", "profile": "lt", "status": "complete"},
        {"id": "unrepaired", "dataset": "synthetic", "profile": "balanced",
         "status": "no_repair_selected"}]})
    return study


class DistillationBaselineTests(unittest.TestCase):
    def test_baseline_changes_only_the_loss_and_is_scored_against_both_references(self):
        with tempfile.TemporaryDirectory() as directory:
            study = _completed_study(Path(directory))
            with redirect_stdout(io.StringIO()):
                report = run_distillation_baseline(study, "rld")
            self.assertEqual(report["status"], "complete")
            self.assertEqual([row["seed"] for row in report["seeds"]], list(SEEDS))
            output = study.parent.parent / "baselines" / "distillation" / "rld"
            for seed in SEEDS:
                # Both configurations as serialized on disk, so tuples and lists compare alike.
                control = json.loads((study.parent.parent / "baselines" / "confirmatory"
                                      / f"kd_{JOB}_seed{seed}" / "config.json").read_text())
                trained = json.loads((output / f"rld_{JOB}_seed{seed}" / "config.json").read_text())
                changed = {(section, key) for section, values in control.items()
                           if isinstance(values, dict)
                           for key in values if values[key] != trained[section][key]}
                changed |= {(key,) for key, value in control.items()
                            if not isinstance(value, dict) and value != trained[key]}
                loss = {("name",), ("output_dir",), ("distillation", "method"), ("distillation", "weight")}
                # The teacher path may be re-anchored; its content is checked just below.
                self.assertTrue(loss <= changed <= loss | {("teacher", "checkpoint")}, changed)
                self.assertEqual(trained["distillation"]["weight"], RESPONSE_WEIGHTS["rld"])
                self.assertEqual(fingerprint(trained["teacher"]["checkpoint"]),
                                 fingerprint(control["teacher"]["checkpoint"]))
            for row in report["seeds"]:
                for name, condition in (("kd_original_teacher", "baseline"),
                                        ("kd_repaired_teacher", "candidate")):
                    reference = json.loads((study / "student_evaluation" / f"seed{row['seed']}"
                                            / condition / "metrics.json").read_text())
                    self.assertAlmostEqual(row["versus"][name]["accuracy_delta"],
                                           row["metrics"]["accuracy"] - reference["accuracy"], places=9)
            self.assertIn("target_recall_delta", report["aggregate"]["versus"]["kd_repaired_teacher"])
            # A second call verifies the frozen report and returns it without retraining.
            student = output / f"rld_{JOB}_seed{SEEDS[0]}" / "student.pt"
            before = student.stat().st_mtime_ns
            again = run_distillation_baseline(study, "rld")
            self.assertEqual(again["protocol_sha256"], report["protocol_sha256"])
            self.assertEqual(again["aggregate"]["accuracy"], report["aggregate"]["accuracy"])
            self.assertEqual(student.stat().st_mtime_ns, before)

    def test_loca_inherits_the_controls_weight_and_matrix_skips_unrepaired_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            study = _completed_study(Path(directory))
            plan = run_matrix_distillation_baselines(directory, methods=("loca", "dkd"), dry_run=True)
            self.assertEqual(list(plan), [JOB])
            self.assertEqual(plan[JOB]["loca"]["response_weight"], DistillationConfig().weight)
            self.assertEqual(plan[JOB]["dkd"]["response_weight"], 1.0)
            self.assertEqual(plan[JOB]["loca"]["runs"], [f"loca_{JOB}_seed{seed}" for seed in SEEDS])
            with self.assertRaisesRegex(ValueError, "unrepaired"):
                run_matrix_distillation_baselines(directory, jobs=["unrepaired"], dry_run=True)
            with self.assertRaisesRegex(ValueError, "one of"):
                run_distillation_baseline(study, "swap", dry_run=True)

    def test_a_control_distilled_from_another_teacher_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            study = _completed_study(Path(directory))
            summary_path = (study.parent.parent / "baselines" / "confirmatory"
                            / f"kd_{JOB}_seed{SEEDS[0]}" / "summary.json")
            summary = json.loads(summary_path.read_text())
            summary["teacher"]["checkpoint_sha256"] = "0" * 64
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "was not distilled from"):
                run_distillation_baseline(study, "dkd", dry_run=True)


if __name__ == "__main__":
    unittest.main()
