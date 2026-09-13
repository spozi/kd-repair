"""Matched CPC–KD study with immutable selection gates before test access."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys

import numpy as np
import torch
import torchvision

from .calibration_metrics import (conditional_calibration_null, extended_prediction_metrics,
                                  paired_seed_comparison)
from .checkpoints import (fingerprint, load_model_checkpoint, metadata, write_json,
                          tensor_state_fingerprint as _state_hash)
from .config import CPCConfig, CalibrationConfig, SupervisedLossConfig, from_dict
from .data import build_data
from .engine import resolve_device, run_experiment, seed_everything
from .evaluation import prediction_metrics, save_prediction_report
from .models import create_model
from .posthoc import LogitCalibrator, collect_logits, fit_calibrators


SEEDS = (42, 43, 44)
TEMPERATURES = (np.arange(1, 501) / 100).tolist()
REPETITIONS = 2000
STATISTICS_SEED = 2026
METRICS = ("accuracy", "aurc", "ece_15_bins", "ece_equal_mass_15_bins", "ece_l2_15_bins",
           "classwise_ece_15_bins", "nll", "brier_score", "macro_f1",
           "selective_accuracy_80", "selective_accuracy_90")
COMPUTE_SOURCES = ("config.py", "data.py", "models.py", "losses.py", "cpc.py", "calibration.py",
                   "adaptive_focal.py", "engine.py", "metrics.py", "evaluation.py", "posthoc.py",
                   "checkpoints.py", "calibration_metrics.py", "cpc_study.py")


def _json(value):
    return json.loads(json.dumps(value))


def _read(path):
    return json.loads(Path(path).read_text())


def _freeze(path, value):
    """Persist an immutable choice; never bless changed artifacts on resume."""
    value = _json(value)
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"Recorded artifact differs: {path}; use a new study directory")
    else:
        write_json(path, value)


def intervention_configs(baseline, output, device):
    """Derive six fresh, interleaved arms from the historical matched recipes."""
    baseline, output = Path(baseline), Path(output).resolve()
    configs = []
    for seed in SEEDS:
        cfg = from_dict(_read(baseline / f"kd_seed{seed}" / "config.json"))
        if (cfg.data.source != "cifar10" or cfg.distillation.method != "kd" or cfg.train.seed != seed
                or cfg.student.checkpoint or cfg.calibration.enabled or cfg.cpc.enabled
                or cfg.supervised_loss.method != "cross_entropy" or cfg.distillation.feature_weight):
            raise ValueError("Expected fresh standard CIFAR-10 KD configurations for seeds 42, 43, 44")
        recipe = (cfg.student.name, cfg.train.epochs, cfg.train.batch_size, cfg.train.learning_rate,
                  cfg.train.momentum, cfg.train.weight_decay, cfg.distillation.temperature,
                  cfg.distillation.weight, cfg.distillation.warmup_epochs,
                  cfg.data.validation_fraction, cfg.data.split_seed, cfg.data.augmentation)
        if recipe != ("cifar_student", 20, 128, .05, .9, .0005, 4., .5, 5, .1, 2026, "basic"):
            raise ValueError("Baseline differs from the predeclared 20-epoch CIFAR-10 CPC study recipe")
        for arm in ("control", "cpc"):
            candidate = replace(cfg, name=f"kd_{arm}_seed{seed}", output_dir=str(output),
                                data=replace(cfg.data, download=False),
                                teacher=replace(cfg.teacher, checkpoint=str(Path(cfg.teacher.checkpoint).resolve())),
                                train=replace(cfg.train, device=device),
                                cpc=CPCConfig(enabled=arm == "cpc", discrimination_weight=.1, exclusion_weight=.1),
                                calibration=CalibrationConfig(), supervised_loss=SupervisedLossConfig())
            candidate.validate()
            configs.append(candidate)
    # The only treatment difference is the CPC switch; all seeds share a teacher/split/recipe.
    for a, b in zip(configs[::2], configs[1::2]):
        ra, rb = a.to_dict(), b.to_dict()
        for field in ("name", "cpc"):
            ra.pop(field)
            rb.pop(field)
        if ra != rb:
            raise ValueError("Control and CPC configurations are not matched")
    if len({fingerprint(c.teacher.checkpoint) for c in configs}) != 1:
        raise ValueError("All arms must use the same frozen teacher")
    return configs


def _data_available(config):
    """Read-only existence check; never construct test datasets in preflight."""
    root = Path(config.data.root)
    if config.data.source == "cifar10":
        required = [root / "cifar-10-batches-py" / name for name in
                    [*(f"data_batch_{i}" for i in range(1, 6)), "test_batch", "batches.meta"]]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"CIFAR-10 must already be downloaded: {missing}")


def _pair_provenance(configs):
    records, provenance, classes = {}, None, None
    for config in configs:
        seed_everything(config.train.seed)
        data = build_data(config.data, config.train, include_test=False)
        model = create_model(config.student.name, config.data.num_classes)
        initial_hash = _state_hash(model.state_dict())
        del model
        # The engine resets augmentation RNG before its first training epoch too.
        seed_everything(config.train.seed)
        digest = hashlib.sha256()
        for index, (images, labels) in enumerate(data.train):
            digest.update(images.numpy().tobytes())
            digest.update(labels.numpy().tobytes())
            if index == 1:
                break
        record = {"initial_student_sha256": initial_hash,
                  "first_two_augmented_batches_sha256": digest.hexdigest(),
                  "data": data.provenance}
        if config.cpc.enabled and record != records[f"kd_control_seed{config.train.seed}"]:
            raise ValueError("Paired initialization or augmented batch streams differ")
        if provenance is not None and (data.provenance != provenance or data.classes != classes):
            raise ValueError("Dataset split or class order differs across arms")
        records[config.name], provenance, classes = record, data.provenance, data.classes
    return {"runs": records, "data": provenance, "classes": classes,
            "stream_verification_scope": "First two augmented batches observed; full streams share identical seeds, loader configurations and engine RNG resets."}


def _protocol(configs):
    return _json({"version": 1, "question": "Does CPC add value beyond standard KD plus temperature scaling?",
                  "configs": [c.to_dict() for c in configs], "seeds": list(SEEDS),
                  "teacher_sha256": fingerprint(configs[0].teacher.checkpoint),
                  "source_sha256": {name: fingerprint(Path(__file__).parent / name) for name in COMPUTE_SOURCES},
                  "selection": "Highest validation accuracy; earliest epoch on ties",
                  "calibration": {"method": "temperature_scaling", "criterion": "validation_nll",
                                  "temperatures": TEMPERATURES, "gammas": [0.0]},
                  "primary_comparison": "KD+CPC+TS minus KD+TS; mean paired test AURC",
                  "numerical_success_rule": "Mean AURC delta < 0 and mean accuracy delta >= -0.005",
                  "accuracy_guardrail": -.005, "inference_device": "cpu",
                  "bootstrap": {"repetitions": REPETITIONS, "seed": STATISTICS_SEED,
                                "pairing": "Same test-image resample across every seed pair"},
                  "calibration_null": {"repetitions": REPETITIONS, "seed": STATISTICS_SEED,
                                       "interpretation": "Conditional perfect-calibration reference; not a hard noise floor"},
                  "metrics": {"ece": "15 equal-width bins", "equal_mass": "15 target groups; confidence ties remain together",
                              "classwise": "Mean marginal one-versus-rest class ECE",
                              "aurc": "Mean cumulative error over retained counts 1..N; expected order within confidence ties",
                              "selective_accuracy": "Expected accuracy at coverage 0.8 and 0.9"},
                  "test_policy": "No test construction before all checkpoint and validation-TS choices are persisted"})


def _complete_run(config, protocol_digest):
    path = Path(config.output_dir) / config.name
    names = ("config.json", "data.json", "history.json", "summary.json", "student.pt", "best.pt", "last.pt")
    manifest_path = path / "completion.json"
    if manifest_path.exists():
        manifest = _read(manifest_path)
        if manifest["protocol_sha256"] != protocol_digest:
            raise ValueError("Completed run belongs to another protocol")
        for name, digest in manifest["files"].items():
            if fingerprint(path / name) != digest:
                raise ValueError(f"Completed run artifact changed: {path / name}")
    if (path / "summary.json").exists():
        if _read(path / "config.json") != _json(config.to_dict()):
            raise ValueError(f"Run configuration differs: {config.name}")
        summary = _read(path / "summary.json")
        history = _read(path / "history.json")
        if len(history) != config.train.epochs or history[-1]["epoch"] != config.train.epochs:
            raise ValueError("Completed summary has incomplete epoch history")
        expected_epoch = max(history, key=lambda r: r["val"]["accuracy"])["epoch"]
        if summary["best_epoch"] != expected_epoch:
            raise ValueError("Summary does not use the declared checkpoint selection rule")
    else:
        if manifest_path.exists():
            raise ValueError("Completed run is missing its summary")
        last = path / "last.pt"
        summary = run_experiment(config, resume=str(last) if last.exists() else None)
    _freeze(manifest_path, {"protocol_sha256": protocol_digest,
                            "files": {name: fingerprint(path / name) for name in names}})
    return summary


def _logits(path, identity, model, loader, classes):
    sidecar = path.with_suffix(".json")
    if path.exists() and sidecar.exists():
        recorded = _read(sidecar)
        if recorded["identity"] != _json(identity) or recorded["sha256"] != fingerprint(path):
            raise ValueError(f"Logit cache identity or contents changed: {path}")
        with np.load(path, allow_pickle=False) as cached:
            if cached["classes"].tolist() != classes:
                raise ValueError("Cached class order differs")
            return cached["labels"], cached["logits"]
    if sidecar.exists():
        raise ValueError(f"Missing signed logit cache: {path}")
    labels, logits = collect_logits(model, loader, torch.device("cpu"))
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, labels=labels, logits=logits, classes=classes)
    os.replace(temporary, path)
    write_json(sidecar, {"identity": identity, "sha256": fingerprint(path)})
    return labels, logits


def _model(config, checkpoint, classes):
    model = create_model(config.student.name, config.data.num_classes)
    state = load_model_checkpoint(model, checkpoint,
                                  metadata(config.student.name, classes, config.data.image_size, config.data.source))
    model.requires_grad_(False).eval()
    return model, state


def _aggregate(rows):
    aggregate = {}
    for condition in ("KD", "KD+TS", "KD+CPC", "KD+CPC+TS"):
        group = [r["test"] for r in rows if r["condition"] == condition]
        aggregate[condition] = {metric: {"mean": statistics.mean(r[metric] for r in group),
                                         "std": statistics.stdev(r[metric] for r in group),
                                         "values": [r[metric] for r in group]} for metric in METRICS}
    return aggregate


def run_cpc_study(baseline="runs/cifar10", output="runs/cifar10-cpc", device="mps", dry_run=False):
    """Train six paired students, freeze six validation temperatures, then evaluate."""
    directory = Path(output).resolve()
    configs = intervention_configs(baseline, directory, device)
    _data_available(configs[0])
    protocol = _protocol(configs)
    if directory.exists() and not (directory / "protocol.json").exists() and any(directory.iterdir()):
        raise FileExistsError(f"Nonempty study directory has no protocol: {directory}")
    if (directory / "protocol.json").exists() and _read(directory / "protocol.json") != protocol:
        raise ValueError("CPC protocol or computational source changed; use a new output directory")
    if dry_run:
        result = {"dry_run": True, "output": str(directory), "run_count": len(configs), "protocol": protocol}
        return result
    resolve_device(device)
    directory.mkdir(parents=True, exist_ok=True)
    _freeze(directory / "protocol.json", protocol)
    protocol_digest = fingerprint(directory / "protocol.json")
    environment = {"python": sys.version, "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                   "numpy": np.__version__, "platform": platform.platform(), "device": device}
    _freeze(directory / "environment.json", environment)
    torch.set_num_threads(configs[0].train.threads)
    print("Verifying paired initialization, validation split and augmented batches...", flush=True)
    pairs = _pair_provenance(configs)
    _freeze(directory / "pairing.json", pairs)
    summaries = {}
    for config in configs:
        summaries[config.name] = _complete_run(config, protocol_digest)
        if summaries[config.name]["data"] != pairs["data"]:
            raise ValueError("Training data provenance differs from the frozen pairing record")
        if summaries[config.name]["initial_student_sha256"] != pairs["runs"][config.name]["initial_student_sha256"]:
            raise ValueError("Actual training initialization differs from the preflight model")
        write_json(directory / "progress.json", {"phase": "training", "completed": list(summaries)})
    selection = {"protocol_sha256": protocol_digest, "pairing_sha256": fingerprint(directory / "pairing.json"),
                 "runs": {c.name: {"checkpoint": str(directory / c.name / "student.pt"),
                                    "checkpoint_sha256": fingerprint(directory / c.name / "student.pt"),
                                    "best_epoch": summaries[c.name]["best_epoch"],
                                    "completion_sha256": fingerprint(directory / c.name / "completion.json")}
                          for c in configs}}
    _freeze(directory / "checkpoint_selection.json", selection)
    calibrators = {}
    validation = build_data(configs[0].data, configs[0].train, include_test=False)
    if validation.provenance != pairs["data"] or validation.classes != pairs["classes"]:
        raise ValueError("Validation split changed")
    for config in configs:
        path, selected = directory / config.name, selection["runs"][config.name]
        model, state = _model(config, selected["checkpoint"], pairs["classes"])
        if state["epoch"] + 1 != selected["best_epoch"]:
            raise ValueError("Exported checkpoint epoch differs from validation selection")
        identity = {"checkpoint_sha256": selected["checkpoint_sha256"], "data": pairs["data"],
                    "metadata": metadata(config.student.name, pairs["classes"], config.data.image_size, config.data.source)}
        labels, logits = _logits(path / "validation_logits.npz", identity, model, validation.val, pairs["classes"])
        artifact_path = path / "calibrator.json"
        fit_identity = {**identity, "validation_logits_sha256": fingerprint(path / "validation_logits.npz"),
                        "fit_criterion": "validation_nll", "protocol_sha256": protocol_digest}
        if artifact_path.exists():
            artifact = _read(artifact_path)
            if artifact["identity"] != fit_identity:
                raise ValueError("Calibrator/checkpoint pairing changed")
            grid = _read(path / "validation_temperature_grid.json")
            winner = min(grid, key=lambda candidate: candidate["nll"])
            if (artifact["parameters"] != {"temperature": winner["temperature"], "gamma": 0.0}
                    or artifact["fit_score"] != winner["nll"]):
                raise ValueError("Calibrator differs from the recorded validation-NLL selection")
        else:
            winners, scores = fit_calibrators(logits, labels, temperatures=TEMPERATURES, gammas=[0.0])
            winner = winners["temperature_nll"]
            artifact = {"identity": fit_identity, **winner, "fit_samples": len(labels),
                        "temperature_at_grid_boundary": winner["parameters"]["temperature"] in (TEMPERATURES[0], TEMPERATURES[-1])}
            # Use a distinct key for the user-facing split criterion and retain fitter's NLL identifier.
            write_json(path / "validation_temperature_grid.json", scores)
            write_json(artifact_path, artifact)
        calibrators[config.name] = artifact
        for variant, parameters in (("raw", LogitCalibrator().to_dict()), ("temperature_nll", artifact["parameters"])):
            probability = LogitCalibrator(**parameters).predict(logits)
            if not np.array_equal(probability.argmax(1), logits.argmax(1)):
                raise ValueError("Temperature scaling changed a validation class prediction")
            quality = save_prediction_report(path / "validation" / variant, labels, probability, pairs["classes"])
            quality.update(extended_prediction_metrics(labels, probability, pairs["classes"], include_curve=True))
            write_json(path / "validation" / variant / "metrics.json", quality)
        print(f"VALIDATION {config.name}: T={artifact['parameters']['temperature']:.2f}, NLL={artifact['fit_score']:.5f}", flush=True)
        del model
    calibration_selection = {"checkpoint_selection_sha256": fingerprint(directory / "checkpoint_selection.json"),
                             "calibrators": {c.name: {"sha256": fingerprint(directory / c.name / "calibrator.json"),
                                                       "parameters": calibrators[c.name]["parameters"]} for c in configs}}
    _freeze(directory / "calibration_selection.json", calibration_selection)
    # A completed computation can be rendered again without inference, fitting or resampling.
    if (directory / "report.json").exists():
        complete = _read(directory / "report_manifest.json")
        if complete != {"protocol_sha256": protocol_digest, "report_sha256": fingerprint(directory / "report.json"),
                        "calibration_selection_sha256": fingerprint(directory / "calibration_selection.json")}:
            raise ValueError("Completed report identity differs")
        from .cpc_reporting import render_report
        return render_report(directory)
    print("All six checkpoints and temperatures frozen. Test access is now unlocked; inference uses CPU.", flush=True)
    test_data = build_data(configs[0].data, configs[0].train, include_test=True)
    expected_provenance = dict(test_data.provenance)
    expected_provenance["split_sizes"] = {k: v for k, v in expected_provenance["split_sizes"].items() if k != "test"}
    if test_data.test is None or test_data.classes != pairs["classes"] or expected_provenance != pairs["data"]:
        raise ValueError("Test data identity differs from the selected training/validation data")
    rows, predictions, common_labels = [], {}, None
    for config in configs:
        path, selected = directory / config.name, selection["runs"][config.name]
        model, _ = _model(config, selected["checkpoint"], pairs["classes"])
        model_hash = _state_hash(model.state_dict())
        identity = {"checkpoint_sha256": selected["checkpoint_sha256"], "data": test_data.provenance,
                    "calibration_selection_sha256": fingerprint(directory / "calibration_selection.json")}
        labels, logits = _logits(path / "test_logits.npz", identity, model, test_data.test, pairs["classes"])
        if len(labels) != len(test_data.test.dataset):
            raise ValueError("Test sample count differs")
        if config.data.source == "cifar10" and (len(labels) != 10000 or not np.all(np.bincount(labels, minlength=10) == 1000)):
            raise ValueError("Official CIFAR-10 test split must contain 1,000 examples in each class")
        if common_labels is not None and not np.array_equal(labels, common_labels):
            raise ValueError("Test-image ordering differs across paired seeds")
        common_labels = labels
        for variant, parameters in (("raw", LogitCalibrator().to_dict()), ("temperature_nll", calibrators[config.name]["parameters"])):
            probability = LogitCalibrator(**parameters).predict(logits)
            if not np.array_equal(probability.argmax(1), logits.argmax(1)):
                raise ValueError("Temperature scaling changed a class prediction")
            condition = ("KD+CPC" if config.cpc.enabled else "KD") + ("+TS" if variant != "raw" else "")
            report_dir = path / "evaluation" / variant
            quality = save_prediction_report(report_dir, labels, probability, pairs["classes"])
            quality.update(extended_prediction_metrics(labels, probability, pairs["classes"], include_curve=True))
            write_json(report_dir / "metrics.json", quality)
            null_path = report_dir / "calibration_null.json"
            null_identity = {"protocol_sha256": protocol_digest, "test_logits_sha256": fingerprint(path / "test_logits.npz"),
                             "parameters": parameters}
            if null_path.exists():
                null = _read(null_path)
                if null["identity"] != null_identity:
                    raise ValueError("Null simulation cache identity differs")
            else:
                print(f"NULL {config.name}/{variant}: {REPETITIONS} conditional label simulations...", flush=True)
                null = {"identity": null_identity, "reference": conditional_calibration_null(
                    labels, probability, pairs["classes"], seed=STATISTICS_SEED, repetitions=REPETITIONS)}
                write_json(null_path, null)
            rows.append({"run": config.name, "seed": config.train.seed, "condition": condition, "variant": variant,
                         "parameters": parameters, "best_epoch": selected["best_epoch"],
                         "training_seconds": summaries[config.name]["training"]["total_epoch_seconds"],
                         "test": quality, "calibration_null": null["reference"]})
            predictions[(config.train.seed, condition)] = probability
            print(f"TEST {config.name}/{variant}: accuracy={quality['accuracy']:.4f} ECE={quality['ece_15_bins']:.4f} AURC={quality['aurc']:.6f}", flush=True)
        if (_state_hash(model.state_dict()) != model_hash
                or fingerprint(selected["checkpoint"]) != selected["checkpoint_sha256"]
                or any(p.grad is not None for p in model.parameters())):
            raise ValueError("Frozen-model inference verification failed")
        del model
        write_json(directory / "progress.json", {"phase": "test_and_statistics", "evaluation_rows": len(rows)})
    comparisons = {}
    for name, a, b in (("primary", "KD+TS", "KD+CPC+TS"), ("raw_secondary", "KD", "KD+CPC")):
        print(f"BOOTSTRAP {name}: {REPETITIONS} shared-image paired resamples across all three seeds...", flush=True)
        comparisons[name] = paired_seed_comparison(common_labels, [predictions[(s, a)] for s in SEEDS],
                                                   [predictions[(s, b)] for s in SEEDS],
                                                   seed=STATISTICS_SEED, repetitions=REPETITIONS)
    aggregate = _aggregate(rows)
    deltas = {m: aggregate["KD+CPC+TS"][m]["mean"] - aggregate["KD+TS"][m]["mean"] for m in METRICS}
    report = {"protocol": protocol, "checkpoint_selection": selection, "calibration_selection": calibration_selection,
              "rows": rows, "aggregate": aggregate, "primary_deltas": deltas, "comparisons": comparisons,
              "primary_numerical_success": deltas["aurc"] < 0 and deltas["accuracy"] >= -.005,
              "verification": {"paired_initialization_and_first_two_batches": True, "all_temperatures_frozen_before_test": True,
                               "test_predictions_unchanged_by_ts": True, "frozen_checkpoint_hashes_unchanged": True,
                               "test_samples": len(common_labels), "training_runs": len(configs), "evaluation_rows": len(rows)},
              "limitations": ["Exploratory: the official CIFAR-10 test set has been examined in earlier experiments.",
                              "Three student seeds and one fixed teacher; CPC weights 0.1/0.1 are untuned project starting values.",
                              "Validation serves both checkpoint selection and temperature fitting; no independent calibration holdout.",
                              "Bootstrap intervals describe test-image uncertainty conditional on fitted models and temperatures, not retraining or calibration-fit uncertainty.",
                              "Conditional-null ranges describe simulated perfectly calibrated labels at fixed probabilities; they are not hard noise floors or confidence intervals on true ECE.",
                              "CPC can conflict with the teacher's non-target class distinctions; this is a CPC–KD adaptation, not a paper benchmark reproduction."]}
    write_json(directory / "report.json", report)
    write_json(directory / "report_manifest.json", {"protocol_sha256": protocol_digest,
                                                   "report_sha256": fingerprint(directory / "report.json"),
                                                   "calibration_selection_sha256": fingerprint(directory / "calibration_selection.json")})
    from .cpc_reporting import render_report
    render_report(directory)
    write_json(directory / "progress.json", {"phase": "complete", "completed": list(summaries), "evaluation_rows": len(rows)})
    return report
