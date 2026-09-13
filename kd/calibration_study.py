"""Matched fixed-KD and triggered-MMCE experiment; test evaluation runs last."""

from dataclasses import replace
import json
from pathlib import Path
import platform
import statistics
import sys

import numpy as np
import torch
import torchvision

from .checkpoints import fingerprint, load_model_checkpoint, metadata, write_json
from .config import CalibrationConfig, from_dict
from .data import build_data
from .engine import resolve_device, run_experiment
from .evaluation import (collect_predictions, paired_calibration_comparison, paired_comparison,
                         save_prediction_report)
from .models import create_model


def intervention_configs(baseline: Path, output: Path, device: str):
    configs = []
    for seed in (42,43,44):
        config = from_dict(json.loads((baseline/f"kd_seed{seed}"/"config.json").read_text()))
        if config.data.source != "cifar10" or config.distillation.method != "kd" or config.train.seed != seed:
            raise ValueError("Expected the original CIFAR-10 KD configurations for seeds 42,43,44")
        for condition in ("control", "mmce"):
            candidate = replace(config, name=f"kd_{condition}_seed{seed}", output_dir=str(output),
                                train=replace(config.train,device=device),
                                calibration=CalibrationConfig(enabled=condition=="mmce"))
            candidate.validate()
            configs.append(candidate)
    return configs


def run_calibration_study(baseline="runs/cifar10", output="runs/cifar10-calibration", device="mps") -> dict:
    baseline, directory = Path(baseline), Path(output)
    configs = intervention_configs(baseline, directory, device)
    source_files = ("config.py","data.py","models.py","losses.py","calibration.py","engine.py","metrics.py","evaluation.py","calibration_study.py")
    protocol = {"version":1, "question":"Does triggered MMCE reduce KD calibration error while preserving accuracy?",
                "loss":"Existing classical KD + gamma * unweighted empirical MMCE norm; Laplacian bandwidth 0.4",
                "trigger":"After warmup (epoch >=5), ECE >0.03 and confidence-accuracy >0.01 for three consecutive validation epochs",
                "controller":"Activate for the next epoch; gamma ramps to 1.0 over three epochs and stays fixed thereafter",
                "selection":"Same as control: highest validation accuracy, earliest epoch on ties; no ECE-based checkpoint selection",
                "primary_success_rule":"Mean test ECE decreases and mean accuracy drop is at most 0.5 percentage points; report NLL/Brier separately",
                "controls":"Fresh paired controls, same teacher, initialization/data seeds, split, augmentation, optimizer and 20-epoch budget",
                "test_policy":"No test access until all six runs and checkpoint selections are complete",
                "caveat":"Exploratory follow-up: this official test set was already evaluated in the prior study; no new independent test set",
                "teacher_sha256":fingerprint(configs[0].teacher.checkpoint),
                "source_sha256":{name:fingerprint(Path(__file__).parent/name) for name in source_files},
                "configs":[c.to_dict() for c in configs]}
    protocol = json.loads(json.dumps(protocol))
    directory.mkdir(parents=True,exist_ok=True)
    protocol_path = directory/"protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Calibration protocol/source differs; choose a new output directory")
    elif any(directory.iterdir()):
        raise FileExistsError(f"Nonempty study directory: {directory}")
    else:
        write_json(protocol_path,protocol)
    environment = {"python":sys.version,"torch":str(torch.__version__),"torchvision":str(torchvision.__version__),
                   "numpy":np.__version__,"platform":platform.platform(),"device":device}
    write_json(directory/"environment.json",environment)
    summaries = {}
    for config in configs:
        path = directory/config.name
        if (path/"summary.json").exists():
            if json.loads((path/"config.json").read_text()) != json.loads(json.dumps(config.to_dict())):
                raise ValueError(f"Run configuration changed: {config.name}")
            summaries[config.name] = json.loads((path/"summary.json").read_text())
        else:
            last = path/"last.pt"
            summaries[config.name] = run_experiment(config,resume=str(last) if last.exists() else None)
        write_json(directory/"progress.json",{"phase":"training","completed":list(summaries)})
    selection = {"protocol_sha256":fingerprint(protocol_path),
                 "checkpoint_sha256":{name:fingerprint(s["checkpoint"]) for name,s in summaries.items()},
                 "best_epochs":{name:s["best_epoch"] for name,s in summaries.items()}}
    selection_path = directory/"selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Checkpoints changed after test evaluation was unlocked")
    write_json(selection_path,selection)
    print("All six runs complete; selections frozen. Evaluating the official test set.",flush=True)
    first = configs[0]
    data = build_data(first.data,first.train,include_test=True)
    target_device = resolve_device(device)
    results, predictions = {}, {}
    for config in configs:
        model = create_model(config.student.name,config.data.num_classes)
        load_model_checkpoint(model,summaries[config.name]["checkpoint"],
                              metadata(config.student.name,data.classes,config.data.image_size,config.data.source))
        model.to(target_device)
        labels, probabilities = collect_predictions(model,data.test,target_device)
        quality = save_prediction_report(directory/config.name/"test",labels,probabilities,data.classes)
        predictions[config.name] = (labels,probabilities)
        results[config.name] = {**summaries[config.name],"test":quality}
        if config.calibration.enabled:
            baseline_labels, baseline_predictions = predictions[f"kd_control_seed{config.train.seed}"]
            if not np.array_equal(labels,baseline_labels):
                raise ValueError("Test ordering differs between paired runs")
            results[config.name]["paired_accuracy"] = paired_comparison(labels,baseline_predictions,probabilities)
            results[config.name]["paired_calibration"] = paired_calibration_comparison(labels,baseline_predictions,probabilities)
        print(f"TEST {config.name}: accuracy={quality['accuracy']:.4f} ECE={quality['ece_15_bins']:.4f} NLL={quality['nll']:.4f}",flush=True)
        del model
        if target_device.type=="mps":
            torch.mps.empty_cache()
    aggregate = {}
    for condition in ("control","mmce"):
        group = [results[f"kd_{condition}_seed{s}"] for s in (42,43,44)]
        aggregate[condition] = {}
        for metric in ("accuracy","ece_15_bins","nll","brier_score","macro_f1"):
            values = [r["test"][metric] for r in group]
            aggregate[condition][metric] = {"mean":statistics.mean(values),"std":statistics.stdev(values),"values":values}
    deltas = {m:aggregate["mmce"][m]["mean"]-aggregate["control"][m]["mean"] for m in aggregate["control"]}
    report = {"protocol":protocol,"selection":selection,"aggregate":aggregate,"deltas":deltas,"runs":results,
              "primary_success":deltas["ece_15_bins"]<0 and deltas["accuracy"]>=-.005,
              "all_interventions_activated":all(results[f"kd_mmce_seed{s}"]["calibration_controller"]["activated_after_epoch"] is not None for s in (42,43,44)),
              "limitations":["Single fixed teacher and three student seeds; fixed loss weight and trigger, not a tuned MMCE benchmark.",
                             "Validation is a feedback/tuning set as well as the checkpoint-selection set.",
                             "Repeated use of the previously evaluated official test set makes this an exploratory follow-up.",
                             "Bootstrap intervals condition on the fitted models and quantify test-example uncertainty only.",
                             "The one-way trigger tests a persistent-error intervention, not a learned optimal controller or causal diagnosis."]}
    write_json(directory/"report.json",report)
    write_json(directory/"progress.json",{"phase":"complete","completed":list(summaries)})
    return report
