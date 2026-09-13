from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from kd.calibration import CalibrationController, MMCELoss
from kd.config import (BenchmarkConfig, CalibrationConfig, DataConfig, DistillationConfig,
                       ExperimentConfig, ModelConfig, TrainConfig, from_dict)
from kd.calibration_study import run_calibration_study
from kd.evaluation import paired_calibration_comparison
from kd.engine import Trainer, run_experiment
from kd.losses import DistillationObjective
from kd.models import ModelOutput


class CalibrationLossTests(unittest.TestCase):
    def test_matches_independent_pairwise_kernel_sum(self):
        probabilities = torch.tensor([[.8, .2], [.3, .7], [.6, .4]], dtype=torch.float64)
        labels = torch.tensor([0, 0, 0])
        residuals, confidence = [.8-1, .7, .6-1], [.8, .7, .6]
        squared = sum(residuals[i]*residuals[j]*math.exp(-abs(confidence[i]-confidence[j])/.4)
                      for i in range(3) for j in range(3))/9
        expected = math.sqrt(squared+1e-12)-1e-6
        self.assertAlmostEqual(MMCELoss()(probabilities.log(), labels).item(), expected, places=10)

    def test_gradient_check_and_extreme_logits(self):
        logits = torch.tensor([[1.1, .2], [.3, 1.7], [2.1, -.4]], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 0, 1])
        self.assertTrue(torch.autograd.gradcheck(lambda x: MMCELoss()(x, labels), (logits,)))
        for target in (torch.tensor([0]), torch.tensor([1])):
            extreme = torch.tensor([[10000., -10000.]], requires_grad=True)
            loss = MMCELoss()(extreme, target)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_calibrated_identical_confidence_and_batch_duplication(self):
        probabilities = torch.tensor([[.8, .2]]*5, dtype=torch.float64)
        labels = torch.tensor([0, 0, 0, 0, 1])
        self.assertAlmostEqual(MMCELoss()(probabilities.log(), labels).item(), 0, places=6)
        logits = torch.tensor([[2., 0.], [0., 1.], [1., 0.]])
        labels = torch.tensor([0, 0, 1])
        torch.testing.assert_close(MMCELoss()(logits, labels), MMCELoss()(logits.repeat(2,1), labels.repeat(2)))

    def test_objective_disabled_is_exact_kd_and_enabled_is_additive(self):
        s = ModelOutput(torch.tensor([[2., .1], [.4, 1.]], requires_grad=True))
        t = ModelOutput(torch.tensor([[3., 0.], [1., 2.]], requires_grad=True))
        labels = torch.tensor([0, 0])
        config = DistillationConfig(method="kd")
        base = DistillationObjective(config)
        calibrated = DistillationObjective(config, calibration_loss=MMCELoss())
        torch.testing.assert_close(base(s,t,labels,0)["total"], calibrated(s,t,labels,0)["total"],rtol=0,atol=0)
        calibrated.calibration_weight = .5
        losses = calibrated(s,t,labels,0)
        torch.testing.assert_close(losses["total"],base(s,t,labels,0)["total"]+.5*MMCELoss()(s.logits,labels))
        losses["total"].backward()
        self.assertIsNone(t.logits.grad)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.bad = {"ece_15_bins":.08,"mean_confidence":.9,"accuracy":.8}
        self.good = {"ece_15_bins":.01,"mean_confidence":.81,"accuracy":.8}

    def test_warmup_patience_causality_and_ramp(self):
        c = CalibrationController(CalibrationConfig(enabled=True))
        for epoch in range(1,8):
            self.assertEqual(c.weight_for_epoch(epoch),0)
            c.observe(epoch,self.bad)
        self.assertEqual(c.state.activated_after_epoch,7)
        self.assertAlmostEqual(c.weight_for_epoch(8),1/3)
        self.assertAlmostEqual(c.weight_for_epoch(9),2/3)
        self.assertEqual(c.weight_for_epoch(10),1)
        c.observe(8,self.good)
        self.assertEqual(c.weight_for_epoch(20),1)

    def test_nonpersistent_or_underconfident_errors_do_not_trigger(self):
        c = CalibrationController(CalibrationConfig(enabled=True,monitor_start_epoch=1))
        for epoch in range(1,8):
            c.observe(epoch,self.bad if epoch%3 else self.good)
        self.assertIsNone(c.state.activated_after_epoch)
        under = dict(self.bad,mean_confidence=.6)
        for epoch in range(8,15):
            c.observe(epoch,under)
        self.assertIsNone(c.state.activated_after_epoch)

    def test_state_round_trip_and_invalid_observations(self):
        c = CalibrationController(CalibrationConfig(enabled=True,monitor_start_epoch=1))
        c.observe(1,self.bad)
        resumed = CalibrationController(c.config)
        resumed.load_state_dict(c.state_dict())
        for epoch in (2,3,4):
            self.assertEqual(c.observe(epoch,self.bad),resumed.observe(epoch,self.bad))
        with self.assertRaises(ValueError):
            c.observe(4,self.bad)
        with self.assertRaises(ValueError):
            c.observe(5,dict(self.bad,ece_15_bins=float("nan")))

    def test_old_configuration_defaults_and_invalid_settings(self):
        self.assertFalse(from_dict({}).calibration.enabled)
        for values in ({"max_weight":0},{"kernel_bandwidth":-1},{"patience":0},{"ece_threshold":2}):
            with self.assertRaises(ValueError):
                from_dict({"calibration":values})


class CalibrationRecoveryTests(unittest.TestCase):
    def test_resume_preserves_activation_and_exact_training(self):
        with tempfile.TemporaryDirectory() as root:
            config = ExperimentConfig(name="full",output_dir=root,
                data=DataConfig(train_samples=16,val_samples=8,test_samples=8),
                train=TrainConfig(epochs=3,batch_size=8,device="cpu"),
                distillation=DistillationConfig(method="supervised"),
                calibration=CalibrationConfig(enabled=True,monitor_start_epoch=1,patience=1,ramp_epochs=2,
                                                ece_threshold=0,overconfidence_threshold=0),
                benchmark=BenchmarkConfig(warmup=0,iterations=1))
            fake_metrics={"accuracy":.25,"mean_confidence":.9,"ece_15_bins":.65,"nll":1.4}
            original=Trainer.train_epoch

            def stop(trainer,loader,epoch):
                if epoch==1:
                    raise InterruptedError()
                return original(trainer,loader,epoch)

            with patch("kd.engine.evaluate",return_value=fake_metrics),redirect_stdout(io.StringIO()):
                run_experiment(config)
                resumed=replace(config,name="resumed")
                with patch.object(Trainer,"train_epoch",stop),self.assertRaises(InterruptedError):
                    run_experiment(resumed)
                run_experiment(resumed,resume=str(Path(root)/"resumed"/"last.pt"))
            full=torch.load(Path(root)/"full"/"last.pt",weights_only=True)
            actual=torch.load(Path(root)/"resumed"/"last.pt",weights_only=True)
            self.assertEqual(full["calibration_controller"],actual["calibration_controller"])
            self.assertEqual(full["calibration_controller"]["activated_after_epoch"],1)
            self.assertEqual([x["calibration_weight"] for x in actual["history"]],[0,.5,1])
            for name, value in full["student"].items():
                torch.testing.assert_close(value,actual["student"][name],rtol=0,atol=0)


class PairedCalibrationTests(unittest.TestCase):
    def test_identical_models_have_zero_deltas_and_intervals(self):
        labels=np.array([0,1,0,1])
        p=np.array([[.8,.2],[.7,.3],[.6,.4],[.3,.7]])
        result=paired_calibration_comparison(labels,p,p,repetitions=40)
        for metric in ("ece","nll","brier"):
            self.assertEqual(result[f"{metric}_delta"],0)
            self.assertEqual(result[f"{metric}_paired_bootstrap_95"],[0,0])

    def test_known_uniform_probability_change(self):
        labels=np.zeros(4,dtype=int)
        baseline=np.array([[.6,.4]]*4)
        candidate=np.array([[.8,.2]]*4)
        r=paired_calibration_comparison(labels,baseline,candidate,repetitions=40)
        for name,expected in (("ece",-.2),("nll",math.log(.6/.8)),("brier",-.24)):
            self.assertAlmostEqual(r[f"{name}_delta"],expected)
            for bound in r[f"{name}_paired_bootstrap_95"]:
                self.assertAlmostEqual(bound,expected)


class InterventionStudyTests(unittest.TestCase):
    def test_complete_study_gates_test_and_resumes_without_retraining(self):
        with tempfile.TemporaryDirectory() as root:
            reference=Path(root)/"reference"
            output=Path(root)/"intervention"

            class MiniCIFAR10(Dataset):
                def __init__(self,directory,train=True,download=False,transform=None):
                    if not train:
                        assert (output/"selection.json").exists()
                    self.targets=np.repeat(np.arange(10),10 if train else 2).tolist()
                    self.classes=[f"class_{i}" for i in range(10)]
                    self.transform=transform
                def __len__(self):
                    return len(self.targets)
                def __getitem__(self,index):
                    return self.transform(Image.new("RGB",(32,32),(index%255,40,70))),self.targets[index]

            with patch("kd.data.datasets.CIFAR10",MiniCIFAR10),redirect_stdout(io.StringIO()):
                teacher_config=ExperimentConfig(name="teacher",output_dir=str(reference),
                    data=DataConfig(source="cifar10",num_classes=10),student=ModelConfig("tiny_medium"),
                    train=TrainConfig(epochs=1,batch_size=32,device="cpu"),
                    distillation=DistillationConfig(method="supervised"),
                    benchmark=BenchmarkConfig(warmup=0,iterations=1))
                teacher=run_experiment(teacher_config)
                for seed in (42,43,44):
                    cfg=replace(teacher_config,name=f"kd_seed{seed}",student=ModelConfig("tiny_small"),
                                teacher=ModelConfig("tiny_medium",teacher["checkpoint"]),
                                train=replace(teacher_config.train,seed=seed),
                                distillation=DistillationConfig(method="kd"))
                    path=reference/cfg.name
                    path.mkdir()
                    (path/"config.json").write_text(json.dumps(cfg.to_dict()))
                report=run_calibration_study(str(reference),str(output),"cpu")
                self.assertEqual(len(report["runs"]),6)
                self.assertFalse(report["all_interventions_activated"])
                for seed in (42,43,44):
                    self.assertEqual(report["runs"][f"kd_mmce_seed{seed}"]["test"]["samples"],20)
                    self.assertEqual(report["runs"][f"kd_mmce_seed{seed}"]["paired_calibration"]["ece_delta"],0)
                with patch("kd.calibration_study.run_experiment",side_effect=AssertionError("Retraining completed run")):
                    repeated=run_calibration_study(str(reference),str(output),"cpu")
                self.assertEqual(report["aggregate"],repeated["aggregate"])
                try:
                    import matplotlib
                except ImportError:
                    return
                from kd.calibration_reporting import render_calibration_report
                self.assertTrue(render_calibration_report(output).exists())
                self.assertTrue((output/"results.csv").exists())
                self.assertTrue((output/"figures"/"intervention.png").exists())
                self.assertTrue((output/"figures"/"reliability.png").exists())
