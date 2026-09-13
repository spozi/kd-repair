from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from kd.checkpoints import fingerprint
from kd.config import (CPCConfig, CalibrationConfig, DataConfig, DistillationConfig,
                       ExperimentConfig, ModelConfig, TrainConfig, BenchmarkConfig,
                       SupervisedLossConfig, from_dict, load_config)
from kd.cpc import PairwiseCalibrationLoss
from kd.engine import Trainer, run_experiment
from kd.losses import DistillationObjective
from kd.models import ModelOutput
from kd.metrics import evaluate
from kd.evaluation import prediction_metrics
from torch.utils.data import DataLoader, TensorDataset


class CPCIntegrationTests(unittest.TestCase):
    def test_extended_evaluation_is_opt_in_and_reuses_predictions(self):
        class FixedModel(torch.nn.Module):
            def forward(self, images):
                return ModelOutput(images)
        logits=torch.tensor([[2.,.1,-1.],[.1,1.,2.],[2.,1.,0.]])
        labels=torch.tensor([0,1,2])
        loader=DataLoader(TensorDataset(logits,labels),batch_size=2)
        model=FixedModel().train()
        normal=evaluate(model,loader,torch.device('cpu'))
        detailed=evaluate(model,loader,torch.device('cpu'),extended=True)
        self.assertTrue(model.training)
        self.assertNotIn('extended',normal)
        self.assertEqual(normal,{k:v for k,v in detailed.items() if k!='extended'})
        quality=prediction_metrics(labels.numpy(),logits.softmax(1).numpy(),['a','b','c'],extended=True)
        self.assertAlmostEqual(quality['extended']['aurc'],detailed['extended']['aurc'])
        self.assertAlmostEqual(quality['ece_15_bins'],quality['extended']['ece_15_bins'])

    def test_config_defaults_examples_and_incompatible_combinations(self):
        self.assertFalse(from_dict({}).cpc.enabled)
        self.assertTrue(load_config('configs/cifar10_kd_cpc.toml').cpc.enabled)
        base = ExperimentConfig(distillation=DistillationConfig(method='kd'),cpc=CPCConfig(enabled=True))
        base.validate(require_teacher=False)
        for cfg in (replace(base,calibration=CalibrationConfig(enabled=True)),
                    replace(base,supervised_loss=SupervisedLossConfig(method='adafocal')),
                    replace(base,distillation=DistillationConfig(method='dkd')),
                    replace(base,distillation=DistillationConfig(method='kd',feature_weight=1))):
            with self.assertRaises(ValueError): cfg.validate(require_teacher=False)
        for c in (CPCConfig(enabled=1),CPCConfig(discrimination_weight=-.1),CPCConfig(exclusion_weight=float('nan')),
                  CPCConfig(warmup_epochs=-1),CPCConfig(warmup_epochs=1.5),CPCConfig(warmup_epochs=True)):
            with self.assertRaises(ValueError): c.validate()
        self.assertEqual(CPCConfig().warmup_epochs,0)

    def test_warmup_ramps_total_but_not_the_logged_weighted_penalty(self):
        student = ModelOutput(torch.tensor([[3.,.4,-1.],[.1,1.,2.]],requires_grad=True))
        teacher = ModelOutput(torch.tensor([[1.,2.,0.],[.2,.5,1.]],requires_grad=True))
        labels = torch.tensor([0,1])
        cfg = DistillationConfig(method='kd',temperature=4,weight=.5,warmup_epochs=5)
        baseline = DistillationObjective(cfg)
        penalty = PairwiseCalibrationLoss(.2,.3,warmup_epochs=4)
        objective = DistillationObjective(cfg,cpc_loss=penalty)
        for epoch,ramp in ((0,.25),(1,.5),(3,1.0),(19,1.0)):
            base = baseline(student,teacher,labels,epoch)['total']
            expected = penalty(student.logits,labels)
            result = objective(student,teacher,labels,epoch)
            torch.testing.assert_close(result['cpc_weighted'],expected)
            torch.testing.assert_close(result['total'],base+ramp*expected)

    def test_additive_components_once_raw_logits_no_teacher_gradients(self):
        student = ModelOutput(torch.tensor([[3.,.4,-1.],[.1,1.,2.]],requires_grad=True))
        teacher = ModelOutput(torch.tensor([[1.,2.,0.],[.2,.5,1.]],requires_grad=True))
        labels = torch.tensor([0,1])
        cfg = DistillationConfig(method='kd',temperature=4,weight=.5,warmup_epochs=5)
        baseline = DistillationObjective(cfg)
        zero = DistillationObjective(cfg,cpc_loss=PairwiseCalibrationLoss(0,0))
        penalty = PairwiseCalibrationLoss(.2,.3)
        objective = DistillationObjective(cfg,cpc_loss=penalty)
        for epoch in (0,4,19):
            base = baseline(student,teacher,labels,epoch)['total']
            torch.testing.assert_close(zero(student,teacher,labels,epoch)['total'],base,rtol=0,atol=0)
            expected = penalty(student.logits,labels)
            with patch.object(penalty,'components',wraps=penalty.components) as called:
                result = objective(student,teacher,labels,epoch)
                self.assertEqual(called.call_count,1)
            torch.testing.assert_close(result['total'],base+expected)
            torch.testing.assert_close(result['cpc_weighted'],expected)
        result['total'].backward()
        self.assertIsNone(teacher.logits.grad)
        self.assertGreater(student.logits.grad.abs().sum(),0)

    def test_resume_exact_logging_and_student_only_export(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            teacher_cfg = ExperimentConfig(name='teacher',output_dir=root,
                data=DataConfig(num_classes=3,train_samples=12,val_samples=6,test_samples=6),
                student=ModelConfig('tiny_medium'),distillation=DistillationConfig(method='supervised'),
                train=TrainConfig(epochs=1,batch_size=6,device='cpu',threads=1),
                benchmark=BenchmarkConfig(warmup=0,iterations=1))
            teacher = run_experiment(teacher_cfg)
            before = fingerprint(teacher['checkpoint'])
            cfg=replace(teacher_cfg,name='full',student=ModelConfig('tiny_small'),
                        teacher=ModelConfig('tiny_medium',teacher['checkpoint']),
                        distillation=DistillationConfig(method='kd'),
                        train=replace(teacher_cfg.train,epochs=3),cpc=CPCConfig(enabled=True))
            full=run_experiment(cfg)
            original=Trainer.train_epoch
            def interrupted(trainer,loader,epoch):
                if epoch == 1: raise InterruptedError('test interruption')
                return original(trainer,loader,epoch)
            resumed=replace(cfg,name='resumed')
            with patch.object(Trainer,'train_epoch',interrupted),self.assertRaises(InterruptedError):
                run_experiment(resumed)
            checkpoint=str(Path(root)/'resumed'/'last.pt')
            with self.assertRaisesRegex(ValueError,'configuration differs'):
                run_experiment(replace(resumed,cpc=CPCConfig(enabled=True,exclusion_weight=.2)),resume=checkpoint)
            summary=run_experiment(resumed,resume=checkpoint)
            a=torch.load(Path(root)/'full'/'last.pt',weights_only=True)
            b=torch.load(checkpoint,weights_only=True)
            self.assertEqual(full['initial_student_sha256'],summary['initial_student_sha256'])
            self.assertEqual(before,fingerprint(teacher['checkpoint']))
            for name,value in a['student'].items():
                torch.testing.assert_close(value,b['student'][name],rtol=0,atol=0)
            for row in b['history']:
                self.assertGreater(row['train']['cpc_weighted'],0)
                self.assertAlmostEqual(row['train']['cpc_weighted'],.1*(row['train']['cpc_discrimination']+row['train']['cpc_exclusion']),places=6)
                self.assertEqual(row['calibration_weight'],0)
            exported=torch.load(summary['checkpoint'],weights_only=True)
            self.assertNotIn('objective',exported)
            self.assertEqual(summary['training']['extra_trainable_parameters'],0)


if __name__ == '__main__':
    unittest.main()
