from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kd.config import (BenchmarkConfig, CPCConfig, DataConfig, DistillationConfig,
                       ExperimentConfig, ModelConfig, TrainConfig)
from kd.cpc_study import run_cpc_study
from kd.data import build_data
from kd.engine import run_experiment


def configs_for(root, teacher):
    cfg=ExperimentConfig(output_dir=str(root),
        data=DataConfig(num_classes=3,train_samples=12,val_samples=6,test_samples=6),
        student=ModelConfig('tiny_small'),teacher=ModelConfig('tiny_medium',str(teacher)),
        distillation=DistillationConfig(method='kd'),
        train=TrainConfig(epochs=2,batch_size=6,threads=1,device='cpu'),
        benchmark=BenchmarkConfig(warmup=0,iterations=1))
    return [replace(cfg,name=f'kd_{arm}_seed{seed}',train=replace(cfg.train,seed=seed),
                    cpc=CPCConfig(enabled=arm=='cpc'))
            for seed in (42,43,44) for arm in ('control','cpc')]


class CPCStudyTests(unittest.TestCase):
    def test_dry_run_does_not_construct_models_data_or_write(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); teacher=root/'teacher.pt'; teacher.write_bytes(b'identity-only fixture')
            output=root/'study'; configs=configs_for(output,teacher)
            with patch('kd.cpc_study.intervention_configs',return_value=configs), \
                 patch('kd.cpc_study.build_data',side_effect=AssertionError('dataset constructed')), \
                 patch('kd.cpc_study.create_model',side_effect=AssertionError('model initialized')):
                result=run_cpc_study(output=output,device='cpu',dry_run=True)
            self.assertEqual(result['run_count'],6)
            self.assertFalse(output.exists())

    def test_complete_study_selection_gates_resume_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as root, redirect_stdout(io.StringIO()):
            root=Path(root); output=root/'study'
            base=configs_for(output,root/'teacher.pt')[0]
            teacher_cfg=replace(base,name='teacher',output_dir=str(root),
                                student=ModelConfig('tiny_medium'),distillation=DistillationConfig(method='supervised'),
                                train=replace(base.train,epochs=1))
            teacher=run_experiment(teacher_cfg)
            configs=configs_for(output,teacher['checkpoint'])
            accesses=[]
            def guarded_data(data,train,*,include_test=True):
                if include_test:
                    self.assertTrue((output/'checkpoint_selection.json').exists())
                    self.assertTrue((output/'calibration_selection.json').exists())
                    for config in configs:
                        self.assertTrue((output/config.name/'summary.json').exists())
                        self.assertTrue((output/config.name/'calibrator.json').exists())
                    accesses.append('test')
                bundle=build_data(data,train,include_test=include_test)
                # Keep synthetic validation/test images shared across model seeds.
                bundle.val.dataset.seed=2026
                if bundle.test is not None: bundle.test.dataset.seed=2027
                return bundle
            with patch('kd.cpc_study.intervention_configs',return_value=configs), \
                 patch('kd.cpc_study.build_data',side_effect=guarded_data), \
                 patch('kd.engine.build_data',side_effect=guarded_data), \
                 patch('kd.cpc_study.TEMPERATURES',[1.,1.5,2.]), \
                 patch('kd.cpc_study.REPETITIONS',20):
                result=run_cpc_study(output=output,device='cpu')
                self.assertEqual(len(result['rows']),12)
                self.assertEqual(accesses,['test'])
                self.assertTrue(result['verification']['all_temperatures_frozen_before_test'])
                for seed in (42,43,44):
                    rows=[r for r in result['rows'] if r['seed']==seed]
                    self.assertEqual(rows[0]['test']['accuracy'],rows[1]['test']['accuracy'])
                    self.assertEqual(rows[2]['test']['accuracy'],rows[3]['test']['accuracy'])
                with patch('kd.cpc_study.run_experiment',side_effect=AssertionError('unexpected retraining')), \
                     patch('kd.cpc_study.collect_logits',side_effect=AssertionError('unexpected inference')), \
                     patch('kd.cpc_study.fit_calibrators',side_effect=AssertionError('unexpected fit')):
                    resumed=run_cpc_study(output=output,device='cpu')
                self.assertEqual(resumed['aggregate'],result['aggregate'])
                self.assertEqual(accesses,['test'])
                artifact=output/configs[0].name/'calibrator.json'
                original=artifact.read_text(); changed=json.loads(original)
                changed['parameters']['temperature']=123.
                artifact.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError,'Calibrator differs'):
                    run_cpc_study(output=output,device='cpu')
                artifact.write_text(original)
                manifest=output/'protocol.json'; protocol=json.loads(manifest.read_text())
                protocol['accuracy_guardrail']=-.5
                manifest.write_text(json.dumps(protocol))
                with self.assertRaisesRegex(ValueError,'protocol or computational source changed'):
                    run_cpc_study(output=output,device='cpu')
            self.assertTrue((output/'report.md').exists())
            self.assertTrue((output/'figures'/'comparison.png').exists())


if __name__ == '__main__':
    unittest.main()
