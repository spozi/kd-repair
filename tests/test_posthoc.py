import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from kd.models import ModelOutput
from kd.posthoc import LogitCalibrator, collect_logits, fit_calibrators


class PostHocTests(unittest.TestCase):
    def test_focal_map_matches_independent_closed_form(self):
        q = np.array([[.6,.3,.1], [.2,.25,.55]])
        for gamma in (-.5, -.25, .5, 1., 5.):
            h = q / ((1-q)**gamma * (1-gamma*q*np.log(q)/(1-q)))
            expected = h / h.sum(1, keepdims=True)
            np.testing.assert_allclose(LogitCalibrator(gamma=gamma).predict(np.log(q)), expected, atol=1e-14)

    def test_temperature_identity_and_transform_order(self):
        logits = np.array([[2., 1., -1.], [-2., .5, 1.]])
        for temperature in (.5, 1., 2.):
            q = np.exp(logits/temperature - (logits/temperature).max(1, keepdims=True))
            q /= q.sum(1, keepdims=True)
            np.testing.assert_allclose(LogitCalibrator(temperature).predict(logits), q)
            np.testing.assert_allclose(LogitCalibrator(temperature,.5).predict(logits),
                                       LogitCalibrator(1.,.5).predict(np.log(q)))

    def test_saturated_logits_and_class_order(self):
        logits = np.vstack([np.random.default_rng(9).normal(size=(100,10))*10,
                            [10000., -10000., *([0.]*8)]])
        for temperature in (.01,1.,5.):
            for gamma in (-.5, -.25, 0., .5, 5.):
                lp = LogitCalibrator(temperature,gamma).log_probabilities(logits)
                self.assertTrue(np.isfinite(lp).all())
                np.testing.assert_allclose(np.exp(lp).sum(1), 1, atol=1e-12)
                np.testing.assert_array_equal(lp.argmax(1), logits.argmax(1))

    def test_fit_recovers_known_temperature_and_focal_contains_temperature(self):
        logits = np.tile([2*np.log(4), 0.], (100,1))
        labels = np.array([0]*80+[1]*20)
        fitted, scores = fit_calibrators(logits,labels,temperatures=[1.,2.,3.],gammas=[0.,.5])
        self.assertEqual(fitted['temperature_nll']['parameters']['temperature'], 2.)
        self.assertLess(fitted['temperature_ece']['fit_score'], 1e-12)
        self.assertEqual(len(scores), 6)
        for metric in ('nll','ece'):
            self.assertLessEqual(fitted['focal_temperature_'+metric]['fit_score'],fitted['temperature_'+metric]['fit_score'])
        restored = LogitCalibrator(**fitted['temperature_nll']['parameters'])
        np.testing.assert_allclose(restored.predict(logits)[:,0], .8)

    def test_invalid_settings_and_labels_rejected(self):
        for kwargs in ({'temperature':0}, {'temperature':float('nan')}, {'gamma':-1}, {'gamma':float('inf')}):
            with self.assertRaises(ValueError): LogitCalibrator(**kwargs)
        for labels in ([2], [.5], []):
            with self.assertRaises(ValueError):
                fit_calibrators([[1.,0.]],labels,temperatures=[1.],gammas=[0.])

    def test_logit_collection_preserves_weights_and_mode(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.layer=torch.nn.Linear(2,2)
            def forward(self,x): return ModelOutput(self.layer(x))
        model=Model().train(); before={k:v.clone() for k,v in model.state_dict().items()}
        loader=DataLoader(TensorDataset(torch.ones(4,2),torch.tensor([0,1,0,1])),batch_size=2)
        y,z=collect_logits(model,loader,torch.device('cpu'))
        self.assertTrue(model.training)
        self.assertEqual(z.shape,(4,2))
        self.assertEqual(y.tolist(),[0,1,0,1])
        for k,v in model.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
