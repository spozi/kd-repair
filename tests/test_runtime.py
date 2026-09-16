import unittest
from unittest.mock import patch

import torch

from kd.config import DataConfig, ExperimentConfig, TrainConfig
from kd.data import build_data
from kd.runtime import (accelerator_workers, cuda_preflight, cuda_requested,
                        loader_performance_kwargs, make_grad_scaler, resolve_device,
                        resolve_precision, valid_device_spec)


class RuntimePolicyTests(unittest.TestCase):
    def test_device_syntax_accepts_indexed_cuda(self):
        for value in ("auto", "cpu", "mps", "cuda", "cuda:0", "cuda:17"):
            self.assertTrue(valid_device_spec(value))
        for value in ("gpu", "cuda:-1", "cuda:x", "cuda:0:1"):
            self.assertFalse(valid_device_spec(value))

    def test_explicit_cuda_index_is_validated_without_allocating(self):
        with (patch("kd.runtime.torch.cuda.is_available", return_value=True),
              patch("kd.runtime.torch.cuda.device_count", return_value=2)):
            self.assertEqual(resolve_device("cuda:1"), torch.device("cuda:1"))
            with self.assertRaisesRegex(ValueError, "found 2 device"):
                resolve_device("cuda:2")

    def test_auto_precision_prefers_bfloat16_when_supported(self):
        device = torch.device("cuda:0")
        with patch("kd.runtime.torch.cuda.get_device_capability", return_value=(8, 0)):
            self.assertEqual(resolve_precision("auto", device), "bfloat16")
        with patch("kd.runtime.torch.cuda.get_device_capability", return_value=(7, 5)):
            self.assertEqual(resolve_precision("auto", device), "float16")
        self.assertEqual(resolve_precision("auto", torch.device("cpu")), "float32")

    def test_loader_policy_pins_cuda_and_guards_worker_only_options(self):
        self.assertTrue(cuda_requested("cuda:3"))
        zero = loader_performance_kwargs("cuda:0", 0, persistent_workers=True)
        self.assertEqual(zero, {"pin_memory": True, "persistent_workers": False})
        workers = loader_performance_kwargs(
            "cuda", 4, persistent_workers=True, prefetch_factor=3)
        self.assertEqual(workers, {
            "pin_memory": True, "persistent_workers": True, "prefetch_factor": 3})
        self.assertEqual(accelerator_workers("cuda:2", 0), 4)
        self.assertEqual(accelerator_workers("cuda:2", 8), 8)
        self.assertEqual(accelerator_workers("cpu", 0), 0)

    def test_build_data_applies_cuda_loader_policy(self):
        bundle = build_data(
            DataConfig(), TrainConfig(device="cuda:0", batch_size=8), include_test=False)
        self.assertTrue(bundle.train.pin_memory)
        self.assertFalse(bundle.train.persistent_workers)

    def test_cpu_scaler_is_disabled(self):
        scaler = make_grad_scaler(torch.device("cpu"), "auto")
        self.assertFalse(scaler.is_enabled())

    def test_cuda_preflight_rejects_non_cuda_device(self):
        with self.assertRaisesRegex(ValueError, "requires a cuda"):
            cuda_preflight("cpu", batch_size=1, iterations=1)

    def test_runtime_configuration_is_validated(self):
        ExperimentConfig(train=TrainConfig(device="cuda:7")).validate(require_teacher=False)
        with self.assertRaisesRegex(ValueError, "precision"):
            ExperimentConfig(train=TrainConfig(precision="int8")).validate(require_teacher=False)
        with self.assertRaisesRegex(ValueError, "cuda_mode"):
            ExperimentConfig(train=TrainConfig(cuda_mode="turbo")).validate(require_teacher=False)
        with self.assertRaisesRegex(ValueError, "channels_last"):
            ExperimentConfig(train=TrainConfig(channels_last=1)).validate(require_teacher=False)


if __name__ == "__main__":
    unittest.main()
