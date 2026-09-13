# Neuron-surgery follow-up campaign

This campaign turns the successful factor-100 result into a staged research
program while keeping accelerator work serial and resumable.

## Matrix

1. Reproduce the original factor-100 study from a source snapshot and a fresh
   output directory.
2. Run factor 10 with the existing matched teacher and three student controls.
3. Train fresh factor-100 controls with seeds 141-144 and split seed 7301. Ten
   percent of each class is sealed for final confirmation and is never used for
   fitting or selection.
4. Run validation-only ablations for original-teacher KD preservation,
   same-class companion differences, and causal channel filtering.
5. Train factor-50 controls for 71 epochs, then run repair and matched KD.
6. Compare the already-completed 2, 4, and 8 channel candidates directly from
   their hash-validated artifacts.
7. Exercise the ResNet-34 to ResNet-18 study only if at least two of factor 10,
   factor 50, and sealed factor 100 satisfy the transfer rule. ResNet mapping is
   implemented and unit-tested independently of this compute gate.

The factor-10 and factor-50 official-test results remain exploratory. The sealed
factor-100 result is the confirmatory endpoint because its 5,000-image balanced
holdout is excluded from every fit and selection decision in that fresh protocol.

## Commands

Validate the campaign and all frozen specs:

```bash
python -m kd neuron-surgery-campaign \
  --output runs/cifar10-lt-neuron-campaign \
  --root data \
  --baseline runs/cifar10-lt-multiseed \
  --reference runs/cifar10-lt-neuron-surgery \
  --device mps \
  --dry-run
```

Run one generalized study directly:

```bash
python -m kd neuron-surgery-study \
  --study-config configs/neuron_surgery/f10.json \
  --baseline runs/cifar10-lt-multiseed \
  --output runs/cifar10-lt-neuron-campaign/studies/factor10 \
  --device mps
```

Generate a missing baseline family before its study:

```bash
python -m kd neuron-surgery-baselines \
  --study-config configs/neuron_surgery/f50.json \
  --output runs/cifar10-lt-neuron-campaign/baselines/factor50 \
  --root data \
  --device mps
```

The campaign writes `progress.json` only at stage boundaries. Long-running jobs
write their usual per-epoch histories and checkpoints, so they remain resumable
without requiring an attached progress monitor.
