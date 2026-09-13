# Post-hoc calibration of a frozen student

`kd.posthoc` separates probability calibration from the model and training objectives. An immutable `LogitCalibrator` transforms raw logits with temperature scaling (gamma 0), or Focal Temperature Scaling (FTS). `fit_calibrators` selects scalar parameters from validation predictions. The study runner handles checkpoint identity, data provenance, selection, test evaluation and artifact export.

Run from the project root in the existing `kd` environment:

```bash
python -m kd.posthoc_study \
  --run runs/cifar10/kd_seed42 \
  --output runs/cifar10-posthoc/seed42-new
```

Choose a new output directory for each run. The runner uses CPU inference, freezes every model parameter and verifies exact preservation of model tensors and the checkpoint SHA256. No optimizer or training loop is used by the calibration study.

Calibration uses 5,000 validation examples for the existing CIFAR-10 split. The primary temperature and focal-temperature variants minimize NLL; separate secondary variants minimize 15-bin equal-width ECE. The grid is fixed before fitting: T from 0.01 to 5.00 in steps of 0.01 and gamma in `[0, -0.5, -0.25, 0.05, 0.25, 0.37, 0.5, 0.75, 1, 5]`. Gamma 0 includes ordinary temperature scaling in the FTS family. Ties preserve the first candidate, and ordinary temperature scaling wins a tie between families.

All four parameter sets and the choices by validation NLL/ECE are written before the test dataset is loaded. The runner then evaluates 10,000 test examples, saves predictions, and reports accuracy, macro F1, NLL, multiclass Brier score and ECE. Paired bootstrap intervals use 2,000 resamples of test examples; they do not account for calibration-fitting or training-seed variability. The validation split previously selected the checkpoint, and the test set has already been examined in this project, so this follow-up is exploratory.

FTS first computes `q = softmax(logits / T)`, then normalizes:

```text
h(q) = q / ((1-q)^gamma * (1 - gamma*q*log(q)/(1-q)))
p_i = h(q_i) / sum_j h(q_j)
```

The implementation works in log space, including a stable complement probability for saturated logits. Its supported gamma range is [-0.5, 5]. The tests check the independent probability formula, transformation order, class ranking, saturated inputs and a known calibration solution. FTS is based on [Komisarenko and Kull, ECAI 2024](https://arxiv.org/html/2408.11598#S4.SS2). Its paper uses equal-mass ECE; this project keeps its existing equal-width metric for comparable evaluations.

## Apply a saved calibrator

`selected_calibrator.json` is selected by validation NLL. `selected_ece_calibrator.json` is the secondary choice selected by validation ECE. These are small separate artifacts; the student checkpoint remains unchanged.

```python
import json
from pathlib import Path
from kd.checkpoints import fingerprint
from kd.posthoc import LogitCalibrator

artifact = json.loads(Path("runs/cifar10-posthoc/seed42/selected_calibrator.json").read_text())
checkpoint = Path(artifact["checkpoint"])
if fingerprint(checkpoint) != artifact["checkpoint_sha256"]:
    raise ValueError("Calibrator belongs to a different checkpoint")
calibrator = LogitCalibrator(**artifact["parameters"])
# Load the checkpoint using artifact["model_metadata"] and its preprocessing.
# raw_logits: [batch, classes] NumPy array from the frozen model's normal forward.
probabilities = calibrator.predict(raw_logits)
```

Do not apply the training-time KD temperature before this transformation. The saved temperature acts on the model's raw logits. Both calibration methods preserve class predictions; the runner verifies this on every validation and test example. Calibrated inference latency is not measured by this study.

The completed seed-42 results are in [the report](../runs/cifar10-posthoc/seed42/report.md), with a reliability plot, CSV comparison, full grid scores, raw logits, split predictions, metrics, fitted parameter files and provenance hashes.
