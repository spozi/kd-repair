"""Validation-only calibration of an existing checkpoint, followed by test evaluation."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoints import fingerprint, load_model_checkpoint, metadata, write_json
from .config import from_dict
from .data import build_data
from .evaluation import paired_calibration_comparison, save_prediction_report
from .models import create_model
from .posthoc import LogitCalibrator, collect_logits, fit_calibrators


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def make_report(output, report, labels, probabilities):
    results = report['results']
    lines = ['# Frozen KD: post-hoc calibration', '',
             'Model weights and checkpoint unchanged. Calibration parameters fitted only on validation logits, then frozen before test loading.', '',
             f"Primary selection by validation NLL: **{report['selection']['primary']}**. "
             f"Secondary selection by validation ECE: **{report['selection']['secondary']}**.", '',
             '| Method | T | Gamma | Val ECE (%) | Test accuracy (%) | Test ECE (%) | Test NLL | Test Brier |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    rows = []
    for name, r in results.items():
        q, p = r['test'], r['parameters']
        lines.append(f"| {name} | {p['temperature']:.2f} | {p['gamma']:g} | {100*r['validation']['ece_15_bins']:.3f} | "
                     f"{100*q['accuracy']:.2f} | {100*q['ece_15_bins']:.3f} | {q['nll']:.5f} | {q['brier_score']:.5f} |")
        rows.append({'method': name, **p, **{k:q[k] for k in ('accuracy','ece_15_bins','nll','brier_score')}})
    lines += ['', '## Paired changes from uncalibrated KD', '',
              '2,000 paired test-example bootstrap resamples, 95% percentile intervals. Negative changes are better.', '',
              '| Method | ECE change (percentage points), 95% CI | NLL change, 95% CI | Brier change, 95% CI |',
              '|---|---|---|---|']
    for name, r in results.items():
        if name == 'uncalibrated':
            continue
        cells = []
        for metric, scale in (('ece',100),('nll',1),('brier',1)):
            c = r['paired_vs_uncalibrated']
            lo, hi = c[metric+'_paired_bootstrap_95']
            cells.append(f"{scale*c[metric+'_delta']:+.4f} [{scale*lo:+.4f}, {scale*hi:+.4f}]")
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines += ['', '![Test reliability and calibration error](comparison.png)', '',
              '## Protocol and limitations', '',
              '- Fixed T grid 0.01 to 5.00, step 0.01; focal gamma grid [0, -0.5, -0.25, 0.05, 0.25, 0.37, 0.5, 0.75, 1, 5]. Gamma 0 nests ordinary temperature scaling.',
              '- Primary variants minimize validation NLL; secondary variants minimize validation ECE. Each family is fitted separately for each criterion. No test-driven parameter selection.',
              '- ECE uses 15 equal-width bins, matching previous project evaluations. The FTS paper uses equal-mass ECE; this is not a direct reproduction of its ECE protocol.',
              *['- '+s for s in report['limitations']], '',
              'Focal Temperature Scaling applies softmax(logits/T) followed by the normalized focal calibration map. '
              'Method: [Komisarenko and Kull, ECAI 2024](https://arxiv.org/html/2408.11598#S4.SS2).', '',
              '## Reproduce and reuse', '',
              'Run from the project root in the kd environment, choosing a new output directory:', '',
              '```bash', 'python -m kd.posthoc_study --run runs/cifar10/kd_seed42 --output runs/cifar10-posthoc/seed42-repeat', '```', '',
              'Use selected_calibrator.json for the primary validation-NLL choice, or selected_ece_calibrator.json for the secondary validation-ECE choice. '
              'Both include checkpoint SHA256, class order and preprocessing. Apply LogitCalibrator(**artifact["parameters"]).predict(raw_logits); '
              'do not apply the KD training temperature first. Full fitting scores, split logits, predictions and metrics are saved alongside this report.', '']
    (output/'report.md').write_text('\n'.join(lines))
    write_csv(output/'results.csv', rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for name, p in probabilities.items():
        confidence, correct = p.max(1), p.argmax(1) == labels
        bins = np.minimum((confidence*15).astype(int),14)
        points = [(confidence[bins==b].mean(), correct[bins==b].mean()) for b in range(15) if (bins==b).sum() >= 20]
        if points:
            axes[0].plot(*zip(*points), 'o-', markersize=3, label=name)
    axes[0].plot([0,1],[0,1], '--', color='gray')
    axes[0].set(xlabel='Mean confidence', ylabel='Observed accuracy', title='Test reliability (bins with ≥20 examples)')
    axes[0].legend(fontsize=7)
    axes[1].barh(list(results), [100*r['test']['ece_15_bins'] for r in results.values()])
    axes[1].set(xlabel='ECE (%) — lower is better', title='All methods retain identical class predictions')
    fig.savefig(output/'comparison.png', dpi=180)
    plt.close(fig)


def run_study(run, output):
    run, output = Path(run), Path(output)
    cfg = from_dict(json.loads((run/'config.json').read_text()))
    checkpoint = run/'student.pt'
    digest = fingerprint(checkpoint)
    # Refuse to replace previous results or calibration artifacts.
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(cfg.train.threads)
    temperatures = (np.arange(1,501)/100).tolist()
    gammas = [0., -.5, -.25, .05, .25, .37, .5, .75, 1., 5.]
    protocol = {'created_utc':datetime.now(timezone.utc).isoformat(), 'checkpoint':str(checkpoint.resolve()),
                'checkpoint_sha256':digest, 'config_sha256':fingerprint(run/'config.json'),
                'device':'cpu', 'temperatures':temperatures, 'gammas':gammas,
                'fit_split':'validation', 'primary_criterion':'nll', 'secondary_criterion':'ece',
                'ece_bins':15, 'ece_binning':'equal_width', 'bootstrap_seed':2026, 'bootstrap_repetitions':2000,
                'source_sha256':{name:fingerprint(Path(__file__).parent/name) for name in ('posthoc.py','posthoc_study.py','evaluation.py','data.py','models.py')},
                'versions':{'torch':str(torch.__version__), 'numpy':np.__version__}}
    write_json(output/'protocol.json', protocol)
    data = build_data(cfg.data,cfg.train,include_test=False)
    if (run/'summary.json').exists():
        prior = json.loads((run/'summary.json').read_text())
        if data.provenance != prior['data']:
            raise ValueError('Dataset provenance differs from the checkpoint evaluation')
    expected = metadata(cfg.student.name,data.classes,cfg.data.image_size,cfg.data.source)
    model = create_model(cfg.student.name,cfg.data.num_classes)
    state = load_model_checkpoint(model,checkpoint,expected)
    model.requires_grad_(False).eval()
    original = {k:v.clone() for k,v in model.state_dict().items()}
    print('Collecting validation logits from frozen checkpoint...',flush=True)
    val_labels, val_logits = collect_logits(model,data.val,torch.device('cpu'))
    np.savez_compressed(output/'validation_logits.npz',labels=val_labels,logits=val_logits,classes=data.classes)
    print(f'Fitting {len(temperatures)*len(gammas)} validation candidates on {len(val_labels)} examples...',flush=True)
    winners, scores = fit_calibrators(val_logits,val_labels,temperatures=temperatures,gammas=gammas)
    write_csv(output/'validation_grid.csv',scores)
    primary = min(('temperature_nll','focal_temperature_nll'),key=lambda k:winners[k]['fit_score'])
    secondary = min(('temperature_ece','focal_temperature_ece'),key=lambda k:winners[k]['fit_score'])
    selection = {'selected_utc':datetime.now(timezone.utc).isoformat(), 'primary':primary, 'secondary':secondary,
                 'winners':winners, 'checkpoint_epoch':state['epoch']+1, 'validation_provenance':data.provenance}
    write_json(output/'selection.json',selection)
    for filename, name in (('selected_calibrator.json',primary),('selected_ece_calibrator.json',secondary)):
        write_json(output/filename,{'format_version':1,'method':name,**winners[name],
                                   'checkpoint':protocol['checkpoint'],'checkpoint_sha256':digest,
                                   'model_metadata':expected, 'fit_samples':len(val_labels)})
    # No test dataset or labels are loaded until every choice has been persisted.
    print('Calibration parameters frozen:',json.dumps(winners),flush=True)
    data_test = build_data(cfg.data,cfg.train,include_test=True)
    if data_test.test is None:
        raise ValueError('A separate test split is required')
    test_labels, test_logits = collect_logits(model,data_test.test,torch.device('cpu'))
    np.savez_compressed(output/'test_logits.npz',labels=test_labels,logits=test_logits,classes=data.classes)
    variants = {'uncalibrated':{'parameters':LogitCalibrator().to_dict()},**winners}
    results, probabilities = {}, {}
    for name, specification in variants.items():
        calibrator = LogitCalibrator(**specification['parameters'])
        p = calibrator.predict(test_logits)
        np.testing.assert_array_equal(p.argmax(1),test_logits.argmax(1))
        v = calibrator.predict(val_logits)
        np.testing.assert_array_equal(v.argmax(1),val_logits.argmax(1))
        result = {'parameters':calibrator.to_dict(),
                  'validation':save_prediction_report(output/name/'validation',val_labels,v,data.classes),
                  'test':save_prediction_report(output/name/'test',test_labels,p,data.classes)}
        if name != 'uncalibrated':
            result['paired_vs_uncalibrated'] = paired_calibration_comparison(test_labels,probabilities['uncalibrated'],p)
        results[name], probabilities[name] = result, p
        print(name,{k:result['test'][k] for k in ('accuracy','ece_15_bins','nll','brier_score')},flush=True)
    unchanged = all(torch.equal(value,model.state_dict()[key]) for key,value in original.items())
    if not unchanged or fingerprint(checkpoint) != digest or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('Frozen checkpoint verification failed')
    report = {'protocol':protocol,'selection':selection,'results':results,
              'focal_vs_temperature_nll':paired_calibration_comparison(test_labels,probabilities['temperature_nll'],probabilities['focal_temperature_nll']),
              'verification':{'checkpoint_sha256_unchanged':True,'model_tensors_unchanged':True,'model_gradients_absent':True,
                              'class_predictions_unchanged_all_variants':True,'validation_samples':len(val_labels),'test_samples':len(test_labels)},
              'limitations':['One checkpoint (seed 42 in this study); intervals measure test-sample uncertainty conditional on fitted calibrators, not training-seed or calibration-fit variability.',
                             'The validation split was also used to select the original checkpoint. Fitting scores are not independent performance estimates.',
                             'The official test set was examined in earlier experiments. This is an exploratory follow-up, not fresh independent confirmation.',
                             'ECE depends on binning. ECE-optimized variants are secondary and may trade off NLL/Brier.',
                             'Finite-grid fitting; parameters are optimal only within the recorded grid. No claim about distribution shift or deployment latency.']}
    write_json(output/'report.json',report)
    make_report(output,report,test_labels,probabilities)
    print(f'Completed: {output}/report.md',flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/cifar10/kd_seed42'))
    parser.add_argument('--output',type=Path,default=Path('runs/cifar10-posthoc/seed42'))
    args = parser.parse_args()
    run_study(args.run,args.output)


if __name__ == '__main__':
    main()
