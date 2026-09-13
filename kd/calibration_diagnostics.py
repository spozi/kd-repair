"""Reproduce extended metrics and conditional-null references from saved logits."""

import argparse
from pathlib import Path

import numpy as np

from .calibration_metrics import conditional_calibration_null
from .checkpoints import fingerprint, write_json
from .evaluation import prediction_metrics
from .posthoc import LogitCalibrator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,required=True,help='NPZ containing labels and logits or probabilities')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--temperature',type=float,default=1.0,help='Fixed supplied temperature; this command does no fitting')
    parser.add_argument('--repetitions',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    with np.load(args.predictions,allow_pickle=False) as archive:
        labels=archive['labels']
        if 'logits' in archive:
            probabilities=LogitCalibrator(args.temperature).predict(archive['logits'])
        else:
            if args.temperature != 1:
                raise ValueError('Temperature transformation requires raw logits')
            probabilities=archive['probabilities']
        classes=archive['classes'].tolist() if 'classes' in archive else [str(i) for i in range(probabilities.shape[1])]
    result={'input':str(args.predictions),'sha256':fingerprint(args.predictions),
            'temperature':args.temperature,'fit_performed':False,
            'metrics':prediction_metrics(labels,probabilities,classes,extended=True),
            'conditional_null':conditional_calibration_null(labels,probabilities,classes,
                                                            seed=args.seed,repetitions=args.repetitions)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,result)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
