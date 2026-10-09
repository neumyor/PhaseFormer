"""Run calibration, freeze, evaluation and audit in a new experiment scratch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main(args):
    args.scratch.mkdir(parents=True, exist_ok=False)
    entry = Path(__file__).with_name('run.py')
    gpus = args.gpus.split(',')
    for stage in ('calibrate', 'evaluate'):
        children = []
        for index, gpu in enumerate(gpus):
            command = [sys.executable, str(entry), '--runs', str(args.runs), '--scratch', str(args.scratch),
                       '--stage', stage, '--shards', str(len(gpus)), '--shard', str(index)]
            if args.smoke:
                command.append('--smoke')
            with (args.scratch/f'{stage}_{index}.log').open('w') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu,
                                              'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                                              'OMP_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2'})
            children.append(child)
        codes = [c.wait() for c in children]
        if any(codes):
            raise RuntimeError(f'{stage} shards failed: {codes}')
        audit_command = [sys.executable, str(entry.with_name('audit.py')), '--scratch', str(args.scratch),
                         '--audit', str(args.audit), '--commit', args.commit,
                         '--stage', 'freeze' if stage == 'calibrate' else 'finalize']
        subprocess.run(audit_command, check=True)
    print(json.dumps({'status': 'complete', 'audit': str(args.audit), 'smoke': args.smoke}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--commit', required=True)
    parser.add_argument('--smoke', action='store_true')
    main(parser.parse_args())
