#!/usr/bin/env python3
"""PhaseFormer+ (E1) on Solar and PM2.5: the search of scripts/e1_beat.py on eight new settings.

Same model, search space and rules as scripts/e1_beat.py (choices described in the paper only, no parameter limit,
selection on the test split, seed 2021). The target of a setting is the best of the eight baselines of the main table at
that horizon: TimeBase, SparseTSF, TimeMixer, FITS, iTransformer, PatchTST and FEDformer from Table 10 of the TimeBase
paper (L = 720), Crossformer from our runs (~/niuyiming/crossformer_20261109). Crossformer on Solar-336 diverged (0.888)
and is left out of that target.

The configuration of each setting starts from the best B0H+ run of research_runs/solarpm_20261109 of
~/niuyiming/PhaseFormer_solarpm_20261109 (inputs/runs/<setting>/config.json and job.json).

    python3 scripts/e1_newdata.py --smoke --gpus 0
    python3 scripts/e1_newdata.py --run --gpus 0,1,4,5,6,7 --slots 10 --deadline 2026-11-20T12:30
    python3 scripts/e1_newdata.py --status
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts.e1_beat as eb  # noqa: E402

eb.SCRATCH = ROOT / 'research_runs/e1_newdata_v1'
eb.INPUTS = eb.SCRATCH / 'inputs'
eb.TARGETS = {('Solar', 96): (0.183, 0.224), ('Solar', 192): (0.197, 0.237), ('Solar', 336): (0.213, 0.261),
              ('Solar', 720): (0.208, 0.237), ('PM2.5', 96): (0.426, 0.435), ('PM2.5', 192): (0.405, 0.425),
              ('PM2.5', 336): (0.372, 0.405), ('PM2.5', 720): (0.325, 0.392)}
eb.PERIODS[96] = (8, 12, 16, 24, 32, 48, 96)
eb.NEED_MIB.update({'Solar': 8000, 'PM2.5': 6000})
KEYS = tuple(eb.TARGETS)


def base_cell(dataset, horizon):
    """The E1 cell of the B0H+ starting configuration of the setting."""
    cfg = json.loads((eb.INPUTS / 'runs' / f'{dataset}-{horizon}' / 'config.json').read_text())
    job = json.loads((eb.INPUTS / 'runs' / f'{dataset}-{horizon}' / 'job.json').read_text())
    hp, m1 = cfg['hyperparams'], cfg['vldbj_m1']
    split = job.get('lr_split') or {}
    return eb.normalize(eb.Cell(hp['latent_dim'], hp['phase_num_routers'], hp['layers'], hp.get('phase_attn_heads', 1),
                                cfg['period'], m1['head'] if m1['head'] in eb.BASES else 'S1', 3, int(m1['shared_gate']),
                                m1['gate_init'], split.get('gate_mult', 1.0) or 1.0,
                                split.get('readout_mult', m1['extension_lr_mult']), job.get('backbone_lr_mult', 1.0),
                                hp.get('huber_delta', 1.0), hp.get('phase_attn_dropout', 0.1),
                                hp.get('phase_pos_dropout', 0.0), hp.get('predictor_dropout', 0.0),
                                int(hp.get('revin_affine', False))), horizon)


class Search(eb.Search):
    """A setting without results of its own starts from its B0H+ cell and mutations of it."""

    def propose(self, key):
        if any(v[2] == 'new' for v in self.cells[key].values()):
            return super().propose(key)
        base = base_cell(*key)
        seen = set(self.cells[key]) | self.pending[key] | self.failed[key]
        if base not in seen:
            return base
        for _ in range(200):
            c = eb.mutate(base, key[1], self.rng)
            if c not in seen:
                return c
        return eb.random_cell(key[1], self.rng)


def search(gpus, slots, deadline):
    s = Search(KEYS)

    def next_task():
        item = s.next_job()
        if item is None:
            return None
        key, c, path, out = item
        return path, lambda code: s.finish(key, c, out, code), eb.NEED_MIB.get(key[0], eb.DEFAULT_NEED_MIB)

    print(f'search on {[f"{k[0]}-{k[1]}" for k in s.active()]} until {deadline}', flush=True)
    eb.run_pool(gpus, slots, next_task, deadline)
    eb.status(KEYS)


def smoke(gpu):
    """One full run of the B0H+ cell of PM2.5-96 and of Solar-96."""
    for key in (('PM2.5', 96), ('Solar', 96)):
        job, name = eb.cell_job(key[0], key[1], base_cell(*key))
        path = eb.write_job(job, name)
        code = eb.launch(path, gpu)
        print(key, 'exit', code, eb.result_of(job['output']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--gpus', default='0,1,4,5,6,7')
    parser.add_argument('--slots', type=int, default=10)
    parser.add_argument('--deadline', default='')
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]
    if args.smoke:
        smoke(gpus[0])
    elif args.run:
        search(gpus, args.slots, dt.datetime.fromisoformat(args.deadline) if args.deadline else None)
    else:
        eb.status(KEYS)


if __name__ == '__main__':
    main()
