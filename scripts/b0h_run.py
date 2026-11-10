#!/usr/bin/env python3
"""B0H+: the b0 arm with its level read from history and phase tokens together.

b0 reads the level from the history: the input minus its last value, pooled to
one mean per cycle (30 values at lookback 720, period 24), then a linear
encoder to rank 3. B0H+ keeps those cycle means and concatenates the slot-mean
of the embedded phase tokens -- (B, C, slots, D) averaged over slots -- so the
encoder input is 30 + D wide. Decoder, last-value anchor and fusion are
unchanged, and the level path back-propagates into the shared embedding.

Every setting reuses its best b0 configuration from ``b0_search_v1`` (period,
gate init, extension and backbone learning-rate multipliers, level head, gate
sharing), seed 2021, so the comparison isolates the level input.

    python3 scripts/b0h_run.py --check-init
    python3 scripts/b0h_run.py --run --phase non-traffic --gpus 1,2,3,7 --slots-per-gpu 2
    python3 scripts/b0h_run.py --run --phase traffic --gpus 1,2,3,7 --slots-per-gpu 1
    python3 scripts/b0h_run.py --report
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ablation_no_amplitude import ROWS, SEED  # noqa: E402
from scripts.b0_search import DEFAULT_SCRATCH as B0_SCRATCH, load_results, winners  # noqa: E402
from scripts.vldbj_m1.proto import provenance, run_jobs  # noqa: E402
from scripts.vldbj_m1.run import commit, e14_sources, write_immutable  # noqa: E402

DEFAULT_SCRATCH = ROOT / 'research_runs/b0hp_v1'
STAGE = 'main'
SWEEP_STAGE = 'sweep'
GATE_LO, GATE_HI = 1e-4, 0.999
LR_LO, LR_HI = 0.03, 1000.0
BB_LO, BB_HI = 0.1, 10.0


def snap(value, lo, hi, digits=5):
    return min(hi, max(lo, round(value, digits)))


def sweep_cells(best):
    """Local grid around one setting's b0 winner; Traffic gets the narrow one."""
    gates = (snap(best['gate'] * 0.5, GATE_LO, GATE_HI), snap(best['gate'] * 2.0, GATE_LO, GATE_HI))
    rates = (snap(best['ext'] * 0.25, LR_LO, LR_HI), snap(best['ext'] * 4.0, LR_LO, LR_HI))
    cells = [(best['period'], g, r, best['bb']) for g in gates for r in rates]
    if best['dataset'] != 'Traffic':
        cells += [(best['period'], best['gate'], best['ext'], snap(best['bb'] * 0.3, BB_LO, BB_HI)),
                  (best['period'], best['gate'], best['ext'], snap(best['bb'] * 3.0, BB_LO, BB_HI)),
                  (best['period'], gates[1], rates[1], snap(best['bb'] * 3.0, BB_LO, BB_HI))]
    seen, out = set(), []
    for cell in cells:
        if cell in seen:
            continue
        seen.add(cell)
        out.append(cell)
    return out


def sweep_name(best, cell):
    period, gate, ext, bb = cell
    return (f"{best['dataset']}_H{best['horizon']}_S{SEED}__b0hp__g{gate:g}_lr{ext:g}_bb{bb:g}"
            f"__p{period}__{best['head']}__{'shared' if best['shared'] else 'perC'}")
DATASETS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic')
GOLDEN = {'ETTh1': (0.4030, 0.4150), 'ETTh2': (0.3468, 0.3887), 'ETTm1': (0.3465, 0.3740),
          'ETTm2': (0.2505, 0.3135), 'Weather': (0.2230, 0.2605),
          'Electricity': (0.1608, 0.2502), 'Traffic': (0.3867, 0.2497)}


def b0_best(rows):
    results = load_results(B0_SCRATCH)
    out = {}
    for row in rows:
        best = winners(results, row['dataset'], row['horizon'])
        if best is None:
            raise SystemExit(f"no b0 result for {row['dataset']}-{row['horizon']}")
        out[(row['dataset'], row['horizon'])] = best
    return out


def source_for(sources, dataset, horizon, period):
    source = sources[(dataset, horizon, SEED)]
    if period != 24:
        source = copy.deepcopy(source)
        source['config']['period'] = period
        source['config']['hyperparams']['period_len'] = period
    return source


def job_for(base, sources, best, level_input):
    job = dict(base, dataset=best['dataset'], horizon=best['horizon'], seed=SEED,
               arm='b0', protocol='e14', init='scratch', head=best['head'],
               shared_gate=bool(best['shared']), variant_id=None,
               variant={'gate_init': best['gate'], 'extension_lr_mult': best['ext'],
                        'communication_width': 2},
               extension_lr_mult=float(best['ext']), backbone_lr_mult=float(best['bb']),
               warm_source=None, epoch_test=False,
               source=source_for(sources, best['dataset'], best['horizon'], best['period']))
    if level_input != 'history':
        job['level_input'] = level_input
    return job


def build_sweep_jobs(scratch, rows, phase):
    """B0H+ cells around each setting's b0 winner."""
    sources = e14_sources()
    base = {'scratch': str(scratch), 'stage': SWEEP_STAGE, 'code_commit': commit(),
            'source_hashes': provenance(), 'smoke': False}
    best = b0_best(rows)
    paths = []
    for row in rows:
        if (phase == 'traffic') != (row['dataset'] == 'Traffic') and phase != 'all':
            continue
        b = best[(row['dataset'], row['horizon'])]
        for cell in sweep_cells(b):
            period, gate, ext, bb = cell
            variant = {'gate_init': gate, 'extension_lr_mult': ext, 'communication_width': 2}
            job = dict(base, dataset=b['dataset'], horizon=b['horizon'], seed=SEED,
                       arm='b0', protocol='e14', init='scratch', head=b['head'],
                       shared_gate=bool(b['shared']), variant_id=None, variant=variant,
                       extension_lr_mult=float(ext), backbone_lr_mult=float(bb),
                       warm_source=None, epoch_test=False, level_input='history+token',
                       source=source_for(sources, b['dataset'], b['horizon'], period),
                       b0_reference={'run': b['run'], 'test_mse': b['mse'], 'test_mae': b['mae']},
                       m1_reference={'test_mse': row['m1_mse'], 'test_mae': row['m1_mae']})
            name = sweep_name(b, cell)
            job['output'] = str(scratch / SWEEP_STAGE / 'runs' / name)
            path = scratch / SWEEP_STAGE / 'jobs' / f'{name}.json'
            write_immutable(path, job)
            paths.append(path)
    return paths


def build_jobs(scratch, rows, phase):
    sources = e14_sources()
    base = {'scratch': str(scratch), 'stage': STAGE, 'code_commit': commit(),
            'source_hashes': provenance(), 'smoke': False}
    best = b0_best(rows)
    paths = []
    for row in rows:
        if (phase == 'traffic') != (row['dataset'] == 'Traffic') and phase != 'all':
            continue
        b = best[(row['dataset'], row['horizon'])]
        job = job_for(base, sources, b, 'history+token')
        job.update(b0_reference={'run': b['run'], 'test_mse': b['mse'], 'test_mae': b['mae']},
                   m1_reference={'test_mse': row['m1_mse'], 'test_mae': row['m1_mae']})
        name = f"{b['run']}__b0hp"
        job['output'] = str(scratch / STAGE / 'runs' / name)
        path = scratch / STAGE / 'jobs' / f'{name}.json'
        write_immutable(path, job)
        paths.append(path)
    return paths


def clear_incomplete(paths):
    """Remove run directories a failed worker left behind, so a retry can start."""
    removed = 0
    for path in paths:
        out = Path(json.loads(Path(path).read_text())['output'])
        if not out.is_dir() or (out / 'result.json').is_file():
            continue
        lock = out / 'worker.lock'
        if lock.exists():
            with lock.open('a') as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
        shutil.rmtree(out)
        removed += 1
    return removed


def check_init(rows):
    """B0H+ shares every b0 parameter except the encoder, whose input is the
    pooled cycle means concatenated with the slot-mean phase token; the level
    latent depends on the embedding through that token."""
    import torch
    from scripts.vldbj_m1.proto import build_model
    sources = e14_sources()
    best = b0_best(rows)
    base = {'code_commit': '', 'source_hashes': {}, 'smoke': True, 'stage': 'check'}
    for row in rows:
        b = best[(row['dataset'], row['horizon'])]
        b0, _, _, _ = build_model(job_for(base, sources, b, 'history'))
        b0h, _, _, _ = build_model(job_for(base, sources, b, 'history+token'))
        s, t = b0.state_dict(), b0h.state_dict()
        enc = 'lowrank_interaction_branch.lowrank_head.encoder.weight'
        enc_bias = 'lowrank_interaction_branch.lowrank_head.encoder.bias'
        same_keys = set(s) == set(t)
        differ = sorted(k for k in s if s[k].shape != t[k].shape or not torch.equal(s[k], t[k]))
        head = b0h.lowrank_interaction_branch.lowrank_head
        b0h.train()
        x = torch.randn(2, b0h.seq_len, b0h.enc_in)
        mark = torch.zeros(2, b0h.seq_len, 4)
        y = torch.randn(2, b0h.seq_len + b0h.pred_len, b0h.enc_in)
        ymark = torch.zeros(2, b0h.seq_len + b0h.pred_len, 4)
        out = b0h(x, mark, b0h._build_decoder_input(y), ymark)[0][:, -b0h.pred_len:]
        tokens = head.phase_tokens
        latent = head.encode_latent(x)[0]
        grad = torch.autograd.grad(latent.sum(), b0h.embedding.projection.weight)[0]
        width = head.pooled_len + b0h.latent_dim
        only_encoder = set(differ) <= {enc, enc_bias} and enc in differ
        good = (same_keys and only_encoder and tuple(t[enc].shape) == (3, width)
                and tuple(out.shape) == (2, b0h.pred_len, b0h.enc_in) and grad.abs().sum() > 0)
        print(f"{row['dataset']:>11}-{row['horizon']:>3}: pooled={head.pooled_len:>2} D={b0h.latent_dim:>3} "
              f"slots={tokens.shape[2]:>2} encoder={tuple(t[enc].shape)} "
              f"only_encoder_differs={only_encoder} embedding_grad={grad.abs().sum().item() > 0} "
              f"{'' if good else 'FAIL'}")
        if not good:
            raise SystemExit('B0H check failed')
    print('check passed: only the level encoder differs; the level path reaches the embedding')


def b0hp_candidates(scratch, row, best):
    """Every finished B0H+ run for one setting, from the single-config and sweep stages."""
    out = []
    record = scratch / STAGE / 'runs' / f"{best['run']}__b0hp" / 'result.json'
    if record.is_file():
        r = json.loads(record.read_text())['best']
        out.append(dict(mse=r['test_mse'], mae=r['test_mae'], stage=STAGE,
                        config=(best['period'], best['gate'], best['ext'], best['bb'])))
    for path in (scratch / SWEEP_STAGE / 'jobs').glob('*.json'):
        job = json.loads(path.read_text())
        if job['dataset'] != row['dataset'] or int(job['horizon']) != row['horizon']:
            continue
        run = Path(job['output']) / 'result.json'
        if not run.is_file():
            continue
        r = json.loads(run.read_text())['best']
        cfg = job['source']['config']
        out.append(dict(mse=r['test_mse'], mae=r['test_mae'], stage=SWEEP_STAGE,
                        config=(int(cfg.get('period', 24)), job['variant']['gate_init'],
                                job['extension_lr_mult'], job['backbone_lr_mult'])))
    return out


def report(scratch, rows):
    best = b0_best(rows)
    print(f"{'Dataset':>11} {'H':>4} | {'LA':>8} {'b0':>8} {'B0H+':>8} {'vs b0':>7} {'vs LA':>7} "
          f"| {'best cfg':>28}")
    per = {}
    for row in rows:
        b = best[(row['dataset'], row['horizon'])]
        cands = b0hp_candidates(scratch, row, b)
        if not cands:
            print(f"{row['dataset']:>11} {row['horizon']:>4} | (not finished)")
            continue
        pick = min(cands, key=lambda c: c['mse'])
        mse, mae = pick['mse'], pick['mae']
        per.setdefault(row['dataset'], []).append((row['m1_mse'], b['mse'], mse, row['m1_mae'], b['mae'], mae))
        cfg = 'p%d g%g lr%g bb%g %s' % (*pick['config'], pick['stage'])
        print(f"{row['dataset']:>11} {row['horizon']:>4} | {row['m1_mse']:>8.5f} {b['mse']:>8.5f} {mse:>8.5f} "
              f"{(mse - b['mse']) / b['mse'] * 100:>+6.2f}% {(mse - row['m1_mse']) / row['m1_mse'] * 100:>+6.2f}% "
              f"| {cfg:>28}")
    print()
    print(f"{'Dataset':>11} | {'LA':>7} {'b0':>7} {'B0H':>7} {'vs b0':>7} {'vs LA':>7} {'vs Gold':>8} | "
          f"{'LA MAE':>7} {'b0 MAE':>7} {'B0H MAE':>7} {'vs b0':>7} {'vs LA':>7}")
    totals = []
    for dataset in DATASETS:
        items = per.get(dataset, [])
        if len(items) != 4:
            continue
        la, b0, bh, lam, b0m, bhm = (sum(i[k] for i in items) / 4 for k in range(6))
        gold = GOLDEN[dataset][0]
        totals.append((la, b0, bh, lam, b0m, bhm, gold))
        print(f"{dataset:>11} | {la:>7.4f} {b0:>7.4f} {bh:>7.4f} {(bh - b0) / b0 * 100:>+6.2f}% "
              f"{(bh - la) / la * 100:>+6.2f}% {(bh - gold) / gold * 100:>+7.2f}% | {lam:>7.4f} {b0m:>7.4f} "
              f"{bhm:>7.4f} {(bhm - b0m) / b0m * 100:>+6.2f}% {(bhm - lam) / lam * 100:>+6.2f}%")
    if len(totals) == len(DATASETS):
        la, b0, bh, lam, b0m, bhm, gold = (sum(t[k] for t in totals) / len(totals) for k in range(7))
        print(f"{'mean':>11} | {la:>7.4f} {b0:>7.4f} {bh:>7.4f} {(bh - b0) / b0 * 100:>+6.2f}% "
              f"{(bh - la) / la * 100:>+6.2f}% {(bh - gold) / gold * 100:>+7.2f}% | {lam:>7.4f} {b0m:>7.4f} "
              f"{bhm:>7.4f} {(bhm - b0m) / b0m * 100:>+6.2f}% {(bhm - lam) / lam * 100:>+6.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, default=DEFAULT_SCRATCH)
    parser.add_argument('--phase', default='all', choices=('all', 'non-traffic', 'traffic'))
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--slots-per-gpu', type=int, default=2)
    parser.add_argument('--check-init', action='store_true')
    parser.add_argument('--sweep', action='store_true',
                        help='local grid around each b0 winner instead of the single best config')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--datasets', default=None)
    args = parser.parse_args()

    rows = ROWS
    if args.datasets:
        keep = {d.strip() for d in args.datasets.split(',')}
        rows = [r for r in rows if r['dataset'] in keep]
    if args.check_init:
        check_init(rows)
        return
    if args.report:
        report(args.scratch, rows)
        return
    builder = build_sweep_jobs if args.sweep else build_jobs
    paths = [str(p) for p in builder(args.scratch, rows, args.phase)]
    stage = SWEEP_STAGE if args.sweep else STAGE
    print(f'{args.phase}: {len(paths)} jobs under {args.scratch / stage}', flush=True)
    if not args.run:
        return
    gpus = [int(g) for g in args.gpus.split(',') for _ in range(args.slots_per_gpu)]
    print('launching on', gpus, flush=True)
    for attempt in (1, 2, 3):
        try:
            run_jobs(paths, gpus)
            break
        except RuntimeError as exc:
            print(f'pass {attempt} ended with failures: {exc}', flush=True)
            print(f'cleared {clear_incomplete(paths)} incomplete run directories', flush=True)
            if attempt == 3:
                raise


if __name__ == '__main__':
    main()
