#!/usr/bin/env python3
"""Collect every E1 tuning run (R0, R2, R3) into one table and select the final E1 configuration of each setting.

Two selections, both on seed 2021:
  val   the run with the lowest validation MSE of its best-validation checkpoint (ties: fewer parameters).
        This is the selection of the main tables of the paper replication.
  test  the selection of the tuning stages: the best score = max(mse / B0H+ mse, mae / B0H+ mae) on the test
        split (R0: lowest test MSE, ties within 0.2% go to fewer parameters; then the best of R0, R2, R3).

Writes 00_protocol/all_runs.csv, e1_final_configs_28.csv and selection_comparison.csv under
research_runs/e1_paper_v1/.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.b0hp_structure_ablation as sa  # noqa: E402

TUNE = ROOT / 'research_runs/e1_tune_v1'
E1_DEFAULT = ROOT / 'research_runs/phaseformer_e12_v1/e1'
OUT = ROOT / 'research_runs/e1_paper_v1/00_protocol'
ORDER = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic')
HORIZONS = (96, 192, 336, 720)


def read_run(stage, jobs_dir, runs_dir, job_path):
    job = json.loads(job_path.read_text())
    out = Path(job['output'])
    record = out / 'result.json'
    if not record.is_file():
        return None
    result = json.loads(record.read_text())
    best = result['best']
    hp = job['source']['config']['hyperparams']
    split = job.get('lr_split') or {}
    cell = job.get('cell') or {}
    return dict(
        stage=stage, dataset=job['dataset'], horizon=job['horizon'], seed=job['seed'], run_dir=str(out),
        job=str(job_path), head=job['head'], shared_gate=int(job['shared_gate']),
        latent_dim=hp['latent_dim'], routers=hp['phase_num_routers'], layers=hp['layers'],
        period=job['source']['config']['period'], heads=hp.get('phase_attn_heads', ''),
        huber_delta=hp.get('huber_delta', 1.0), gate_init=job['variant']['gate_init'],
        gate_mult=split.get('gate_mult', job['extension_lr_mult']), readout_mult=split.get('readout_mult', job['extension_lr_mult']),
        backbone_mult=job['backbone_lr_mult'], attn_dropout=hp.get('phase_attn_dropout', ''),
        pos_dropout=hp.get('phase_pos_dropout', ''), pred_dropout=hp.get('predictor_dropout', ''),
        revin_affine=int(bool(hp.get('revin_affine', False))), state_dim=job.get('rank_override', 3),
        params=result['parameters']['total'], val_mse=best['val_mse'], val_mae=best['val_mae'],
        test_mse=best['test_mse'], test_mae=best['test_mae'], epochs=result['epochs'],
        code_commit=result['code_commit'], checkpoint=int((out / 'checkpoints' / 'best.ckpt').is_file()),
        round=job.get('round', ''), cell=json.dumps(cell) if cell else '')


def collect():
    rows = []
    for stage, base in (('r0', TUNE / 'r0'), ('r2', TUNE / 'r2'), ('r3', TUNE / 'r3')):
        for path in sorted((base / 'jobs').glob('*.json')):
            row = read_run(stage, base / 'jobs', base / 'runs', path)
            if row:
                rows.append(row)
    for path in sorted((E1_DEFAULT / 'jobs').glob('*_S2021__*__e1.json')):      # the E1 default runs also exist for seeds 2022, 2023
        row = read_run('e1_default', E1_DEFAULT / 'jobs', E1_DEFAULT / 'runs', path)
        if row:
            rows.append(row)
    return rows


def main():
    record = {(r['dataset'], int(r['horizon'])): (float(r['test_mse']), float(r['test_mae']))
              for r in csv.DictReader(sa.settings_csv().open())}
    rows = collect()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'all_runs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault((r['dataset'], r['horizon']), []).append(r)
    final, compare = [], []
    for dataset in ORDER:
        for horizon in HORIZONS:
            key = (dataset, horizon)
            runs = by[key]
            bm, ba = record[key]
            score = lambda r: max(r['test_mse'] / bm, r['test_mae'] / ba)
            val_sel = min(runs, key=lambda r: (r['val_mse'], r['params']))
            r0 = [r for r in runs if r['stage'] in ('r0', 'e1_default')]
            best0 = min(r['test_mse'] for r in r0)
            c0 = min((r for r in r0 if r['test_mse'] <= best0 * 1.002), key=lambda r: (r['params'], r['test_mse']))
            cands = [c0]
            for stage in ('r2', 'r3'):
                pool = [r for r in runs if r['stage'] == stage]
                if pool:
                    cands.append(min(pool, key=score))
            test_sel = min(cands, key=score)
            for kind, r in (('val', val_sel), ('test', test_sel)):
                final.append(dict(selection=kind, **r))
            compare.append(dict(
                dataset=dataset, horizon=horizon, n_runs=len(runs),
                same_run=int(val_sel['run_dir'] == test_sel['run_dir']),
                val_sel_stage=val_sel['stage'], val_sel_val_mse=val_sel['val_mse'], val_sel_test_mse=val_sel['test_mse'],
                val_sel_test_mae=val_sel['test_mae'], val_sel_params=val_sel['params'],
                test_sel_stage=test_sel['stage'], test_sel_val_mse=test_sel['val_mse'], test_sel_test_mse=test_sel['test_mse'],
                test_sel_test_mae=test_sel['test_mae'], test_sel_params=test_sel['params'],
                b0hp_test_mse=bm, b0hp_test_mae=ba))
    with (OUT / 'e1_final_configs_28.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(final[0]))
        writer.writeheader()
        writer.writerows(final)
    with (OUT / 'selection_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compare[0]))
        writer.writeheader()
        writer.writerows(compare)
    print(f'{len(rows)} runs, {len(compare)} settings, same run for both selections on '
          f'{sum(c["same_run"] for c in compare)}')


if __name__ == '__main__':
    main()
