#!/usr/bin/env python3
"""Aggregate the two supplementary VLDBJ experiments after their artifacts are pulled.

(A) Envelope read-out on the retrained PhaseFormer checkpoints (ladder column 1) in the format
    of Table tab:envelope, next to the PhaseFormer-LA values (final protocol configuration,
    seed-2021 checkpoint analyses of the paper), plus per-horizon numbers.
(B) Static-q ablation: per-dataset MSE/MAE averages of static-q next to the PhaseFormer,
    PhaseFormer-L, PhaseFormer-LA and LA (modulation off) protocol numbers, and the number of
    the 28 settings where full LA beats static-q.

Inputs (all under research_runs/, none under paper/):
  --pf-root      vldbj_pf_envelope      runs/<ds>_H<h>_S<seed>__pf/{analysis.json,analysis_windows.npz}
  --staticq-root vldbj_staticq          results.csv (staticq.py --stage collect)
  --pilot-root   vldbj_m1_pilot         main/, tune/, confirm/ of the LA protocol runs
  --e14-csv      phaseformer_L_e14_main_v1/results.csv   (PhaseFormer column of the ladder)

Outputs (--out): envelope_pf_vs_la.md/.tex/.json/_per_horizon.csv, staticq_vs_ladder.md/.json,
staticq_per_setting.csv. Missing seeds/settings are reported, never imputed.

Run:  python scripts/summarize_vldbj_supplementary.py --out research_runs/vldbj_supplementary_summary
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic')
HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022, 2023)
SHARED_GATE = {'ETTh1': False, 'ETTh2': True, 'ETTm1': True, 'ETTm2': True,
               'Weather': False, 'Electricity': False, 'Traffic': False}
# Final LA configuration of the paper (scripts/vldbj_m1/protocol.py FINAL_VARIANTS): indices of
# the eight per-setting extension variants that survived the confirm round; others use main.
FINAL_VARIANTS = {('ETTm1', 96): 1, ('ETTm2', 96): 5, ('Electricity', 720): 1, ('Traffic', 720): 4,
                  ('Weather', 96): 5, ('Weather', 192): 0, ('Weather', 336): 5, ('Weather', 720): 3}


def gate_tag(dataset):
    return 'shared' if SHARED_GATE[dataset] else 'perC'


# ---------------------------------------------------------------- envelope (A)
def envelope_record(run_dir):
    """The tab:envelope quantities of one analyzed checkpoint (floor recomputed from the windows)."""
    run_dir = Path(run_dir)
    a = json.loads((run_dir / 'analysis.json').read_text())
    z = np.load(run_dir / 'analysis_windows.npz')
    # analysis.json stores a floor that is off by a factor of pred_len in old runs; recompute.
    floor = float((z['sigma'] ** 2 * z['dist'] ** 2).sum() / z['mse'].sum())
    util = float((np.abs(z['levF'] - a['constants']['c0']) / a['constants']['C1']).max())
    return {'C1': a['constants']['C1'], 'B_phi': a['constants']['B_phi'],
            'g_s': a['gates']['g_shape_mean'], 'g_l': a['gates']['g_level_mean'],
            'outside': a['level']['share_outside_interval'], 'floor': floor,
            'omega_over': a['amplitude']['share_demand_over_budget'], 'util_max': util,
            'name': a['name']}


def la_dir(pilot, dataset, horizon, seed=2021):
    vid = FINAL_VARIANTS.get((dataset, horizon))
    base = f'{dataset}_H{horizon}_S{seed}__m1__S2__{gate_tag(dataset)}'
    return pilot / ('main/runs/' + base if vid is None else f'tune/runs/{base}__t{vid}')


def pf_dir(pf_root, dataset, horizon, seed):
    return pf_root / 'runs' / f'{dataset}_H{horizon}_S{seed}__pf'


FIELDS = ('C1', 'B_phi', 'g_s', 'outside', 'floor', 'omega_over')


def mean_fields(records):
    return {k: float(np.mean([r[k] for r in records])) for k in FIELDS} if records else None


def envelope_tables(pf_root, pilot):
    rows, missing = [], []
    per_h = {}
    for dataset in DATASETS:
        for horizon in HORIZONS:
            la = envelope_record(la_dir(pilot, dataset, horizon))
            per_h[('LA', dataset, horizon, 2021)] = la
            for seed in SEEDS:
                d = pf_dir(pf_root, dataset, horizon, seed)
                if (d / 'analysis.json').exists():
                    per_h[('PF', dataset, horizon, seed)] = envelope_record(d)
                else:
                    missing.append(f'{dataset}-{horizon}-S{seed}')
    summary = {}
    for dataset in DATASETS:
        la = [per_h[('LA', dataset, h, 2021)] for h in HORIZONS]
        pf3 = [per_h[('PF', dataset, h, s)] for h in HORIZONS for s in SEEDS if ('PF', dataset, h, s) in per_h]
        pf1 = [per_h[('PF', dataset, h, 2021)] for h in HORIZONS if ('PF', dataset, h, 2021) in per_h]
        summary[dataset] = {'LA_seed2021': mean_fields(la), 'PF_seed2021': mean_fields(pf1),
                            'PF_3seed': mean_fields(pf3), 'PF_n': len(pf3)}
    util = {'PF': max((r['util_max'] for k, r in per_h.items() if k[0] == 'PF'), default=None),
            'LA': max(r['util_max'] for k, r in per_h.items() if k[0] == 'LA')}
    return per_h, summary, missing, util


def pct(x, digits=1):
    return f'{100 * x:.{digits}f}\\%'


def render_envelope(summary, util, missing, tex=False):
    lines = []
    if tex:
        lines += [r'\begin{tabular}{lccccc|cccccc}', r'\toprule',
                  r' & \multicolumn{5}{c|}{\sysname (retrained, 3 seeds)} & \multicolumn{6}{c}{\sysLA (seed 2021)} \\',
                  r'Dataset & $C_1$ & $B_\varphi$ & $\Lambda\notin\mathcal I_\varphi$ & floor/MSE & $\Omega$ over env.'
                  r' & $C_1$ & $B_\varphi$ & $g_s$ & $\Lambda\notin\mathcal I_\varphi$ & floor/MSE & $\Omega$ over env. \\',
                  r'\midrule']
        for ds in DATASETS:
            p, a = summary[ds]['PF_3seed'], summary[ds]['LA_seed2021']
            if p is None:
                continue
            lines.append(f"{ds} & {p['C1']:.2f} & {p['B_phi']:.2f} & {pct(p['outside'])} & "
                         f"{pct(p['floor'], 2 if p['floor'] < 0.01 else 1)} & {pct(p['omega_over'])} & "
                         f"{a['C1']:.2f} & {a['B_phi']:.2f} & {a['g_s']:.2f} & {pct(a['outside'])} & "
                         f"{pct(a['floor'], 2 if a['floor'] < 0.01 else 1)} & {pct(a['omega_over'])} \\\\")
        lines += [r'\bottomrule', r'\end{tabular}']
        return '\n'.join(lines) + '\n'
    lines.append('# Envelopes: retrained PhaseFormer vs PhaseFormer-LA (protocol configuration)\n')
    lines.append('All values averaged over the four horizons. PhaseFormer: plain backbone, no shape gate, '
                 'so the amplitude envelope is B_phi itself (g_s = 0). LA: seed-2021 checkpoint analyses '
                 'of the paper (final protocol configuration, modulation off). Floor recomputed as '
                 'sum(sigma^2 dist^2)/sum(mse).\n')
    if missing:
        lines.append(f'**Incomplete PhaseFormer read-out: {len(missing)} analyses missing** '
                     f'(e.g. {", ".join(missing[:5])}).\n')
    head = ('| Dataset | model | C1 | B_phi | g_s | outside I_phi | floor/MSE | Omega over env. |\n'
            '|---|---|---:|---:|---:|---:|---:|---:|')
    lines.append(head)
    for ds in DATASETS:
        for label, key in (('PF 3-seed', 'PF_3seed'), ('PF seed 2021', 'PF_seed2021'), ('LA seed 2021', 'LA_seed2021')):
            r = summary[ds][key]
            if r is None:
                continue
            lines.append(f"| {ds} | {label} | {r['C1']:.2f} | {r['B_phi']:.2f} | {r['g_s']:.2f} | "
                         f"{100 * r['outside']:.1f}% | {100 * r['floor']:.2f}% | {100 * r['omega_over']:.1f}% |")
    lines.append('')
    lines.append(f"Max level utilization max|levF-c0|/C1 over all checkpoints: PF {util['PF']}, LA {util['LA']:.2f}")
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------- ladder + static-q (B)
def load_e14(e14_csv):
    out = {}
    with open(e14_csv) as handle:
        for row in csv.DictReader(handle):
            if row['arm'] == 'phase_only' and row['status'] == 'read':
                out[(row['dataset'], int(row['horizon']), int(row['seed']))] = (
                    float(row['test_mse']), float(row['test_mae']))
    return out


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def la_reads(pilot, dataset, horizon, seed):
    """LA (protocol configuration) native and modulation-off reads of one run."""
    vid = FINAL_VARIANTS.get((dataset, horizon))
    base = f'{dataset}_H{horizon}_S{seed}__m1__S2__{gate_tag(dataset)}'
    if vid is None:
        return read_json(pilot / 'main/runs' / base / 'test_result.json')
    stage = 'tune' if seed == 2021 else 'confirm'
    return read_json(pilot / stage / 'runs' / f'{base}__t{vid}' / 'tune_result.json')


def ladder_numbers(pilot, e14_csv):
    e14 = load_e14(e14_csv)
    table = {}
    for dataset in DATASETS:
        for horizon in HORIZONS:
            for seed in SEEDS:
                key = (dataset, horizon, seed)
                b0 = read_json(pilot / 'main/runs' / f'{dataset}_H{horizon}_S{seed}__b0__S2__{gate_tag(dataset)}'
                               / 'test_result.json')
                la = la_reads(pilot, dataset, horizon, seed)
                if key not in e14 or b0 is None or la is None:
                    continue
                table[key] = {'PF': e14[key], 'L': (b0['test_mse'], b0['test_mae']),
                              'LA': (la['test_mse'], la['test_mae']),
                              'LAoff': (la['test_mse_modulation_off'], la['test_mae_modulation_off'])}
    return table


def load_staticq(path):
    out = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            out[(row['dataset'], int(row['horizon']), int(row['seed']))] = {
                'SQ': (float(row['test_mse']), float(row['test_mae'])),
                'SQoff': (float(row['test_mse_modulation_off']), float(row['test_mae_modulation_off'])),
                'epochs': int(row['epochs']), 'params': int(row['modulation_params'])}
    return out


ARMS = ('PF', 'L', 'LA', 'LAoff', 'SQ', 'SQoff')


def staticq_tables(ladder, sq):
    per_setting = {}
    missing = []
    for dataset in DATASETS:
        for horizon in HORIZONS:
            seeds = [s for s in SEEDS if (dataset, horizon, s) in ladder and (dataset, horizon, s) in sq]
            if len(seeds) < len(SEEDS):
                missing.append(f'{dataset}-{horizon}: {len(seeds)}/3 seeds')
                continue
            rec = {'seeds': seeds}
            for arm in ARMS:
                src = sq if arm.startswith('SQ') else ladder
                vals = [src[(dataset, horizon, s)][arm] for s in seeds]
                rec[arm] = (float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals])))
            rec['LA_beats_SQ_seeds'] = sum(ladder[(dataset, horizon, s)]['LA'][0] < sq[(dataset, horizon, s)]['SQ'][0]
                                           for s in seeds)
            per_setting[(dataset, horizon)] = rec
    dataset_avg = {}
    for dataset in DATASETS:
        recs = [per_setting[(dataset, h)] for h in HORIZONS if (dataset, h) in per_setting]
        if len(recs) == len(HORIZONS):
            dataset_avg[dataset] = {arm: tuple(float(np.mean([r[arm][i] for r in recs])) for i in (0, 1))
                                    for arm in ARMS}
    counts = {'settings': len(per_setting),
              'LA_beats_SQ_mse': sum(r['LA'][0] < r['SQ'][0] for r in per_setting.values()),
              'LA_beats_SQ_mae': sum(r['LA'][1] < r['SQ'][1] for r in per_setting.values()),
              'LA_beats_SQ_mse_seed_pairs': sum(r['LA_beats_SQ_seeds'] for r in per_setting.values()),
              'seed_pairs': 3 * len(per_setting),
              'LA_beats_L_mse': sum(r['LA'][0] < r['L'][0] for r in per_setting.values()),
              'SQ_beats_L_mse': sum(r['SQ'][0] < r['L'][0] for r in per_setting.values()),
              'SQ_beats_PF_mse': sum(r['SQ'][0] < r['PF'][0] for r in per_setting.values()),
              'LA_beats_PF_mse': sum(r['LA'][0] < r['PF'][0] for r in per_setting.values()),
              'native_beats_off_SQ_mse': sum(r['SQ'][0] < r['SQoff'][0] for r in per_setting.values())}
    return per_setting, dataset_avg, counts, missing


def render_staticq(dataset_avg, counts, missing, per_setting):
    lines = ['# Static-q ablation next to the protocol ladder\n',
             'Test MSE/MAE averaged over the four horizons and three seeds (protocol configuration). '
             'PF = retrained PhaseFormer, L = PhaseFormer-L, LA = PhaseFormer-LA, LA off = LA with the '
             'modulation disabled on the trained checkpoint, static-q = LA with q a learned per-channel '
             'constant (this experiment; its modulation-off read is the last column).\n']
    if missing:
        lines.append(f'**Incomplete: {len(missing)} settings missing/partial** ({"; ".join(missing[:6])}).\n')
    lines.append('| Dataset | PF MSE | PF MAE | L MSE | L MAE | LA MSE | LA MAE | LA off MSE | LA off MAE |'
                 ' static-q MSE | static-q MAE | static-q off MSE |')
    lines.append('|---|' + '---:|' * 11)
    for ds in DATASETS:
        if ds not in dataset_avg:
            continue
        a = dataset_avg[ds]
        lines.append(f"| {ds} | " + ' | '.join(f'{a[k][i]:.4f}' for k, i in
                     (('PF', 0), ('PF', 1), ('L', 0), ('L', 1), ('LA', 0), ('LA', 1), ('LAoff', 0), ('LAoff', 1),
                      ('SQ', 0), ('SQ', 1), ('SQoff', 0))) + ' |')
    lines.append('')
    n = counts['settings']
    lines.append(f"Settings with all three seeds: {n}/28.")
    lines.append(f"- full LA beats static-q on **{counts['LA_beats_SQ_mse']}/{n}** settings in 3-seed mean MSE "
                 f"({counts['LA_beats_SQ_mae']}/{n} in MAE; {counts['LA_beats_SQ_mse_seed_pairs']}/"
                 f"{counts['seed_pairs']} seed-paired MSE comparisons).")
    lines.append(f"- LA beats L on {counts['LA_beats_L_mse']}/{n}, static-q beats L on {counts['SQ_beats_L_mse']}/{n}; "
                 f"LA beats PF on {counts['LA_beats_PF_mse']}/{n}, static-q beats PF on {counts['SQ_beats_PF_mse']}/{n}.")
    lines.append(f"- static-q native beats its own modulation-off read on {counts['native_beats_off_SQ_mse']}/{n}.")
    gain = {}
    for ds in dataset_avg:
        a = dataset_avg[ds]
        total = a['L'][0] - a['LA'][0]
        gain[ds] = (a['L'][0] - a['SQ'][0]) / total if abs(total) > 1e-6 else float('nan')
    lines.append('')
    lines.append('Share of the L -> LA MSE change that static-q already delivers, '
                 '(L - static-q)/(L - LA), dataset average (unstable where L - LA is tiny): '
                 + ', '.join(f'{ds} {v:.2f}' for ds, v in gain.items()))
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pf-root', type=Path, default=ROOT / 'research_runs/vldbj_pf_envelope')
    parser.add_argument('--staticq-root', type=Path, default=ROOT / 'research_runs/vldbj_staticq')
    parser.add_argument('--pilot-root', type=Path, default=ROOT / 'research_runs/vldbj_m1_pilot')
    parser.add_argument('--e14-csv', type=Path,
                        default=ROOT / 'research_runs/phaseformer_L_e14_main_v1/results.csv')
    parser.add_argument('--out', type=Path, default=ROOT / 'research_runs/vldbj_supplementary_summary')
    parser.add_argument('--only', choices=('A', 'B'), help='aggregate one experiment only')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.only != 'B':
        per_h, summary, missing, util = envelope_tables(args.pf_root, args.pilot_root)
        (args.out / 'envelope_pf_vs_la.md').write_text(render_envelope(summary, util, missing))
        (args.out / 'envelope_pf_vs_la.tex').write_text(render_envelope(summary, util, missing, tex=True))
        (args.out / 'envelope_pf_vs_la.json').write_text(json.dumps(
            {'summary': summary, 'missing_pf_analyses': missing, 'max_util': util}, indent=1))
        with (args.out / 'envelope_per_horizon.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(('model', 'dataset', 'horizon', 'seed') + FIELDS + ('g_l', 'util_max'))
            for (model, ds, h, seed), r in sorted(per_h.items()):
                writer.writerow([model, ds, h, seed] + [r[k] for k in FIELDS] + [r['g_l'], r['util_max']])
        print((args.out / 'envelope_pf_vs_la.md').read_text())

    if args.only != 'A':
        ladder = ladder_numbers(args.pilot_root, args.e14_csv)
        sq = load_staticq(args.staticq_root / 'results.csv')
        per_setting, dataset_avg, counts, missing = staticq_tables(ladder, sq)
        text = render_staticq(dataset_avg, counts, missing, per_setting)
        (args.out / 'staticq_vs_ladder.md').write_text(text)
        (args.out / 'staticq_vs_ladder.json').write_text(json.dumps(
            {'dataset_avg': dataset_avg, 'counts': counts, 'incomplete': missing}, indent=1))
        with (args.out / 'staticq_per_setting.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['dataset', 'horizon'] + [f'{a}_{m}' for a in ARMS for m in ('mse', 'mae')]
                            + ['LA_beats_SQ_seeds_of_3'])
            for (ds, h), r in sorted(per_setting.items()):
                writer.writerow([ds, h] + [r[a][i] for a in ARMS for i in (0, 1)] + [r['LA_beats_SQ_seeds']])
        print(text)


if __name__ == '__main__':
    main()
