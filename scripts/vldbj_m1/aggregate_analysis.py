"""Aggregate the VLDBJ checkpoint analyses into compact paper numbers."""
import glob
import json
import re
import sys
from collections import defaultdict

import numpy as np

RUNS = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/yyk/yyk03/niuyiming/PhaseFormer_vldbj_m1_20261010/research_runs/vldbj_m1_pilot/main/runs'

DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Electricity', 'Traffic']

records = {}
for p in sorted(glob.glob(f'{RUNS}/*/analysis.json')):
    d = json.load(open(p))
    records[d['name']] = d

rows = {}
for name, d in records.items():
    m = re.match(r'(\w+)_H(\d+)_S\d+__(b0|m1)__S\d+__(\w+)$', name)
    rows[(m.group(1), int(m.group(2)), m.group(3))] = d

out = {'per_setting': {}, 'gain_stratification': {}, 'dataset_avg': {}}

print(f"{'setting':<16} {'arm':<3} {'c0':>6} {'C1':>5} {'Cphi':>5} {'Bphi':>5} "
      f"{'gL':>5} {'gS':>5} {'out%':>5} {'floor%':>7} {'amp%':>5} {'ho-corr':>7} {'q-corr':>6}")
for ds in DATASETS:
    for H in (96, 192, 336, 720):
        for arm in ('b0', 'm1'):
            d = rows[(ds, H, arm)]
            k = f'{ds}-{H}-{arm}'
            out['per_setting'][k] = {
                'c0': d['constants']['c0'], 'C1': d['constants']['C1'],
                'C_phi': d['constants']['C_phi'], 'B_phi': d['constants']['B_phi'],
                'g_level': d['gates']['g_level_mean'], 'g_shape': d['gates']['g_shape_mean'],
                'share_outside_interval': d['level']['share_outside_interval'],
                'floor_share_of_mse': d['level']['floor_share_of_mse'],
                'share_demand_over_budget': d['amplitude']['share_demand_over_budget'],
                'mean_abs_lambda': d['level']['mean_abs_lambda'],
                'handover_corr': d['handover']['corr_branch_level_lambda'],
                'levF_miss': d['level']['mean_abs_levF_miss'],
                'modulation': d.get('modulation')}
            mod = d.get('modulation') or {}
            print('{:<16} {:<3} {:6.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.2f} {:5.1f} {:7.4f} {:5.1f} {:7.3f} {:6.3f}'.format(
                f'{ds}-{H}', arm, d['constants']['c0'], d['constants']['C1'],
                d['constants']['C_phi'], d['constants']['B_phi'],
                d['gates']['g_level_mean'], d['gates']['g_shape_mean'],
                100 * d['level']['share_outside_interval'],
                100 * d['level']['floor_share_of_mse'],
                100 * d['amplitude']['share_demand_over_budget'],
                d['handover']['corr_branch_level_lambda'],
                mod.get('corr_q_scalar_lambda', float('nan'))))

# Gain stratification: per-window test MSE gain of m1 over b0 by |Lambda| quintiles.
for ds in DATASETS:
    for H in (96, 192, 336, 720):
        b0 = rows[(ds, H, 'b0')]
        m1 = rows[(ds, H, 'm1')]
        zb = np.load(f"{RUNS}/{b0['name']}/analysis_windows.npz")
        zm = np.load(f"{RUNS}/{m1['name']}/analysis_windows.npz")
        # m1's base pass ran with the modulation off; the native forecast is the LA arm.
        mse_b0, mse_m1 = zb['mse'], zm['mse_native']   # (N, C) both seed 2021
        lam = zm['lambda']
        gain = mse_b0 - mse_m1
        alam = np.abs(lam).flatten()
        g = gain.flatten()
        qs = np.quantile(alam, [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(alam, qs)
        strat = []
        for q in range(5):
            mask = bins == q
            strat.append({'mean_abs_lambda': float(alam[mask].mean()),
                          'mean_gain': float(g[mask].mean()),
                          'share_improved': float((g[mask] > 0).mean()),
                          'n': int(mask.sum())})
        out['gain_stratification'][f'{ds}-{H}'] = strat
        print(f'{ds}-{H}: gain by |Lambda| quintile: ' +
              ' '.join(f'{s["mean_gain"]:+.4f}' for s in strat) +
              f'  (share improved overall {(g > 0).mean():.2f})')

# Dataset averages (over horizons) of the headline analysis numbers, m1 arm.
for ds in DATASETS:
    vals = defaultdict(list)
    for H in (96, 192, 336, 720):
        d = rows[(ds, H, 'm1')]
        vals['C_phi'].append(d['constants']['C_phi'])
        vals['B_phi'].append(d['constants']['B_phi'])
        vals['g_level'].append(d['gates']['g_level_mean'])
        vals['g_shape'].append(d['gates']['g_shape_mean'])
        vals['outside'].append(d['level']['share_outside_interval'])
        vals['floor'].append(d['level']['floor_share_of_mse'])
        vals['amp_over'].append(d['amplitude']['share_demand_over_budget'])
        vals['q_corr'].append(d['modulation']['corr_q_scalar_lambda'])
        vals['q_slope'].append(d['modulation']['slope_q_on_lambda'])
    out['dataset_avg'][ds] = {k: float(np.mean(v)) for k, v in vals.items()}

print()
print('dataset averages (m1):')
for ds in DATASETS:
    a = out['dataset_avg'][ds]
    print('{:<12} C_phi {:.2f}  B_phi {:.2f}  gL {:.2f} gS {:.2f}  outside {:.1%}  '
          'amp-over {:.1%}  q-corr {:.2f}  q-slope {:.2f}'.format(
              ds, a['C_phi'], a['B_phi'], a['g_level'], a['g_shape'],
              a['outside'], a['amp_over'], a['q_corr'], a['q_slope']))

with open('/home/yyk/yyk03/niuyiming/analysis_tmp/analysis_summary.json', 'w') as fh:
    json.dump(out, fh, indent=1)
print('\nwritten analysis_summary.json')
