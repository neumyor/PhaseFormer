"""Same-protocol ladder: PhaseFormer (E14 phase_only) -> L (b0) -> LA (m1)."""
import csv
import glob
import json
import re
import statistics as st
from collections import defaultdict

E14_ROOT = '/home/yyk/yyk03/niuyiming/PhaseFormer'
RUNS_ROOT = '/home/yyk/yyk03/niuyiming/PhaseFormer_vldbj_m1_20261010/research_runs/vldbj_m1_pilot/main/runs'

e14 = defaultdict(list)
for r in csv.DictReader(open(f'{E14_ROOT}/research_runs/phaseformer_L_e14_main_v1/results.csv')):
    if r['arm'] == 'phase_only' and r['status'] == 'read':
        e14[(r['dataset'], int(r['horizon']))].append((float(r['test_mse']), float(r['test_mae'])))

runs = defaultdict(list)
for p in sorted(glob.glob(f'{RUNS_ROOT}/*/test_result.json')):
    d = json.load(open(p))
    m = re.match(r'(\w+)_(H\d+)_(S\d+)__(b0|m1)__S\d+__(\w+)', d['name'])
    runs[(m.group(1), int(m.group(2)[1:]), m.group(4))].append(d)

DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'Traffic', 'Weather']
print(f"{'setting':<18} {'PF(e14)':>8} {'L(b0)':>8} {'LA(m1)':>8} {'L%':>6} {'LA%':>6} {'mod-off%':>8}")
imp_L = imp_LA = 0
ds_sum = defaultdict(list)
for ds in DATASETS:
    for H in [96, 192, 336, 720]:
        pf = st.mean(x[0] for x in e14[(ds, H)])
        b0 = st.mean(x['test_mse'] for x in runs[(ds, H, 'b0')])
        m1 = st.mean(x['test_mse'] for x in runs[(ds, H, 'm1')])
        off = st.mean(x['test_mse_modulation_off'] for x in runs[(ds, H, 'm1')])
        dL = 100 * (b0 - pf) / pf
        dA = 100 * (m1 - pf) / pf
        dO = 100 * (m1 - off) / off
        imp_L += dL < 0
        imp_LA += dA < 0
        ds_sum[ds].append((pf, b0, m1))
        print('{:<18} {:8.4f} {:8.4f} {:8.4f} {:6.2f} {:6.2f} {:8.2f}'.format(
            f'{ds}-{H}', pf, b0, m1, dL, dA, dO))
print('branch (L) improves:', imp_L, '/28 ;  LA improves:', imp_LA, '/28')
print()
print('horizon-avg per dataset (PF / L / LA):')
for ds in DATASETS:
    v = ds_sum[ds]
    pf = st.mean(x[0] for x in v)
    b0 = st.mean(x[1] for x in v)
    m1 = st.mean(x[2] for x in v)
    print('{:<12} {:.4f} {:.4f} {:.4f}   L {:+.2f}%  LA {:+.2f}%'.format(
        ds, pf, b0, m1, 100 * (b0 - pf) / pf, 100 * (m1 - pf) / pf))
