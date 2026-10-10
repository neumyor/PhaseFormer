"""Dataset-level aggregation of the modulation activation analysis."""
import glob
import re

import numpy as np

RUNS = '/home/yyk/yyk03/niuyiming/PhaseFormer_vldbj_m1_20261010/research_runs/vldbj_m1_pilot/main/runs'

per_ds = {}
for p in sorted(glob.glob(f'{RUNS}/*__m1__*/analysis_windows.npz')):
    name = re.match(r'.*/(\w+)_H(\d+)_S\d+__m1__', p)
    ds, H = name.group(1), int(name.group(2))
    z = np.load(p)
    lam, q = z['lambda'], z['q_scalar']
    lc = []
    for c in range(lam.shape[1]):
        if lam[:, c].std() > 0 and q[:, c].std() > 0:
            lc.append(np.corrcoef(q[:, c], lam[:, c])[0, 1])
    lc = np.abs(np.array(lc))
    lc = lc[~np.isnan(lc)]
    per_ds.setdefault(ds, []).append((H, lc.mean(), (lc > 0.3).mean(), lc.size))

print(f"{'dataset':<12} {'mean|corr|':>10} {'share>0.3':>9}  per-horizon mean|corr|")
for ds, rows in per_ds.items():
    rows.sort()
    mean_c = float(np.mean([r[1] for r in rows]))
    mean_s = float(np.mean([r[2] for r in rows]))
    detail = ' '.join(f'H{h}:{c:.2f}' for h, c, _, _ in rows)
    print('{:<12} {:10.3f} {:9.2f}  {}'.format(ds, mean_c, mean_s, detail))

# Sign consistency of the significant channels.
print()
for ds, rows in per_ds.items():
    for p in sorted(glob.glob(f'{RUNS}/{ds}_H*_S2021__m1__*/analysis_windows.npz')):
        z = np.load(p)
        lam, q = z['lambda'], z['q_scalar']
        sig_pos = sig_neg = 0
        for c in range(lam.shape[1]):
            if lam[:, c].std() > 0 and q[:, c].std() > 0:
                r = np.corrcoef(q[:, c], lam[:, c])[0, 1]
                if abs(r) > 0.3:
                    if r > 0:
                        sig_pos += 1
                    else:
                        sig_neg += 1
        h = re.search(r'_H(\d+)_', p).group(1)
        print(f'{ds}-H{h}: significant channels +{sig_pos} / -{sig_neg}')
