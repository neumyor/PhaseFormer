"""Per-channel modulation statistics from the analysis npz files."""
import glob
import re

import numpy as np

RUNS = '/home/yyk/yyk03/niuyiming/PhaseFormer_vldbj_m1_20261010/research_runs/vldbj_m1_pilot/main/runs'

hdr = ('setting', '|corr(q,L)|', 'share>0.3', 'corr(q,Om)', 'share>0.3', 'q_var/mean', 'ones_share')
print('{:<16} {:>11} {:>9} {:>10} {:>10} {:>10} {:>9}'.format(*hdr))
for p in sorted(glob.glob(f'{RUNS}/*__m1__*/analysis_windows.npz')):
    name = re.match(r'.*/(\S+)/analysis_windows.npz', p).group(1)
    z = np.load(p)
    lam, q, om = z['lambda'], z['q_scalar'], z['omega']
    lc, oc = [], []
    for c in range(lam.shape[1]):
        if lam[:, c].std() > 0 and q[:, c].std() > 0:
            lc.append(np.corrcoef(q[:, c], lam[:, c])[0, 1])
        if om[:, c].std() > 0 and q[:, c].std() > 0:
            oc.append(np.corrcoef(q[:, c], om[:, c])[0, 1])
    lc, oc = np.abs(np.array(lc)), np.array(oc)
    qw = z['q_weights']                       # (d, width)
    d = qw.shape[0]
    ones = np.ones(d) / np.sqrt(d)
    share = (qw.T @ ones) ** 2 / (np.linalg.norm(qw, axis=0) ** 2 + 1e-12)
    ch_std = q.std(0).mean()
    ch_abs = np.abs(q).mean(0).mean()
    print('{:<16} {:11.3f} {:9.2f} {:10.3f} {:10.2f} {:10.3f} {:9.2f}'.format(
        name, lc.mean(), (lc > 0.3).mean(), oc.mean(), (np.abs(oc) > 0.3).mean(),
        ch_std / (ch_abs + 1e-12), share.mean()))
