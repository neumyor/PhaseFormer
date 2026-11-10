#!/usr/bin/env python3
"""Candidate windows for the trend case study.

For each setting, runs PhaseFormer and PhaseFormer+ over the test split and keeps, per
(window, channel), the trend error (cycle levels, P=24) and periodic error of both models.
The candidates with the largest trend-error reduction are saved together with the last
cycles of the input window.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / 'niuyiming'))
import case_study as cs  # noqa: E402
from gate_analysis import register_heads  # noqa: E402
from scripts.vldbj_m1.run import e14_sources  # noqa: E402
from src.dataset.data_factory import data_provider  # noqa: E402

OUT = cs.OUT.parent / 'case_trend_v1'
P = 24
KEEP = 600
SETTINGS = [s.split('-') for s in sys.argv[1:]] or [('ETTh2', '96')]


def split_err(e):
    # e: (B, H, C) forecast error; returns trend and periodic MSE per (window, channel)
    B, H, C = e.shape
    c = e.reshape(B, H // P, P, C)
    m = c.mean(2, keepdim=True)
    return (m.expand_as(c) ** 2).mean((1, 2)), ((c - m) ** 2).mean((1, 2))


def main():
    register_heads()
    sources = e14_sources()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda:0')
    for d, h in SETTINGS:
        h = int(h)
        rows = cs.rows_of() if hasattr(cs, 'rows_of') else None
        import csv
        rows = {(r['dataset'], int(r['horizon'])): r for r in csv.DictReader(cs.CSV.open())}
        plus, args_plus = cs.load(cs.job_of(rows[(d, h)]['run']), cs.checkpoint_of(rows[(d, h)]['run']))
        src = sources[(d, h, 2021)]
        from scripts.run_lowrank_interaction_stage2 import construct as legacy_construct
        pf, args_pf = legacy_construct(src['config'])
        ckpt = cs.E14_ROOT / src['source_run'] / 'attempts' / '001' / 'checkpoints' / 'best.ckpt'
        pf.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['state_dict'])
        plus.to(dev).eval(); pf.to(dev).eval()
        _, loader = data_provider(args_plus.dataset_args, 'test')
        cand = []
        base = 0
        with torch.inference_mode():
            for bx, by, bxm, bym in loader:
                bx, by, bxm, bym = [v.float().to(dev) for v in (bx, by, bxm, bym)]
                yp = plus(bx, bxm, plus._build_decoder_input(by), bym)[0][:, -h:]
                yf = pf(bx, bxm, pf._build_decoder_input(by), bym)[0][:, -h:]
                y = by[:, -h:]
                tr_f, pe_f = split_err(yf - y)
                tr_p, pe_p = split_err(yp - y)
                gain = tr_f - tr_p
                B, C = gain.shape
                k = min(KEEP, B * C)
                val, idx = gain.reshape(-1).topk(k)
                for v, i in zip(val.tolist(), idx.tolist()):
                    b, c = divmod(i, C)
                    cand.append((v, base + b, c, tr_f[b, c].item(), tr_p[b, c].item(), pe_f[b, c].item(), pe_p[b, c].item(),
                                 bx[b, -4 * P:, c].cpu().numpy(), y[b, :, c].cpu().numpy(),
                                 yf[b, :, c].cpu().numpy(), yp[b, :, c].cpu().numpy()))
                base += B
                cand.sort(key=lambda t: -t[0])
                cand = cand[:KEEP * 4]
        np.savez_compressed(OUT / ('%s-%d.npz' % (d, h)),
                            gain=np.array([c[0] for c in cand]), window=np.array([c[1] for c in cand]),
                            channel=np.array([c[2] for c in cand]), tr_pf=np.array([c[3] for c in cand]),
                            tr_plus=np.array([c[4] for c in cand]), pe_pf=np.array([c[5] for c in cand]),
                            pe_plus=np.array([c[6] for c in cand]), hist=np.stack([c[7] for c in cand]),
                            truth=np.stack([c[8] for c in cand]), pf=np.stack([c[9] for c in cand]),
                            plus=np.stack([c[10] for c in cand]))
        print(d, h, 'saved', len(cand), 'top gain', cand[0][0], flush=True)


if __name__ == '__main__':
    main()
