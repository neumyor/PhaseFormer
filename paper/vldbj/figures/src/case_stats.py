#!/usr/bin/env python3
"""Mean trend and periodic errors of PhaseFormer and PhaseFormer+ over all test windows, and the share of
(window, channel) pairs and of windows on which PhaseFormer+ has the lower trend error."""
import csv, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path.home() / 'niuyiming'))
import case_study as cs
from gate_analysis import register_heads
from scripts.vldbj_m1.run import e14_sources
from src.dataset.data_factory import data_provider
P = 24
def split_err(e):
    B, H, C = e.shape
    c = e.reshape(B, H // P, P, C); m = c.mean(2, keepdim=True)
    return (m.expand_as(c) ** 2).mean((1, 2)), ((c - m) ** 2).mean((1, 2))
def main():
    register_heads(); sources = e14_sources(); dev = torch.device('cuda:0'); out = {}
    rows = {(r['dataset'], int(r['horizon'])): r for r in csv.DictReader(cs.CSV.open())}
    for s in sys.argv[1:]:
        d, h = s.split('-'); h = int(h)
        plus, ap = cs.load(cs.job_of(rows[(d, h)]['run']), cs.checkpoint_of(rows[(d, h)]['run']))
        src = sources[(d, h, 2021)]
        from scripts.run_lowrank_interaction_stage2 import construct as legacy_construct
        pf, _ = legacy_construct(src['config'])
        pf.load_state_dict(torch.load(cs.E14_ROOT / src['source_run'] / 'attempts' / '001' / 'checkpoints' / 'best.ckpt', map_location='cpu', weights_only=False)['state_dict'])
        plus.to(dev).eval(); pf.to(dev).eval()
        _, loader = data_provider(ap.dataset_args, 'test')
        acc = dict(trf=0., trp=0., pef=0., pep=0., n=0, win_better=0, pair_better=0, pairs=0)
        with torch.inference_mode():
            for bx, by, bxm, bym in loader:
                bx, by, bxm, bym = [v.float().to(dev) for v in (bx, by, bxm, bym)]
                yp = plus(bx, bxm, plus._build_decoder_input(by), bym)[0][:, -h:]
                yf = pf(bx, bxm, pf._build_decoder_input(by), bym)[0][:, -h:]
                y = by[:, -h:]
                tf, ef = split_err(yf - y); tp, ep = split_err(yp - y)
                acc['trf'] += tf.sum().item(); acc['trp'] += tp.sum().item()
                acc['pef'] += ef.sum().item(); acc['pep'] += ep.sum().item()
                acc['pairs'] += tf.numel(); acc['pair_better'] += (tp < tf).sum().item()
                acc['n'] += tf.shape[0]; acc['win_better'] += (tp.mean(1) < tf.mean(1)).sum().item()
        k = acc['pairs']
        out[s] = dict(windows=acc['n'], trend_pf=acc['trf'] / k, trend_plus=acc['trp'] / k, periodic_pf=acc['pef'] / k,
                      periodic_plus=acc['pep'] / k, window_share=acc['win_better'] / acc['n'], pair_share=acc['pair_better'] / k)
        print(s, out[s], flush=True)
    Path(cs.OUT.parent / 'case_trend_v1' / ('stats_' + '_'.join(sys.argv[1:]) + '.json')).write_text(json.dumps(out, indent=1))
main()
