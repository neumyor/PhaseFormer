#!/usr/bin/env python3
"""Extract what the trained level branch reads and writes, for the PCA study.

For one setting (dataset, horizon) the script loads the best b0 checkpoint of the
b0 search (seed 2021), runs the test loader, and stores for a fixed sample of
channels:
  zeta    the pooled cycle means the level encoder reads, (N, C, 30)
  pred    the level branch's forecast, pooled to one value per cycle, (N, C, 30)
  demand  the true future cycle means minus the last observed value, (N, C, 30)
  value   the same window read point-wise, one sample every 24 steps, minus the
          last value, (N, C, 30): the naive alternative to the cycle means
in the RevIN-normalized space the branch actually operates in, plus the branch's
three horizon directions pooled to the same 30 bins, (30, rank).

    python3 scripts/level_pca_extract.py --dataset Traffic --horizon 720 --out research_runs/level_pca_v1
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHANNELS = 16


def winner_job(dataset, horizon):
    """Best b0 cell with the fixed-profile head: the paper's level decoder. Pooling 24."""
    from scripts.b0_search import DEFAULT_SCRATCH as B0, load_results
    rows = [r for r in load_results(B0) if r['dataset'] == dataset and r['horizon'] == horizon
            and r['head'] == 'S2']
    if not rows:
        raise SystemExit(f'no S2 cell for {dataset}-{horizon}')
    best = min(rows, key=lambda r: r['mse'])
    hits = glob.glob(str(B0 / '*' / 'jobs' / f"{best['run']}.json"))
    if len(hits) != 1:
        raise SystemExit(f'job for {best["run"]}: {hits}')
    return json.loads(Path(hits[0]).read_text()), best


def branch_basis(model, pooled_len=30):
    """Horizon directions the level decoder can write, pooled to cycle bins."""
    head = model.lowrank_interaction_branch.lowrank_head
    rank = head.rank
    dev = next(head.parameters()).device
    lat = torch.zeros(1, rank, rank, device=dev)
    for i in range(rank):
        lat[0, i, i] = 1.0                                # channel i = one unit latent
    last = torch.zeros(1, 1, rank, device=dev)
    with torch.inference_mode():
        out = head.decode_latent(lat, last)[0]            # (H, rank)
    basis = out.permute(1, 0).unsqueeze(0)                # (1, H, rank)
    pooled = F.adaptive_avg_pool1d(basis, head.pooled_len)[0]
    return pooled.cpu().numpy(), head.pooled_len, rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--horizon', type=int, default=720)
    parser.add_argument('--out', type=Path, default=ROOT / 'research_runs/level_pca_v1')
    parser.add_argument('--max-batches', type=int, default=10**9)
    args = parser.parse_args()

    from scripts.vldbj_m1.proto import build_model
    from src.dataset.data_factory import data_provider

    job, best = winner_job(args.dataset, args.horizon)
    model, train_args, cfg, _ = build_model(job)
    record = json.loads((Path(job['output']) / 'result.json').read_text())
    state = torch.load(record['checkpoint'], map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(state)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    basis, pooled_len, rank = branch_basis(model, 30)
    enc_in = model.enc_in
    rng = np.random.default_rng(0)
    chans = np.arange(enc_in) if enc_in <= CHANNELS else np.sort(rng.choice(enc_in, CHANNELS, replace=False))

    zeta, pred, demand, value, target = [], [], [], [], []
    captured = {}

    def hook(module, inputs, output):
        x = inputs[0]                                      # (B, L, C) normalized input
        o = output[0] if isinstance(output, tuple) else output
        captured['x'], captured['o'] = x.detach(), o.detach()

    handle = model.lowrank_interaction_branch.register_forward_hook(hook)
    _, test_loader = data_provider(train_args.dataset_args, 'test')
    pred_len = model.pred_len
    with torch.inference_mode():
        for i, batch in enumerate(test_loader):
            if i >= args.max_batches:
                break
            bx, by, bxm, bym = [v.float().to(device) for v in batch]
            model(bx, bxm, model._build_decoder_input(by), bym)
            x, o = captured['x'], captured['o']            # (B,L,C), (B,H,C)
            # x is already RevIN-normalized; take the statistics from the raw input
            mu = bx.mean(1, keepdim=True)
            sd = (bx.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
            y = (by[:, -pred_len:] - mu) / sd              # normalized future
            last = x[:, -1:, :]                            # normalized last value
            z = F.adaptive_avg_pool1d(((x - last).permute(0, 2, 1)), pooled_len)
            p = F.adaptive_avg_pool1d(o.permute(0, 2, 1), pooled_len)
            last_norm = (bx[:, -1:, :] - mu) / sd
            d = F.adaptive_avg_pool1d(y.permute(0, 2, 1), pooled_len) - last_norm.permute(0, 2, 1)
            v = ((x - last)[:, ::pooled_len] - 0.0).permute(0, 2, 1)
            zeta.append(z[:, chans].cpu().numpy())
            value.append(v[:, chans].cpu().numpy())
            pred.append(p[:, chans].cpu().numpy())
            demand.append(d[:, chans].cpu().numpy())
            target.append(y[:, :, chans].cpu().numpy())
    handle.remove()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    name = f'{args.dataset}_H{args.horizon}'
    np.savez_compressed(out / f'{name}.npz',
                        zeta=np.concatenate(zeta), pred=np.concatenate(pred),
                        demand=np.concatenate(demand), value=np.concatenate(value),
                        channels=chans, basis=basis,
                        pooled_len=pooled_len, rank=rank,
                        dataset=args.dataset, horizon=args.horizon,
                        period=int(cfg.get('period', 24)), run=best['run'],
                        test_mse=best['mse'], test_mae=best['mae'])
    print(f'{name}: zeta {np.concatenate(zeta).shape} basis {basis.shape} '
          f'period {cfg.get("period", 24)} run {best["run"]}')


if __name__ == '__main__':
    main()
