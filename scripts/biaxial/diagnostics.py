"""E5 and E6 analyses on saved checkpoints.

    python -m scripts.biaxial.diagnostics e6 runs/RESULTS.jsonl GROUP [GROUP ...] [--settings ETTh1-96 ...]
    python -m scripts.biaxial.diagnostics e5 runs/RESULTS.jsonl M0_GROUP BACKBONE_GROUP [--settings ...]

e6  per model: test MSE split into the cycle-level part and the within-cycle part, and the MSE in
    terciles of (a) the true level change and (b) the periodic strength of the input window.
e5  (1) how well the backbone's pooled latent predicts the original mean phase token (ridge, R^2 on test);
    (2) normalized entropy of the temporal and phase routers' gather attention;
    (3) the largest cosine between the rows of W_t (columns of W_p) and the uniform vector.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.biaxial.model import PhaseFormerPlus, RouterAttn  # noqa: E402
from scripts.biaxial.train import LOOKBACK, build_cfg, load_splits, windows  # noqa: E402


def read_rows(path, groups):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {g: {f"{r['dataset']}-{r['horizon']}": r for r in rows if r['group'] == g} for g in groups}


def load_model(row, channels, ckpt_dir, device):
    cfg = build_cfg({'horizon': row['horizon'], 'period': row['period'], 'model': row['model']}, channels)
    model = PhaseFormerPlus(cfg)
    model.load_state_dict(torch.load(ckpt_dir / (row['id'].replace('/', '__') + '.pt'), map_location='cpu'))
    return model.to(device).eval()


@torch.inference_mode()
def predict(model, data, horizon, batch=256):
    xs, ys, ps = [], [], []
    n = data.shape[0] - LOOKBACK - horizon + 1
    for start in range(0, n, batch):
        x, y = windows(data, torch.arange(start, min(start + batch, n), device=data.device), horizon)
        xs.append(x), ys.append(y), ps.append(model(x))
    return torch.cat(xs), torch.cat(ys), torch.cat(ps)


def tercile_mse(sq, key):
    """Mean of ``sq`` in the lower, middle and upper tercile of ``key`` (same shape)."""
    order = key.flatten().argsort()
    parts = order.chunk(3)
    flat = sq.flatten()
    return [flat[p].mean().item() for p in parts]


def e6(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    table = read_rows(args.results, args.groups)
    ckpt_dir = Path(args.results).parent / 'ckpt'
    print('| setting | model | MSE | level MSE | shape MSE | by level change (low/mid/high) | by periodic strength (low/mid/high) |')
    print('|---|---|---:|---:|---:|---|---|')
    settings = args.settings or sorted(set.intersection(*(set(t) for t in table.values())))
    for setting in settings:
        first = table[args.groups[0]][setting]
        splits, _ = load_splits(first['dataset'], first['horizon'])
        test = splits['test'].to(device)
        for group in args.groups:
            row = table[group][setting]
            model = load_model(row, test.shape[1], ckpt_dir, device)
            x, y, pred = predict(model, test, row['horizon'])
            P = row['period']
            err = (pred - y).square()
            B, H, C = y.shape
            diff = pred - y
            level_err = diff.view(B, H // P, P, C).mean(2).square().mean().item()
            level_change = (y.mean(1) - x[:, -P:].mean(1)).abs()
            cyc = x.view(B, LOOKBACK // P, P, C)
            periodic = (cyc - cyc.mean(2, keepdim=True)).square().mean((1, 2)) / (x.var(1) + 1e-8)
            sq = err.mean(1)
            lo = tercile_mse(sq, level_change)
            ps = tercile_mse(sq, periodic)
            print(f"| {setting} | {group} | {err.mean().item():.4f} | {level_err:.4f} | "
                  f"{err.mean().item() - level_err:.4f} | {lo[0]:.4f} / {lo[1]:.4f} / {lo[2]:.4f} | "
                  f"{ps[0]:.4f} / {ps[1]:.4f} / {ps[2]:.4f} |")


def r2_ridge(feat_fit, target_fit, feat_test, target_test, lam=1e-3):
    def design(f):
        return torch.cat([f, torch.ones(f.shape[0], 1, device=f.device)], 1)
    a = design(feat_fit)
    w = torch.linalg.solve(a.T @ a + lam * torch.eye(a.shape[1], device=a.device), a.T @ target_fit)
    pred = design(feat_test) @ w
    ss_res = (target_test - pred).square().sum(0)
    ss_tot = (target_test - target_test.mean(0)).square().sum(0)
    return (1 - ss_res / ss_tot).mean().item()


def entropy_by_axis(model, x):
    stats = {}
    for name, module in model.phase.named_modules():
        if isinstance(module, RouterAttn):
            module.keep_attn = True
    model(x)
    for name, module in model.phase.named_modules():
        if isinstance(module, RouterAttn):
            w = module.last_weights[0]                                   # gather: (N, h, M, T)
            ent = -(w * (w + 1e-12).log()).sum(-1).mean().item() / math.log(w.shape[-1])
            stats[name] = ent
            module.keep_attn = False
    return stats


def e5(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    table = read_rows(args.results, [args.m0, args.backbone])
    ckpt_dir = Path(args.results).parent / 'ckpt'
    settings = args.settings or sorted(set(table[args.m0]) & set(table[args.backbone]))
    for setting in settings:
        r0, r1 = table[args.m0][setting], table[args.backbone][setting]
        splits, _ = load_splits(r0['dataset'], r0['horizon'])
        m0 = load_model(r0, splits['test'].shape[1], ckpt_dir, device)
        bb = load_model(r1, splits['test'].shape[1], ckpt_dir, device)
        feats = {}
        for flag in ('val', 'test'):
            x, _, _ = predict(m0, splits[flag].to(device), r0['horizon'])
            x = x[:2048]
            with torch.inference_mode():
                xn, X, _, _ = m0.fold(x)
                _, zbar = m0.phase(X)
                bb(x)
                feats[flag] = (bb.phase.pooled, zbar)
        r2 = r2_ridge(feats['val'][0], feats['val'][1], feats['test'][0], feats['test'][1])
        with torch.inference_mode():
            ent = entropy_by_axis(bb, x)
        inp = bb.phase.inp
        uniform_t = torch.ones(inp.Wt.shape[-1], device=device)
        cos_t = torch.nn.functional.cosine_similarity(inp.Wt[0], uniform_t[None], dim=1).abs().max().item()
        wp = inp.Wp[0]
        cos_p = torch.nn.functional.cosine_similarity(wp.T, torch.ones(1, wp.shape[0], device=device), dim=1).abs().max().item()
        print(f'{setting}: R2(mean phase token from backbone latent)={r2:.3f}; gather-attention entropy {ent}; '
              f'max |cos(W_t row, uniform)|={cos_t:.3f}; max |cos(W_p column, uniform)|={cos_p:.3f}')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('e6')
    a.add_argument('results')
    a.add_argument('groups', nargs='+')
    b = sub.add_parser('e5')
    b.add_argument('results')
    b.add_argument('m0')
    b.add_argument('backbone')
    for p in (a, b):
        p.add_argument('--settings', nargs='*')
        p.add_argument('--gpu', type=int, default=4)
    args = parser.parse_args()
    {'e6': e6, 'e5': e5}[args.cmd](args)


if __name__ == '__main__':
    main()
