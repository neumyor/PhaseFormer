"""Checkpoint analyses for the VLDBJ extension (seed-2021 main runs).

Measures on trained checkpoints, per theory Section 5 and Appendix B:
  * level envelope: c0, C1, C_phi read from the weights (Theorem 3), the level
    demand Lambda per window, the share of windows outside the interval, and
    the share of test MSE accounted for by the level error floor;
  * amplitude envelope: B_phi (Theorem 5a), the amplitude demand Omega per
    window, and the share of windows whose demand exceeds the phase budget;
  * level hand-over: the trained gates g_l and g_s, and the branch level;
  * modulation activation (m1 only): the modulation vector q per window and
    its correlation with the level demand Lambda.

Each job writes ``analysis.json`` and ``analysis_windows.npz`` into the run
directory.  Window-level joins for gain stratification are done offline.

``--selected-variants`` analyzes the seed-2021 tune runs of the variants chosen
by ``run.py --stage tune-select``: those run directories keep their best
checkpoint in ``checkpoints/best.ckpt`` and their job spec (source config,
variant) in the stage's ``jobs/`` directory, from which the config is rebuilt.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys

import numpy as np

# VLDBJ_REPO_ROOT lets the analysis run from outside the frozen bundle while
# the tune stage is in flight (a new file under scripts/vldbj_m1/ would change
# the provenance hash and make pending tune workers refuse to start).
ROOT = Path(os.environ.get('VLDBJ_REPO_ROOT', Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT))
from scripts.run_lowrank_interaction_stage2 import digest, save_json  # noqa: E402
from scripts.vldbj_m1.model import legendre_profiles  # noqa: E402

ANALYSIS_SEED = 2021
WINDOW_KEYS = ('lam', 'levF', 'dist', 'omega', 'sigma', 'mse', 'branch_lev', 'budget_excess')


def commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def pi_basis(pred_len, period, device, dtype):
    """Orthonormal basis of U = J_c span{tilt, curvature} on the horizon."""
    import torch
    k_primes, p = pred_len // period, period
    prof = legendre_profiles(pred_len).to(device=device, dtype=dtype)  # (H, 3)
    basis = []
    for col in (1, 2):
        v = prof[:, col].reshape(k_primes, p)
        v = (v - v.mean(1, keepdim=True)).reshape(-1)
        for w in basis:
            v = v - (v @ w) * w
        basis.append(v / (v @ v).sqrt())
    return torch.stack(basis, 1)  # (H, 2)


def analysis_job(path):
    import torch
    from src.dataset.data_factory import data_provider
    from scripts.vldbj_m1.model import construct

    job = json.loads(Path(path).read_text())
    run_dir = Path(job['run_dir'])
    with (run_dir / 'analysis.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run_dir / 'analysis.json').exists():
            return
        variant = None
        if (run_dir / 'result.json').exists():
            result = json.loads((run_dir / 'result.json').read_text())
            cfg = json.loads((run_dir / 'config.json').read_text())
            if digest(result['checkpoint']) != result['checkpoint_sha256']:
                raise ValueError('Checkpoint changed after training')
            checkpoint = result['checkpoint']
        else:
            # Tune/confirm run of a tuning variant: rebuild the config from the
            # stage's job spec and read the best checkpoint Lightning kept.
            result = json.loads((run_dir / 'tune_result.json').read_text())
            stage_dir = run_dir.parent.parent
            spec = json.loads((stage_dir / 'jobs' / f'{run_dir.name}.json').read_text())
            from scripts.vldbj_m1.model import prepare_config
            cfg = prepare_config(spec['source']['config'], spec['arm'], spec['head'],
                                 spec['shared_gate'], spec.get('variant'))
            checkpoint = str(run_dir / 'checkpoints' / 'best.ckpt')
            variant = spec.get('variant')
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('medium')
        model, args = construct({**cfg, 'checkpoint': ''})
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
        model.load_state_dict(state, strict=True)
        device = torch.device('cuda:0')
        model.to(device).eval()

        period, pred_len = model.period_len, model.pred_len
        k_primes, d = pred_len // period, model.latent_dim
        arm, is_m1 = job['arm'], job['arm'] == 'm1'

        # --- envelope constants read from the weights (Theorems 3 and 5a) ---
        ln = model.routing_layers[-1].interact.norm2
        gamma, beta = ln.weight.detach().float(), ln.bias.detach().float()
        phi_w = model.predictor.decoder.weight.detach().float()   # (K', d)
        phi_b = model.predictor.decoder.bias.detach().float()     # (K',)
        A = phi_w * gamma                                         # (K', d)
        o = phi_w @ beta + phi_b                                 # (K',)
        c0 = o.mean().item()
        Jd = torch.eye(d, device=device) - 1.0 / d
        C1 = (d ** 0.5 / k_primes) * (Jd @ (A.T @ torch.ones(k_primes, device=device))).norm().item()
        C_phi = abs(c0) + C1
        B_phi = (d / k_primes) ** 0.5 * torch.linalg.matrix_norm(A @ Jd, ord=2).item()
        gate = model.lowrank_interaction_gate.detach().float().reshape(-1).cpu()
        delta = model.lowrank_dynamic_fusion.shape_bias_delta.detach().float().reshape(-1).cpu()
        channels = model.enc_in
        g_level = torch.sigmoid(gate).expand(channels).clone()    # (C,)
        g_shape = torch.sigmoid(gate + delta).expand(channels).clone()
        budget = (1.0 - g_shape) * B_phi                          # (C,)
        U = pi_basis(pred_len, period, device, torch.float32)

        branch = model.lowrank_interaction_branch
        captured = {}

        def branch_hook(module, inputs, output):
            captured['branch'] = output[0].detach()               # (B, H, C) normalized

        handle_branch = model.lowrank_interaction_branch.register_forward_hook(branch_hook)
        _, loader = data_provider(args.dataset_args, 'test')

        def run_pass(intervention, capture_q):
            store = {k: [] for k in WINDOW_KEYS}
            qhook = thook = None
            if capture_q:
                qhook = branch.communication.output.register_forward_hook(
                    lambda m, i, out: captured.__setitem__('q', out.detach()))
                thook = branch.communication.source.register_forward_pre_hook(
                    lambda m, inputs: captured.__setitem__('t', inputs[0].detach()))
                captured.setdefault('q_list', [])
                captured.setdefault('t_list', [])
            with torch.inference_mode():
                for batch in loader:
                    bx, by, bxm, bym = [v.float().to(device) for v in batch]
                    if is_m1:
                        branch.intervention = intervention
                    _, (mu, sigma) = model.revin.normalize(bx)     # (B,1,C) each
                    y_hat, _, phase_steps = model(bx, bxm, model._build_decoder_input(by), bym)
                    y_fut = by[:, -pred_len:, :]                  # (B, H, C)
                    y_norm = (y_fut - mu) / sigma
                    lam = y_norm.mean(1)                          # (B, C)
                    # phase_steps (B, C, P, K') -> time order (k, p).
                    F = phase_steps.permute(0, 1, 3, 2).reshape(y_fut.shape)
                    levF = F.mean(1)
                    dist = ((lam - c0).abs() - C1).clamp_min(0)
                    yc = y_norm.permute(0, 2, 1).reshape(-1, k_primes, period)
                    yc = yc - yc.mean(2, keepdim=True)            # J_c
                    flat = yc.reshape(yc.shape[0], -1)            # (B*C, H)
                    pi_y = flat - (flat @ U) @ U.T
                    omega = (pi_y.norm(dim=1) / pred_len ** 0.5).reshape(lam.shape)
                    mse = (y_hat[:, -pred_len:, :] - y_fut).square().mean(1)
                    store['lam'].append(lam.cpu())
                    store['levF'].append(levF.cpu())
                    store['dist'].append(dist.cpu())
                    store['omega'].append(omega.cpu())
                    store['sigma'].append(sigma.squeeze(1).cpu())
                    store['mse'].append(mse.cpu())
                    store['branch_lev'].append(captured['branch'].mean(1).cpu())
                    store['budget_excess'].append((omega.cpu() - budget).clamp_min(0))
                    if capture_q:
                        captured['q_list'].append(captured['q'].cpu())
                        captured['t_list'].append(captured['t'].cpu())
            if qhook:
                qhook.remove()
            if thook:
                thook.remove()
            return {k: torch.cat(v) for k, v in store.items()}

        base = run_pass('off' if is_m1 else 'native', capture_q=False)
        if is_m1:
            native = run_pass('native', capture_q=True)
            branch.intervention = 'native'
        handle_branch.remove()

        lam, dist, mse, sigma = base['lam'], base['dist'], base['mse'], base['sigma']
        levF, omega = base['levF'], base['omega']
        lam_flat = lam.flatten().numpy()
        record = {'name': result['name'], 'arm': arm, 'seed': ANALYSIS_SEED, 'variant': variant,
                  'code_commit': commit(),
                  'constants': {'c0': c0, 'C1': C1, 'C_phi': C_phi, 'B_phi': B_phi,
                                'd': d, 'K_prime': k_primes},
                  'gates': {'g_level_mean': g_level.mean().item(),
                            'g_level_min': g_level.min().item(),
                            'g_level_max': g_level.max().item(),
                            'g_shape_mean': g_shape.mean().item(),
                            'shared': bool(cfg['vldbj_m1']['shared_gate'])},
                  'level': {
                      'mean_abs_lambda': lam.abs().mean().item(),
                      'share_outside_interval': (dist > 0).float().mean().item(),
                      'mean_dist': dist.mean().item(),
                      'floor_share_of_mse': float((sigma.square() * dist.square()).sum()
                                                  / mse.sum()),
                      'mean_abs_levF_miss': (levF - lam).abs().mean().item()},
                  'amplitude': {
                      'share_demand_over_budget': (base['budget_excess'] > 0).float().mean().item(),
                      'mean_omega': omega.mean().item(),
                      'mean_budget': budget.mean().item()},
                  'handover': {
                      'mean_branch_level': base['branch_lev'].mean().item(),
                      'corr_branch_level_lambda': float(np.corrcoef(
                          base['branch_lev'].flatten().numpy(), lam_flat)[0, 1])}}

        npz = {'lambda': lam.numpy(), 'levF': levF.numpy(), 'dist': dist.numpy(),
               'omega': omega.numpy(), 'sigma': sigma.numpy(), 'mse': mse.numpy(),
               'branch_lev': base['branch_lev'].numpy(),
               'g_level': g_level.numpy(), 'g_shape': g_shape.numpy()}

        if is_m1:
            q = torch.cat(captured['q_list']).reshape(-1, channels, d)   # (N, C, d)
            t = torch.cat(captured['t_list']).reshape(-1, channels, 3)  # (N, C, 3)
            q_scalar = q.mean(-1)                                        # (N, C)
            qs_flat = q_scalar.flatten().numpy()
            slope, intercept = np.polyfit(lam_flat, qs_flat, 1)
            residual = qs_flat - (slope * lam_flat + intercept)
            alam = lam.abs().flatten()
            top = alam > alam.quantile(0.8)
            bottom = alam < alam.quantile(0.2)
            record['modulation'] = {
                'corr_q_scalar_lambda': float(np.corrcoef(qs_flat, lam_flat)[0, 1]),
                'slope_q_on_lambda': float(slope),
                'r2_q_on_lambda': float(1.0 - residual.var() / qs_flat.var()),
                'mean_abs_q_top_lambda_quintile': float(q_scalar.flatten()[top].abs().mean()),
                'mean_abs_q_bottom_lambda_quintile': float(q_scalar.flatten()[bottom].abs().mean())}
            npz.update(q_scalar=q_scalar.numpy(), t=t.numpy(),
                       q_weights=branch.communication.output.weight.detach().cpu().numpy(),
                       v1=branch.communication.source.weight.detach().cpu().numpy(),
                       mse_native=native['mse'].numpy(),
                       branch_lev_native=native['branch_lev'].numpy())
        np.savez_compressed(run_dir / 'analysis_windows.npz', **npz)
        if is_m1:
            # Invariance check last: it promotes the model to double precision.
            from scripts.vldbj_m1.run import cycle_mean_check
            record['cycle_mean_check'] = cycle_mean_check(model, loader, device)
        save_json(run_dir / 'analysis.json', record)


def plan_jobs(runs_dir):
    jobs = []
    for run in sorted(Path(runs_dir).glob('*/test_result.json')):
        name = run.parent.name
        if f'_S{ANALYSIS_SEED}__' not in name:
            continue
        m = re.match(r'(\w+)_H(\d+)_S\d+__(b0|m1)__S\d+__(\w+)$', name)
        if not m:
            raise ValueError(name)
        jobs.append({'run_dir': str(run.parent), 'arm': m.group(3),
                     'dataset': m.group(1), 'horizon': int(m.group(2))})
    out = []
    for job in jobs:
        path = Path(runs_dir).parent / 'analysis' / 'jobs' / f"{Path(job['run_dir']).name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path, job)
        out.append(path)
    return out


def plan_selected_variants(scratch):
    """Analyze the seed-2021 tune runs of the variants chosen by tune-select."""
    selection = json.loads((scratch / 'selection.json').read_text())
    chosen = json.loads((scratch / 'tune_selection.json').read_text())['settings']
    head = selection['head']
    jobs = []
    for name, record in sorted(chosen.items()):
        if record['chosen'] is None:
            continue
        dataset, horizon = name.split('_H')
        gate = 'shared' if selection['shared_gate'][dataset] else 'perC'
        run_dir = scratch / 'tune' / 'runs' / f'{name}_S{ANALYSIS_SEED}__m1__{head}__{gate}__t{record["chosen"]}'
        jobs.append({'run_dir': str(run_dir), 'arm': 'm1', 'dataset': dataset,
                     'horizon': int(horizon)})
    out = []
    for job in jobs:
        path = scratch / 'analysis' / 'variant_jobs' / f"{Path(job['run_dir']).name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path, job)
        out.append(path)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs-dir', type=Path,
                        help='main/runs directory of the vldbj_m1 scratch')
    parser.add_argument('--selected-variants', type=Path,
                        help='vldbj_m1 scratch dir with tune_selection.json; analyzes '
                             'the seed-2021 tune runs of every chosen variant')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    if args.job:
        return analysis_job(args.job)
    if args.selected_variants:
        jobs = plan_selected_variants(args.selected_variants)
    else:
        jobs = plan_jobs(args.runs_dir)
    print(json.dumps({'analysis_jobs': len(jobs), 'code': commit()}), flush=True)
    tasks = queue.Queue()
    for path in jobs:
        tasks.put(path)

    def bucket(gpu):
        while True:
            try:
                path = tasks.get_nowait()
            except queue.Empty:
                return
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            code = subprocess.call([sys.executable, str(Path(__file__).resolve()), '--job', str(path)],
                                   cwd=ROOT, env=env,
                                   stdout=open(Path(json.loads(path.read_text())['run_dir']) / 'analysis.log', 'a'),
                                   stderr=subprocess.STDOUT)
            if code:
                raise RuntimeError(f'analysis job failed: {path}')

    gpus = args.gpus.split(',')
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        for future in [pool.submit(bucket, g) for g in gpus]:
            future.result()


if __name__ == '__main__':
    main()
