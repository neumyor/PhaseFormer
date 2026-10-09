"""Fit-only linear summaries and capacity-matched residual probes."""
import numpy as np

FAMILIES = ('t_own2', 't_own4', 't_pmean', 't_pslot', 't_mismatch_mean',
            't_mismatch_slot', 'p_own2', 'p_own4', 'p_trend', 'p_mismatch_trend')
CROSS = ('t_pmean', 't_pslot', 'p_trend')
DEGREES = (1., 2., 3., 4.)
CONTROLS = {'t_pmean': ('t_own4', 't_mismatch_mean'),
            't_pslot': ('t_own4', 't_mismatch_slot'),
            'p_trend': ('p_own4', 'p_mismatch_trend')}


def output_basis(horizon, period=24):
    if horizon % period or horizon < 2 * period:
        raise ValueError('Four modes require at least two complete cycles')
    cycle = np.repeat(np.arange(horizon // period, dtype=float), period)
    phase = np.arange(horizon) % period
    raw = np.stack([np.ones(horizon), cycle-cycle.mean(),
                    np.sin(2*np.pi*phase/period), np.cos(2*np.pi*phase/period)], axis=1)
    q, _ = np.linalg.qr(raw)
    np.testing.assert_allclose(q.T @ q, np.eye(4), atol=1e-12)
    return q


def fit_pca(x, width=4):
    x = np.asarray(x, dtype=np.float64)
    center, scale = x.mean(0), np.maximum(x.std(0), 1e-8)
    _, _, vh = np.linalg.svd((x-center)/scale, full_matrices=False)
    if len(vh) < width:
        raise ValueError('Insufficient fit data or representation width')
    return {'center': center.tolist(), 'scale': scale.tolist(), 'axes': vh[:width].tolist()}


def project(x, transform):
    return ((x-np.asarray(transform['center']))/np.asarray(transform['scale'])) @ np.asarray(transform['axes']).T


def fit_transforms(raw):
    return {k: fit_pca(raw[k], 4) for k in ('t', 'mean', 'slot')}


def mismatch_indices(source_ids, target_ids, channels, support, same_role=False):
    """Use the same variable at a deterministic, support-disjoint fit time."""
    source_ids = np.asarray(source_ids)
    if same_role:
        positions = (np.arange(len(target_ids))+len(source_ids)//2) % len(source_ids)
    else:
        positions = np.asarray(target_ids) % len(source_ids)
    if np.any(np.abs(source_ids[positions]-target_ids) < support):
        raise ValueError('Mismatched sources overlap target history/forecast support')
    return (positions[:, None]*channels+np.arange(channels)[None]).reshape(-1)


def design(raw, transforms, family, mismatch):
    own_t, own_p = project(raw['t'], transforms['t']), project(raw['slot'], transforms['slot'])
    mean = project(raw['mean'], transforms['mean'])
    base = np.c_[raw['anchor'], raw['trend_modes' if family.startswith('t_') else 'phase_modes']]
    own = own_t if family.startswith('t_') else own_p
    if family.endswith('own2'):
        return np.c_[base, own[:, :2]]
    if family.endswith('own4'):
        return np.c_[base, own[:, :4]]
    source = own_t if family.startswith('p_') else (mean if family.endswith('mean') else own_p)
    if 'mismatch' in family:
        source = mismatch['t' if family.startswith('p_') else ('mean' if family.endswith('mean') else 'slot')]
    return np.c_[base, own[:, :2], source[:, :2]]


def fit_ridge(x, y, degrees):
    center, scale = x.mean(0), np.maximum(x.std(0), 1e-8)
    a = (x-center)/scale
    yc = y.mean(0)
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    eigen = s*s
    rank = int(np.sum(s > s.max()*1e-10))
    if degrees >= rank:
        raise ValueError(f'Effective degrees {degrees} requires rank > {degrees}; rank={rank}')
    lo, hi = 0., max(float(eigen.max()), 1.)
    while (eigen/(eigen+hi)).sum() > degrees:
        hi *= 2
    for _ in range(80):
        mid = (lo+hi)/2
        if (eigen/(eigen+mid)).sum() > degrees:
            lo = mid
        else:
            hi = mid
    penalty = (lo+hi)/2
    beta = vh.T @ ((s/(eigen+penalty))[:, None]*(u.T @ (y-yc)))
    return {'center': center.tolist(), 'scale': scale.tolist(), 'beta': beta.tolist(),
            'target_center': yc.tolist(), 'penalty': penalty, 'slope_edf': float((eigen/(eigen+penalty)).sum()),
            'requested_edf': degrees, 'rank': rank, 'features': x.shape[1],
            'readout_parameters': (x.shape[1]+1)*y.shape[1]}


def predict(x, fitted):
    if fitted is None:
        return np.zeros((len(x), 4))
    return ((x-np.asarray(fitted['center']))/np.asarray(fitted['scale'])) @ np.asarray(fitted['beta']) + np.asarray(fitted['target_center'])


def sample_metrics(error):
    return {'mse': np.square(error).mean(-1), 'mae': np.abs(error).mean(-1)}


def summarize(metrics):
    mae = metrics['mae']
    return {'mse': float(metrics['mse'].mean()), 'mae': float(mae.mean()),
            'mae_p95': float(np.quantile(mae, .95)),
            'mae_top1': float(np.sort(mae)[-max(1, int(np.ceil(len(mae)*.01))):].mean())}


def rubric(results, selected):
    rows = results[results.config_id == selected]
    if selected == 'b0':
        return {'pass': False, 'reason': 'Calibration selected no correction'}
    checks = {'average_mse': bool(rows.relative_mse.mean() <= -1),
              'average_mae': bool(rows.relative_mae.mean() <= 0),
              'setting_wins': bool((rows.relative_mse < 0).sum() >= 8),
              'task_wins': bool((rows.groupby(['dataset', 'horizon']).relative_mse.mean() < 0).sum() >= 4),
              'worst_mse': bool(rows.relative_mse.max() <= 2),
              'tail': bool((rows.relative_p95 <= 2).all() and (rows.relative_top1 <= 2).all())}
    for control in (*CONTROLS[selected], 'receiver_matched', 'mismatch_matched'):
        other = results[results.config_id == control].set_index('setting')
        ratio = rows.set_index('setting').mse/other.mse-1
        checks['beats_'+control] = bool(ratio.mean() < 0 and (ratio < 0).sum() >= 8)
    return {'pass': all(checks.values()), 'checks': checks,
            'relative_mse': float(rows.relative_mse.mean()), 'relative_mae': float(rows.relative_mae.mean())}
