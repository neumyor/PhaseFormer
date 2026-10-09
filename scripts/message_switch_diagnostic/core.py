"""Fixed small selectors and support-disjoint temporal roles."""
import numpy as np
from scipy.optimize import minimize

FEATURES = ['recent_shift', 'last_innovation', 'period_difference', 'period_correlation',
            'message_rms', 'branch_level_difference', 'write_cosine', 'cancellation']


def temporal_roles(train_length, val_length, lookback, horizon, period=24):
    cut = int(train_length * .6)
    support = lookback + horizon
    roles = {'fit': np.arange(0, cut-support+1, period),
             'cal': np.arange(cut, train_length-support+1, period),
             'eval': np.arange(lookback, val_length-support+1)}
    if any(len(v) == 0 for v in roles.values()):
        raise ValueError('Insufficient complete temporal supports')
    assert roles['fit'][-1] + support <= roles['cal'][0]
    # Val data starts train_length-lookback; eval first history starts at train_length.
    assert roles['cal'][-1] + support <= train_length
    assert train_length-lookback+roles['eval'][0] >= train_length
    return roles, cut


def fit_selector(frame, columns):
    x = frame[columns].to_numpy(np.float64)
    lo, hi = np.quantile(x, [.005, .995], axis=0)
    x = np.clip(x, lo, hi)
    center = np.median(x, axis=0)
    q = np.quantile(x, [.25, .75], axis=0)
    scale = np.maximum(q[1]-q[0], 1e-5)
    x = np.c_[np.ones(len(x)), (x-center)/scale]
    difference = frame.off_mse.to_numpy()-frame.candidate_mse.to_numpy()
    target = (difference < 0).astype(float)
    weights = np.zeros(len(frame))
    for _, indices in frame.groupby('setting', sort=False).indices.items():
        baseline = max(frame.iloc[indices].baseline_mse.mean(), 1e-8)
        weights[indices] = np.abs(difference[indices])/baseline/len(indices)
    weights /= weights.sum()
    def loss(beta):
        logits = x @ beta
        value = np.dot(weights, np.logaddexp(0, logits)-target*logits) + .005*np.dot(beta[1:], beta[1:])
        prob = 1/(1+np.exp(-np.clip(logits, -60, 60)))
        gradient = x.T @ (weights*(prob-target))
        gradient[1:] += .01*beta[1:]
        return value, gradient
    result = minimize(loss, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(result.message)
    return {'columns': columns, 'lo': lo.tolist(), 'hi': hi.tolist(),
            'center': center.tolist(), 'scale': scale.tolist(), 'beta': result.x.tolist(),
            'iterations': int(result.nit), 'objective': float(result.fun), 'l2': .01}


def score(frame, selector):
    x = frame[selector['columns']].to_numpy(np.float64)
    x = (np.clip(x, selector['lo'], selector['hi'])-selector['center'])/selector['scale']
    logits = selector['beta'][0] + x @ np.array(selector['beta'][1:])
    return 1/(1+np.exp(-np.clip(logits, -60, 60)))


def summary(frame, disable):
    mse = np.where(disable, frame.off_mse, frame.candidate_mse)
    mae = np.where(disable, frame.off_mae, frame.candidate_mae)
    benefit = np.maximum(frame.off_mse-frame.candidate_mse, 0).to_numpy()
    harmful = np.maximum(frame.candidate_mse-frame.off_mse, 0).to_numpy()
    positive = np.flatnonzero(benefit > 0)
    count = max(1, int(np.ceil(len(positive)*.01)))
    extreme = positive[np.argsort(-benefit[positive], kind='stable')[:count]]
    oracle = np.minimum(frame.off_mse, frame.candidate_mse).mean()
    capacity = frame.candidate_mse.mean()-oracle
    return {'mse': float(mse.mean()), 'mae': float(mae.mean()),
            'mae_p95': float(np.quantile(mae, .95)),
            'mae_top1': float(np.sort(mae)[-max(1, int(np.ceil(len(mae)*.01))):].mean()),
            'disabled_fraction': float(np.mean(disable)),
            'lost_benefit_fraction': float(benefit[disable].sum()/max(benefit.sum(), 1e-20)),
            'lost_extreme_benefit_fraction': float(benefit[extreme][disable[extreme]].sum()/max(benefit[extreme].sum(), 1e-20)),
            'lost_benefit_mse': float(benefit[disable].sum()/len(frame)),
            'missed_harm_mse': float(harmful[~disable].sum()/len(frame)),
            'oracle_mse': float(oracle), 'oracle_recovery': float((frame.candidate_mse.mean()-mse.mean())/max(capacity, 1e-20)),
            'native_mse': float(frame.candidate_mse.mean()), 'native_mae': float(frame.candidate_mae.mean()),
            'b0_mse': float(frame.baseline_mse.mean()), 'b0_mae': float(frame.baseline_mae.mean())}


def calibration_choice(frame, scores):
    choices = [('all_on', float('inf')), ('all_off', -float('inf'))]
    choices += [(f'q{q:02d}', float(np.quantile(scores, q/100))) for q in range(5, 100, 5)]
    attempts = []
    for name, threshold in choices:
        records = []
        for setting, indices in frame.groupby('setting').indices.items():
            part = frame.iloc[indices]
            record = summary(part, scores[indices] >= threshold)
            native = summary(part, np.zeros(len(part), bool))
            record.update(setting=setting, relative_mse=(record['mse']/native['mse']-1)*100,
                          relative_mae=(record['mae']/native['mae']-1)*100,
                          tail_pass=record['mae_p95']<=native['mae_p95']*1.02
                          and record['mae_top1']<=native['mae_top1']*1.02)
            records.append(record)
        valid = (np.mean([r['relative_mae'] for r in records]) <= 1e-10
                 and all(r['tail_pass'] and r['lost_benefit_fraction']<=.1 for r in records))
        attempts.append({'name': name, 'threshold': threshold, 'valid': bool(valid),
                         'relative_mse': float(np.mean([r['relative_mse'] for r in records])), 'records': records})
    selected = min((a for a in attempts if a['valid']), key=lambda a:(a['relative_mse'], a['name']))
    return selected, attempts
