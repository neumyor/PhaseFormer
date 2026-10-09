"""Frozen b0 probe calibration and subsequent evaluation on disjoint supports."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from scripts.information_probe.core import (FAMILIES, DEGREES, CONTROLS, output_basis, fit_transforms,
    project, mismatch_indices, design, fit_ridge, predict, sample_metrics, summarize)
from scripts.message_switch_diagnostic.core import temporal_roles
from scripts.message_switch_diagnostic.extract import batches
from scripts.lowrank_token_interaction.model import load_model, state_hash
from scripts.lowrank_token_interaction.activation_audit import digest, save_json
from scripts.analyze_lowrank_stage2 import pick_cases
from scripts.analyze_lowrank_six import pack
from src.dataset.data_factory import data_provider

JOBS = [(d, h, s) for d in ('ETTh1', 'ETTm2', 'Weather') for h in (96, 720) for s in (2021, 2022)]


class Trace:
    def __init__(self, model):
        self.values = {}
        self.hooks = [model.lowrank_interaction_branch.register_forward_pre_hook(self.branch),
                      model.lowrank_dynamic_fusion.register_forward_pre_hook(self.fusion)]

    def branch(self, module, inputs):
        self.values['anchor'] = inputs[0][:, -1].detach()

    def fusion(self, module, inputs):
        self.values.update(phase=inputs[0].detach(), trend=inputs[1].detach(),
                           t=inputs[2].detach(), z=inputs[3].detach())

    def close(self):
        for hook in self.hooks:
            hook.remove()


def extract_batch(model, batch, trace, q, device):
    x, y, xm, ym = [v.float().to(device) for v in batch]
    # Future values never enter the predictor invocation or feature summaries.
    pred, _, _ = model(x, xm, torch.zeros_like(y), ym)
    truth = y[:, -model.pred_len:]
    sigma = x.var(1, unbiased=False).add(model.revin.eps).sqrt().double()
    values = trace.values
    phase = torch.arange(24, dtype=values['z'].dtype, device=device)
    z = values['z']
    sin = (z*torch.sin(2*torch.pi*phase/24)[None, None, :, None]).mean(-2)
    cos = (z*torch.cos(2*torch.pi*phase/24)[None, None, :, None]).mean(-2)
    raw = {'anchor': values['anchor'].reshape(-1, 1).cpu().double().numpy(),
           't': values['t'].flatten(0, 1).cpu().double().numpy(),
           'mean': z.mean(-2).flatten(0, 1).cpu().double().numpy(),
           'slot': torch.cat([z.mean(-2), sin, cos], -1).flatten(0, 1).cpu().double().numpy()}
    for name in ('phase', 'trend'):
        raw[name+'_modes'] = (values[name].double().transpose(1, 2) @ q).flatten(0, 1).cpu().numpy()
    error = (pred.double()-truth.double()).transpose(1, 2).flatten(0, 1).cpu().numpy()
    scale = sigma.flatten().cpu().numpy()
    target = (-error/scale[:, None]) @ q.cpu().numpy()
    return raw, target, error, scale, pred.cpu().numpy(), truth.cpu().numpy()


def concatenate(records):
    return {k: np.concatenate([r[k] for r in records]) for k in records[0]}


def role_data(model, data, ids, trace, q, device, batch_size):
    raws, targets, errors, scales = [], [], [], []
    for _, batch in batches(data, ids, batch_size):
        raw, target, error, scale, _, _ = extract_batch(model, batch, trace, q, device)
        raws.append(raw); targets.append(target); errors.append(error); scales.append(scale)
    return concatenate(raws), np.concatenate(targets), np.concatenate(errors), np.concatenate(scales)


def mismatch_bank(fit_raw, transforms, indices):
    return {k: project(fit_raw[k], transforms[k])[indices] for k in ('t', 'mean', 'slot')}


def work(args):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda:0')
    args.scratch.mkdir(parents=True, exist_ok=True)
    for j, (dataset, horizon, seed) in enumerate(JOBS):
        if j % args.shards != args.shard:
            continue
        setting = f'{dataset}_H{horizon}_S{seed}'
        out = args.scratch/setting
        out.mkdir(exist_ok=True)
        marker = out/(args.stage+'_complete.json')
        if marker.exists():
            prior = json.loads(marker.read_text())
            assert prior['source_hashes'] == source_hashes()
            continue
        started = time.monotonic()
        run = args.runs/f'{setting}_val__b0'
        model, config = load_model(run, device)
        assert config.dataset_args.seq_len == 720 and model.period_len == 24
        before = state_hash(model.state_dict())
        train, _ = data_provider(copy.deepcopy(config.dataset_args), 'train')
        val, _ = data_provider(copy.deepcopy(config.dataset_args), 'val')
        roles, cut = temporal_roles(len(train.data_x), len(val.data_x), 720, horizon)
        if args.smoke:
            # Retain disjoint supports and enough windows to exercise the ridge fits.
            roles['fit'] = roles['fit'][::max(1, len(roles['fit'])//80)]
            roles['cal'] = roles['cal'][:8]
            roles['eval'] = roles['eval'][:8]
        channels = train.data_x.shape[1]
        qn = output_basis(horizon)
        q = torch.as_tensor(qn, dtype=torch.float64, device=device)
        trace = Trace(model)
        provenance = {'setting': setting, 'dataset': dataset, 'horizon': horizon, 'seed': seed,
                      'cut': cut, 'channels': channels, 'source_hashes': source_hashes(),
                      'checkpoint_sha256': digest(json.loads((run/'config.json').read_text())['checkpoint']),
                      'dataset_sha256': digest(Path(config.dataset_args.root_path)/config.dataset_args.data_path),
                      'roles': {}, 'smoke': args.smoke}
        for role, ids in roles.items():
            offset = len(train.data_x)-720 if role == 'eval' else 0
            data = val if role == 'eval' else train
            provenance['roles'][role] = {'windows': len(ids), 'start': int(ids[0])+offset,
                'end': int(ids[-1])+offset+720+horizon-1,
                'history_start': str(data.timestamps[int(ids[0])]),
                'target_end': str(data.timestamps[int(ids[-1])+720+horizon-1])}
        with torch.inference_mode():
            if args.stage == 'calibrate':
                fit_raw, target, _, _ = role_data(model, train, roles['fit'], trace, q, device, config.dataset_args.batch_size)
                cal_raw, _, error, scale = role_data(model, train, roles['cal'], trace, q, device, config.dataset_args.batch_size)
                transforms = fit_transforms(fit_raw)
                support = 720+horizon
                fit_index = mismatch_indices(roles['fit'], roles['fit'], channels, support, same_role=True)
                cal_index = mismatch_indices(roles['fit'], roles['cal'], channels, support)
                fit_bank = mismatch_bank(fit_raw, transforms, fit_index)
                cal_bank = mismatch_bank(fit_raw, transforms, cal_index)
                baseline = summarize(sample_metrics(error))
                frozen = {'setting': setting, 'transforms': transforms, 'families': {},
                          'baseline_calibration': baseline, 'evaluation_opened': False,
                          'provenance': provenance, 'attempts': []}
                for family in FAMILIES:
                    xf = design(fit_raw, transforms, family, fit_bank)
                    xc = design(cal_raw, transforms, family, cal_bank)
                    attempts = [{'family': family, 'degrees': 0., 'fitted': None, **baseline}]
                    for degrees in DEGREES:
                        fitted = fit_ridge(xf, target, degrees)
                        corrected = error-scale[:, None]*(predict(xc, fitted) @ qn.T)
                        attempts.append({'family': family, 'degrees': degrees, 'fitted': fitted,
                                         **summarize(sample_metrics(corrected))})
                    chosen = min(attempts, key=lambda a: (a['mse'], a['degrees']))
                    frozen['families'][family] = chosen
                    frozen['attempts'].extend(attempts)
                # Only historical fit features are retained for the mismatch control.
                np.savez_compressed(out/'fit_bank.npz', ids=roles['fit'], **{
                    k: project(fit_raw[k], transforms[k]) for k in ('t', 'mean', 'slot')})
                save_json(out/'frozen.json', frozen)
            else:
                selection = json.loads((args.scratch/'selection_frozen.json').read_text())
                assert selection['evaluation_opened'] is False
                frozen = json.loads((out/'frozen.json').read_text())
                assert frozen['provenance']['source_hashes'] == source_hashes()
                assert frozen['provenance']['checkpoint_sha256'] == provenance['checkpoint_sha256']
                assert frozen['provenance']['dataset_sha256'] == provenance['dataset_sha256']
                transforms = frozen['transforms']; selected = selection['selected_family']
                readouts = {f: (f, frozen['families'][f]['fitted']) for f in FAMILIES}
                if selected != 'b0':
                    degree = frozen['families'][selected]['degrees']
                    for alias, family in zip(('receiver_matched', 'mismatch_matched'), CONTROLS[selected]):
                        trial = next(a for a in frozen['attempts'] if a['family'] == family and a['degrees'] == degree)
                        readouts[alias] = (family, trial['fitted'])
                with np.load(out/'fit_bank.npz') as z:
                    bank = {k: z[k].copy() for k in z.files}
                rows, frames = {k: [] for k in ('b0', *readouts)}, []
                for pos in range(0, len(roles['eval']), config.dataset_args.batch_size):
                    ids = roles['eval'][pos:pos+config.dataset_args.batch_size]
                    _, batch = next(batches(val, ids, len(ids)))
                    raw, target, error, scale, _, _ = extract_batch(model, batch, trace, q, device)
                    global_ids = ids+len(train.data_x)-720
                    index = mismatch_indices(bank['ids'], global_ids, channels, 720+horizon)
                    mb = {k: bank[k][index] for k in ('t', 'mean', 'slot')}
                    metrics = {'b0': sample_metrics(error)}
                    for name, (family, fitted) in readouts.items():
                        coeff = predict(design(raw, transforms, family, mb), fitted)
                        metrics[name] = sample_metrics(error-scale[:, None]*(coeff @ qn.T))
                    for name in metrics:
                        rows[name].append(metrics[name])
                    frame = pd.DataFrame({'sample_id': np.repeat(ids, channels),
                        'channel': np.tile(np.arange(channels), len(ids)), 'setting': setting+'_eval',
                        'dataset': dataset, 'horizon': horizon, 'seed': seed,
                        'baseline_config_id': 'b0', 'candidate_config_id': selected})
                    for label, name in [('baseline', 'b0'), ('candidate', selected)]:
                        for metric in ('mse', 'mae'):
                            frame[label+'_'+metric] = metrics[name][metric]
                    for metric in ('mse', 'mae'):
                        frame['delta_'+metric] = frame['candidate_'+metric]-frame['baseline_'+metric]
                    for family in readouts:
                        frame[family+'_mse'] = metrics[family]['mse']
                        frame[family+'_mae'] = metrics[family]['mae']
                    # The orthogonal oracle is descriptive and never a deployable candidate.
                    frame['normalized_mode_energy'] = np.square(target).sum(1)/horizon
                    frame['normalized_residual_energy'] = np.square(error/scale[:, None]).mean(1)
                    frame['orthogonal_oracle_mse'] = metrics['b0']['mse']-scale**2*np.square(target).sum(1)/horizon
                    frame['support_block'] = (global_ids.repeat(channels)-len(train.data_x))//(720+horizon)
                    frame['time_range'] = [f'{val.timestamps[int(i)+720]}/{val.timestamps[int(i)+720+horizon-1]}' for i in frame.sample_id]
                    frames.append(frame)
                errors = pd.concat(frames, ignore_index=True)
                errors.to_csv(out/'sample_errors.csv', index=False)
                results = []
                base = summarize(concatenate(rows['b0']))
                for name, parts in rows.items():
                    record = summarize(concatenate(parts))
                    record.update(setting=setting+'_eval', dataset=dataset, horizon=horizon, seed=seed,
                                  config_id=name, selected=name==selected, windows=len(roles['eval']))
                    for metric in ('mse', 'mae', 'mae_p95', 'mae_top1'):
                        label = {'mae_p95': 'p95', 'mae_top1': 'top1'}.get(metric, metric)
                        record['relative_'+label] = (record[metric]/base[metric]-1)*100
                    fitted = readouts[name][1] if name != 'b0' else None
                    record.update(degrees=0. if fitted is None else fitted['slope_edf'],
                                  parameters=0 if fitted is None else fitted['readout_parameters'])
                    results.append(record)
                pd.DataFrame(results).to_csv(out/'results.csv', index=False)
                requests = pick_cases(errors, horizon, top_k=3)
                cases = []
                for request in requests:
                    sample, ch = int(request['sample_id']), int(request['channel'])
                    item = val[sample]
                    batch = [torch.as_tensor(v[None]) for v in item]
                    raw, _, _, scale, bp, truth = extract_batch(model, batch, trace, q, device)
                    index = mismatch_indices(bank['ids'], np.array([sample+len(train.data_x)-720]), channels, 720+horizon)
                    mb = {k: bank[k][index] for k in ('t', 'mean', 'slot')}
                    coeff = np.zeros((channels, 4)) if selected == 'b0' else predict(
                        design(raw, transforms, selected, mb), frozen['families'][selected]['fitted'])
                    cp = bp[0].T+scale[:, None]*(coeff @ qn.T)
                    for label, prediction in [('baseline', bp[0, :, ch]), ('candidate', cp[ch])]:
                        check = sample_metrics((prediction-truth[0, :, ch]).astype(np.float64)[None])
                        for metric in ('mse', 'mae'):
                            np.testing.assert_allclose(check[metric][0], request[label+'_'+metric], rtol=5e-4, atol=5e-5)
                    cases.append({'metadata': request, 'history': item[0][:, ch], 'truth': truth[0, :, ch],
                                  'baseline': bp[0, :, ch], 'candidate': cp[ch], 'off': bp[0, :, ch],
                                  'phase': model.last_phase_forecast[0, :, ch].cpu().numpy(),
                                  'trend': model.last_lowrank_forecast[0, :, ch].cpu().numpy(),
                                  'gate': model.lowrank_dynamic_fusion.last_gate[0, ch].cpu().numpy(),
                                  'acceptance': coeff[ch]})
                pack(cases, out/'selected_cases.npz')
                save_json(out/'case_requests.json', requests)
                provenance['selection_sha256'] = digest(args.scratch/'selection_frozen.json')
                provenance['frozen_sha256'] = digest(out/'frozen.json')
        trace.close()
        assert before == state_hash(model.state_dict())
        save_json(marker, {**provenance, 'state_unchanged': True, 'seconds': time.monotonic()-started,
                           'completed_at': time.time()})
        print(setting, args.stage, 'complete', flush=True)
        del model, train, val
        torch.cuda.empty_cache()


def source_hashes():
    paths = [Path(__file__), Path(__file__).with_name('core.py'),
             Path('scripts/message_switch_diagnostic/core.py'), Path('scripts/lowrank_token_interaction/model.py')]
    return {str(p): digest(p) for p in paths}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--stage', choices=('calibrate', 'evaluate'), required=True)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    work(parser.parse_args())
