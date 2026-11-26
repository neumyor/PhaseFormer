#!/usr/bin/env python3
"""Trend and periodic errors of the selected E1 runs (cycle-level definition, P_eval = 24).

For each of the twelve settings (Weather, Traffic, ETTh1, ETTh2, ETTm1, ETTm2 at H = 96, 192) and each selection (val, test)
of 00_protocol/e1_final_configs_28.csv, the E1 model is rebuilt from the job of the run, loaded from best.ckpt and run on the
test split in dataloader order. The element-mean test MSE of the exported predictions must equal the recorded test_mse of the run
(otherwise the script stops). The truth array and the scoring are those of the component-MSE experiment
(scripts/component_mse/dump.py and trend_definitions.py, truth under ~/niuyiming/component_mse_20261023/predictions).

For ETTm2-96 the predictions and the per-(window, channel) errors are kept for the case study.

    CUDA_VISIBLE_DEVICES=4 python3 scripts/e1_paper/trend_errors.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

import scripts.b0hp_structure_ablation as sa  # noqa: E402
import scripts.phaseformer_e13 as pe13  # noqa: E402
from scripts.component_mse import dump  # noqa: E402
from scripts.component_mse import trend_definitions as td  # noqa: E402
from scripts.vldbj_m1.proto import build_model  # noqa: E402
from src.dataset.data_factory import data_provider  # noqa: E402

BASE = ROOT / 'research_runs/e1_paper_v1'
DUMP = Path.home() / 'niuyiming/component_mse_20261023/predictions'
SETTINGS = tuple((d, h) for d in ('Weather', 'Traffic', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2') for h in (96, 192))
KEEP = ('ETTm2', 96)
PERIOD = 24


def export(row):
    job = dict(json.loads(Path(row['job']).read_text()), smoke=True)
    job.pop('source_hashes', None)
    job.pop('code_commit', None)
    record = json.loads((Path(row['run_dir']) / 'result.json').read_text())
    model, args, _, _ = build_model(job)
    pe13.apply_structure(model, job['ablation_arm'], job)
    state = torch.load(record['checkpoint'], map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(state)
    device = torch.device('cuda:0')
    model.to(device).eval()
    _, loader = data_provider(args.dataset_args, 'test')
    predictions, truths = [], []
    with torch.inference_mode():
        for batch in loader:
            bx, by, bxm, bym = [v.float().to(device) for v in batch]
            pred = model(bx, bxm, model._build_decoder_input(by), bym)[0][:, -model.pred_len:]
            predictions.append(pred.cpu().numpy())
            truths.append(by[:, -model.pred_len:].cpu().numpy())
    return np.concatenate(predictions), np.concatenate(truths), record['best']['test_mse']


def main():
    sa.register_heads()
    rows = [r for r in csv.DictReader((BASE / '00_protocol/e1_final_configs_28.csv').open())
            if (r['dataset'], int(r['horizon'])) in SETTINGS]
    out, scored = [], {}
    (BASE / '03_trend_errors/e1_predictions').mkdir(parents=True, exist_ok=True)
    for r in rows:
        key = (r['dataset'], int(r['horizon']))
        if r['run_dir'] in scored:
            result = scored[r['run_dir']]
        else:
            prediction, truth, recorded = export(r)
            reference = dump.load_truth(DUMP, key[0], key[1])
            if reference.shape != truth.shape or not np.allclose(reference, truth, atol=1e-5):
                raise AssertionError(f'{key}: truth of the loader differs from the shared truth array')
            mse, mae = dump.test_arrays(prediction, truth)
            if abs(mse - recorded) > max(1e-5, 1e-4 * recorded):
                raise AssertionError(f'{key} {r["selection"]}: exported MSE {mse} != recorded {recorded}')
            s = td.score_file(prediction, truth, PERIOD, ('cycle-level',))['cycle-level']
            result = dict(mse_trend=s['trend'], mse_periodic=s['periodic'], mse_total=s['total'],
                          test_mse_exported=mse, test_mse_recorded=recorded, test_mae_exported=mae,
                          windows=int(prediction.shape[0]))
            scored[r['run_dir']] = result
            if key == KEEP:
                error = prediction - truth
                trend = td.apply_trend(error.astype('float32'), 'cycle-level', PERIOD)
                np.savez_compressed(BASE / f'03_trend_errors/e1_predictions/E1-{r["selection"]}__{key[0]}_H{key[1]}.npz',
                                    prediction=prediction.astype('float32'),
                                    trend_error=np.square(trend).mean(1).astype('float32'),
                                    periodic_error=np.square(error - trend).mean(1).astype('float32'))
            print(key, r['selection'], {k: round(v, 5) if isinstance(v, float) else v for k, v in result.items()}, flush=True)
        out.append(dict(dataset=key[0], horizon=key[1], selection=r['selection'], stage=r['stage'], run_dir=r['run_dir'], **result))
    target = BASE / '03_trend_errors/e1_trend_scores.csv'
    with target.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print('wrote', target, len(out))


if __name__ == '__main__':
    main()
