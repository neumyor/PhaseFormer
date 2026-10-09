"""Close the fixed pilot and verify saved evidence without loading models."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import yaml

from scripts.analyze_lowrank_stage2 import markdown_table, pick_cases
from scripts.analyze_lowrank_six import unpack

FILES = {'run.yaml', 'results.csv', 'sample_errors.csv', 'selected_cases.npz',
         'objective_error_analysis.md', 'objective_error_analysis.zip', 'figures'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(scratch, audit):
    marker = json.loads((scratch / 'audit_complete.json').read_text())
    actual = {str(p.relative_to(audit)): digest(p) for p in audit.rglob('*') if p.is_file()}
    assert marker['status'] == 'passed' and actual == marker['artifact_hashes']
    assert {p.name for p in audit.iterdir()} == FILES
    run = yaml.safe_load((audit / 'run.yaml').read_text())
    assert all(digest(ROOT / name) == sha for name, sha in run['code']['source_hashes'].items())
    table = pd.read_csv(audit / 'results.csv')
    cases = unpack(audit / 'selected_cases.npz')
    settings = {s['setting'] for s in run['experiment']['settings']}
    assert settings == set(table.setting) == {c['metadata']['setting'] for c in cases}
    assert settings == {s['setting'] for s in run['execution']['settings']}
    assert settings == {s['setting'] for s in run['selection']['selected_configs']}
    assert not table.duplicated(['setting', 'config_id']).any() and len(table) == 24
    assert set(table.config_id) == {'b0', 'tp_add', 'tp_mod', 'tp_mod_centered'}
    assert not table.pretrained_checkpoint_loaded.any() and not table.smoke.any()
    for entry in run['selection']['selected_configs']:
        assert ((table.setting == entry['setting']) & (table.config_id == entry['config_id'])
                & table.selected).sum() == 1
    for _, group in table.groupby('setting'):
        assert len(group) == 4 and group.common_random_state_sha256.nunique() == 1
        assert group.dataset_sha256.nunique() == 1
    mean_cols = {c[5:]: c for c in table if c.startswith('mean_')}
    rank_cols = ['setting', 'candidate_config_id', 'sample_id', 'channel',
                 'baseline_mae', 'candidate_mae', 'candidate_mse', 'delta_mae']
    sums, counts, ranked = {}, {}, {}
    rows = 0
    for chunk in pd.read_csv(audit / 'sample_errors.csv', chunksize=50000):
        assert set(chunk.split) == {'val'}
        assert set(mean_cols).issubset(chunk.columns)
        for key, group in chunk.groupby(['setting', 'candidate_config_id']):
            values = group[list(mean_cols)].to_numpy(dtype=np.float64)
            assert np.isfinite(values).all()
            sums[key] = sums.get(key, np.zeros(len(mean_cols))) + values.sum(axis=0)
            counts[key] = counts.get(key, 0) + len(group)
            ranked.setdefault(key, []).append(group[rank_cols])
        rows += len(chunk)
    assert rows == marker['sample_rows'] == 1490440 and len(cases) == 162
    max_mean_difference, max_metric_difference = 0., 0.
    case_key = lambda m: (m['category'], m['rank'], m['sample_id'], m['channel'])
    for record in table.to_dict('records'):
        key = record['setting'], record['config_id']
        means = sums[key] / counts[key]
        expected = np.array([record[c] for c in mean_cols.values()])
        np.testing.assert_allclose(means, expected, atol=1e-7, rtol=3e-5)
        max_mean_difference = max(max_mean_difference, float(np.abs(means - expected).max()))
        frame = pd.concat(ranked[key], ignore_index=True)
        assert not frame.duplicated(['sample_id', 'channel']).any()
        for metric in ('mse', 'mae'):
            values = frame['candidate_' + metric]
            max_metric_difference = max(max_metric_difference, abs(values.mean() - record[metric]))
            np.testing.assert_allclose(values.mean(), record[metric], atol=1e-7)
            np.testing.assert_allclose(values.quantile(.95), record['candidate_' + metric + '_p95'], atol=1e-7)
            np.testing.assert_allclose(values.nlargest(max(1, int(np.ceil(len(values)*.01)))).mean(),
                                       record['candidate_' + metric + '_top1_mean'], atol=1e-7)
        if record['config_id'] != 'b0':
            expected_cases = pick_cases(frame, int(record['horizon']), top_k=3)
            actual_cases = [c['metadata'] for c in cases
                            if (c['metadata']['setting'], c['metadata']['candidate_config_id']) == key]
            assert {case_key(m) for m in expected_cases} == {case_key(m) for m in actual_cases}
            registered = [m for m in run['analysis']['selections']
                          if (m['setting'], m['candidate_config_id']) == key]
            assert {case_key(m) for m in expected_cases} == {case_key(m) for m in registered}
    for case in cases:
        for label in ('baseline', 'candidate', 'off', 'phase', 'trend'):
            error = case[label].astype(np.float64) - case['truth'].astype(np.float64)
            for metric, value in [('mse', np.square(error).mean()), ('mae', np.abs(error).mean())]:
                np.testing.assert_allclose(value, case['metadata'][label + '_' + metric],
                                           rtol=5e-4, atol=5e-5)
    report = (audit / 'objective_error_analysis.md').read_text()
    refs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', report)
    assert len(refs) == len(set(refs)) == 10
    assert {str(p.relative_to(audit)) for p in (audit / 'figures').iterdir()} == set(refs)
    for name in refs:
        path = Path(name)
        assert path.parts[0] == 'figures' and len(path.parts) == 2
        assert not (audit / name).is_symlink()
    with zipfile.ZipFile(audit / 'objective_error_analysis.zip') as archive:
        assert set(archive.namelist()) == {'objective_error_analysis.md', *refs}
        for name in archive.namelist():
            assert archive.read(name) == (audit / name).read_bytes()
    return {'status': 'passed', 'sample_rows': rows, 'case_records': len(cases),
            'aggregate_records': len(table), 'figures': len(refs),
            'max_mean_recomputation_difference': max_mean_difference,
            'max_mse_mae_recomputation_difference': float(max_metric_difference),
            'all_numeric_sample_means_tails_rankings_and_case_predictions_checked': True,
            'artifact_hashes': actual}


def finalize(scratch, audit):
    verification = verify(scratch, audit)
    run = yaml.safe_load((audit / 'run.yaml').read_text())
    if run['analysis'].get('interpretation_finalized'):
        raise ValueError('Interpretation already finalized')
    original = scratch / 'closure_previous_artifacts'
    original.mkdir(exist_ok=False)
    for name in ('run.yaml', 'results.csv', 'objective_error_analysis.md', 'objective_error_analysis.zip'):
        shutil.copy2(audit / name, original / name)
    shutil.copy2(scratch / 'audit_complete.json', original / 'audit_complete.json')
    (scratch / 'local_original_verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    table = pd.read_csv(audit / 'results.csv')
    repeat = pd.DataFrame(json.loads((scratch / 'benchmark_recheck.json').read_text()))
    assert len(repeat) == len(table) and not repeat.duplicated(['setting', 'config_id']).any()
    assert set(zip(repeat.setting, repeat.config_id)) == set(zip(table.setting, table.config_id))
    repeat = repeat.rename(columns={c: 'recheck_' + c for c in repeat if c not in ('setting', 'config_id')})
    table = table.merge(repeat, on=['setting', 'config_id'], validate='one_to_one')
    full = table[table.config_id == 'tp_mod_centered'].set_index('setting').sort_index()
    base = table[table.config_id == 'b0'].set_index('setting').sort_index()
    old = table[table.config_id == 'tp_mod'].set_index('setting').sort_index()
    assert full.index.equals(base.index) and full.index.equals(old.index)
    recheck_gates = {tag + '_latency': bool((full['recheck_' + tag + '_latency_vs_b0'] <= 1.05).all()
                                          and (full['recheck_' + tag + '_p95_vs_b0'] <= 1.1).all())
                     for tag in ('batch1', 'native_batch')}
    run['analysis']['timing_recheck'] = {
        'scope': 'Existing weights, same frozen benchmark, serial idle GPU; no training or source changes.',
        'benchmark_sha256': digest(scratch / 'benchmark_recheck.json'),
        'launch': json.loads((scratch / 'benchmark_recheck_launch.json').read_text()),
        'checks': recheck_gates,
        'epoch_cost_status': 'Unconfirmed: concurrent training uses a shared host; no isolated epoch repeat.'}
    run['selection']['decision_status'] = 'pilot_rejected_no_expansion'
    for entry in run['selection']['selected_configs']:
        entry['search_notes'] = 'Retain b0: centered candidate fails fixed effect, tail and repeated latency gates; no controls, expansion or new test.'
    run['analysis'].update(interpretation_finalized=True,
        architecture_decision='Retain static level/shape fusion b0; reject this centered TP-Mod upgrade.',
        advance_to_controls=False, finalizer_sha256=digest(Path(__file__)),
        training_code_commit=run['code']['commit'])
    comparison = pd.DataFrame({'setting': full.index,
        'level_delta_vs_b0': (full.mean_candidate_level_mse - base.mean_candidate_level_mse).values,
        'shape_delta_vs_b0': (full.mean_candidate_shape_mse - base.mean_candidate_shape_mse).values,
        'native_minus_off_shape': (full.mean_candidate_shape_mse - full.mean_off_shape_mse).values,
        'top1_mae_percent_vs_b0': ((full.candidate_mae_top1_mean / base.candidate_mae_top1_mean - 1)*100).values,
        'top1_mae_percent_vs_tp_mod': ((full.candidate_mae_top1_mean / old.candidate_mae_top1_mean - 1)*100).values})
    cases = unpack(audit / 'selected_cases.npz')
    rain = next(c for c in cases if c['metadata']['setting'] == 'Weather_H720_S2022_val'
                and c['metadata']['candidate_config_id'] == 'tp_mod_centered'
                and c['metadata']['sample_id'] == 1702 and c['metadata']['channel'] == 14)['metadata']
    shape_delta = float((full.mean_candidate_shape_mse - base.mean_candidate_shape_mse).mean())
    relative_shape = float(((full.mean_candidate_shape_mse / base.mean_candidate_shape_mse - 1)*100).mean())
    cycle = [ast.literal_eval(v) for v in full.cycle_mean_check]
    lines = ['', '## 12. 最终结论与独立复核（2026-10-09）', '',
        '**保留静态双分量融合 b0，不采用中心化 TP-Mod，不扩展控制、seed、任务或新 test。**',
        '24/24 正式训练成功，无失败或续跑；六个 H720 validation setting、两 seed、四臂从头联合训练。18 个历史对照 MSE/MAE 完全复现，因此这 18 次是协议复现，不是新增独立随机重复。',
        '中心化相对 b0 的 setting 等权平均相对 MSE/MAE 为 +0.051515%/+0.157214%，仅 2/6 MSE 胜出；相对原 TP-Mod 为 +0.051340%/+0.034373%，同样 2/6 胜出。既未达到平均 MSE 改善 1% 的门槛，也未达到 MAE 不升等条件。',
        '以下是三个数据集各两 seed 的原始 MSE/MAE 均值，全部是 validation H720；不是此前六任务 test、论文或 Golden 结果。', '']
    lines += markdown_table(table.groupby(['dataset', 'config_id'])[['mse', 'mae']].mean().reset_index())
    lines += ['', '### 约束成立与收益失败', '',
        f'六个最高精度周期均值检查均通过，最大直接变化范围 {min(c["max_abs_cycle_mean_change"] for c in cycle):.3e} 至 {max(c["max_abs_cycle_mean_change"] for c in cycle):.3e}。这是数值容差内的实现证据，不能据此称独立联合训练后的整个模型水平预测不变。',
        'ETTm2 两 seed 的直接 shape 写入都改善，但重新联合训练的 level 误差增加抵消了收益；ETTh1 两 seed 的 native-minus-off shape 误差为正，直接波形修正本身有害。Weather 的直接 shape 修正小幅改善，但尾部 MAE 超限。',
        f'相对 b0，原始 shape MSE 的六 setting 均值变化为 {shape_delta:+.8f}；平均相对 shape 变化为 {relative_shape:+.6f}%。二者权重不同，不能用微小相对 shape 改善替代整体效果门槛。', '']
    lines += markdown_table(comparison)
    lines += ['', f'程序自动选出的 Weather-S2022 rain 样本 1702/c14：b0/native/off MAE = {rain["baseline_mae"]:.6f}/{rain["candidate_mae"]:.6f}/{rain["off_mae"]:.6f}；level MSE = {rain["baseline_level_mse"]:.6f}/{rain["candidate_level_mse"]:.6f}/{rain["off_level_mse"]:.6f}。该退化主要位于独立训练后的 level 预测，native 与 off 的 level 相同；不能归因于中心化模块直接写入周期均值。',
        'same-checkpoint off 仅定位直接写入作用，不能替代独立 b0，也不能因果分离联合训练适应、优化与信息冗余。此次结果否定的是当前固定中心化调制实例的稳定实用收益，不能否定全部 shape 交互假设。', '',
        '### 效率原始记录与隔离复测', '',
        '新增参数为 244/196/196，占 b0 约 0.13%--0.15%；新增算术保守比例低于 0.34%，显存增量很小。但低算术量不自动等于低实测延迟。首次 Weather-S2021 native median +60.69% 在复测变为 +2.97%；ETTm2 两 seed 的 batch1/native median 仍超过 5% 边界。保留全部原始和复测列，不选择最快读数。',
        'epoch 时间来自并发任务与共享 CPU，存在负载混杂；原 rubric 的数值失败保留，但不能据此确定架构的独立 epoch 开销。没有为已失败的效果先导追加训练计时。', '']
    columns = ['batch1_latency_vs_b0', 'native_batch_latency_vs_b0',
               'recheck_batch1_latency_vs_b0', 'recheck_native_batch_latency_vs_b0',
               'recheck_batch1_p95_vs_b0', 'recheck_native_batch_p95_vs_b0']
    lines += markdown_table(full.reset_index()[['setting', *columns]])
    lines += ['', '### 审计范围与可比性', '',
        '完整 1490440 条 sample/channel/arm 记录、162 案例快照、10 图已本地流式重算全部数值均值、MSE/MAE 尾部、Top3 去重排名及五种预测的案例误差；目录/setting/图片白名单/ZIP 字节一致性通过。',
        '最初同步时样本表仍在远程合并，得到不完整副本；本地重算发现后重新同步完成原件并核对 SHA256。该问题属于传输一致性，不是训练失败；原始完整远程结果保留。',
        '研究方向受此前 test 查看影响，本轮仅使用已暴露 validation。两 seed、重叠预测窗及小差值不构成统计显著性或盲测泛化证据，不宣称胜过 Golden。rank≤2 只约束 BA 映射；中心化也不提供收缩或稳定性保证。', '']
    report = (audit / 'objective_error_analysis.md').read_text() + '\n'.join(lines)
    refs = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', report)
    table.to_csv(audit / 'results.csv', index=False)
    (audit / 'objective_error_analysis.md').write_text(report)
    (audit / 'run.yaml').write_text(yaml.safe_dump(run, sort_keys=False))
    with zipfile.ZipFile(audit / 'objective_error_analysis.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ['objective_error_analysis.md', *refs]:
            archive.write(audit / name, name)
    marker = json.loads((scratch / 'audit_complete.json').read_text())
    marker['artifact_hashes'] = {str(p.relative_to(audit)): digest(p) for p in audit.rglob('*') if p.is_file()}
    (scratch / 'audit_complete.json').write_text(json.dumps(marker, indent=2) + '\n')
    final = verify(scratch, audit)
    (scratch / 'local_final_verification.json').write_text(json.dumps(final, indent=2) + '\n')
    print(json.dumps({k: v for k, v in final.items() if k != 'artifact_hashes'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch-dir', type=Path, required=True)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify(args.scratch_dir, args.audit_dir), indent=2))
    else:
        finalize(args.scratch_dir, args.audit_dir)
