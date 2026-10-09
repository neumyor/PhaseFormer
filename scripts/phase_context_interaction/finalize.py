"""Append the fixed pilot interpretation without overwriting original readings."""
import argparse
import json
from pathlib import Path
import re
import zipfile

import pandas as pd
import yaml


def finalize(scratch, audit):
    from scripts.run_lowrank_interaction_stage2 import digest, save_json
    complete=json.loads((scratch/'audit_complete.json').read_text())
    if complete['status']!='passed':raise ValueError('Audit has not passed')
    for name,sha in complete['artifact_hashes'].items():
        if digest(audit/name)!=sha:raise ValueError('Changed artifact: '+name)
    run=yaml.safe_load((audit/'run.yaml').read_text())
    if run['analysis'].get('interpretation_finalized'):raise ValueError('Already finalized')
    table=pd.read_csv(audit/'results.csv')
    recheck=pd.DataFrame(json.loads((scratch/'timing_recheck/benchmark.json').read_text()))
    cols=[c for c in recheck if c not in ('setting','config_id')]
    recheck=recheck.rename(columns={c:'recheck_'+c for c in cols})
    table=table.merge(recheck,on=['setting','config_id'],validate='one_to_one')
    table.to_csv(audit/'results.csv',index=False)
    run['analysis']['timing_recheck']={
        'scope':'existing weights, no training, serial idle-GPU repeat; original readings retained',
        'code_commit':run['code']['commit'],
        'interpretation':'Epoch training times came from concurrent jobs on different GPUs and shared host CPU; no isolated full-epoch remeasurement, so exact training-cost claim remains unconfirmed.'}
    full=table[table.config_id=='context_full'].set_index('setting')
    tp=table[table.config_id=='tp_add'].set_index('setting')
    observations={}
    for field in ('mean_direct_harm_positive','mean_direct_harm_indicator','mean_severe_cancellation'):
        observations[field]={'full':float(full[field].mean()),'tp_add':float(tp[field].mean()),
                             'relative_change_percent':float((full[field].mean()/tp[field].mean()-1)*100)}
    scores={
        'H1':{'score':0,'conclusion':'Not supported in this pilot: only1/4 MSE wins versus b0; mean MSE worse than b0 and TP-Add. Matched-context controls not run because screening failed.'},
        'H2':{'score':None,'conclusion':'Untested experimentally: no untied training after pilot failure; algebraic rank constraint and gradients verified.'},
        'H3':{'score':1,'conclusion':'Mixed diagnostic evidence only: positive harm severity falls about9.9%, but severe cancellation rises about15.0%; efficacy condition also fails.'},
        'H4':{'score':1,'conclusion':'Very small parameter/compute and inference-memory increments; inference timings measured twice. Strict full resource claim unconfirmed, especially concurrent per-epoch training time.'}}
    run['analysis'].update(hypothesis_scores=scores,mechanism_observations=observations,
        interpretation_finalized=True)
    run['selection']['decision_status']='pilot_rejected_no_expansion'
    for entry in run['selection']['selected_configs']:
        entry['search_notes']='b0 retained; full candidate failed preregistered validation gate; no control/expansion/test run'
    report=(audit/'objective_error_analysis.md').read_text()
    lines=['','## 12. 最终先导判断（2026-10-09）','',
        '**不升级架构，不追加错配/非共享控制、完整六任务或第三seed训练。**16次正式训练全部成功；模型机制、固定Gate及原rank保持约束。',
        '相对b0，主候选setting等权平均相对MSE为+0.169258%、MAE为-0.155780%，仅1/4 MSE胜出；相对TP-Add分别+0.229480%/+0.305922%。平均相对变化和原始误差均值分别报告，不能混用。',
        '先导失败否定的是当前线性均值读取/共享写入实例的实用收益，不是所有phase条件化交互。两seed、小幅差值不构成统计显著性证明。研究方向受既有test查看影响，本轮没有读取新test。','',
        table.groupby(['dataset','config_id'])[['mse','mae']].mean().to_markdown(),'']
    lines+=['| 假设 | 评分 | 客观结论 |','|---|---|---|']
    for h,d in scores.items():lines.append(f"| {h} | {d['score'] if d['score'] is not None else '未测'} | {d['conclusion']} |")
    lines+=['','错误修正正值均值相对TP-Add下降约9.90%，误修正发生频率下降约1.30%，但严重抵消发生率上升约15.04%。这些读数使用各自checkpoint off参考，只能辅助归因，不能证明去重或因果机制。','',
        '重复退化案例：ETTh1-H720两seed都自动选中sample811/c4；native MAE为3.44745/3.23881，独立b0为3.35198/3.17287，same-checkpoint off为2.84494/2.76212。这两个局部案例支持错误写入的诊断，不能外推总体比例。',
        'Weather-S2021 sample1702/c14 rain(mm)：b0/native/off MAE=4.98161/7.46177/68.83461，仍存在有用补偿但独立架构更差的情况。S2022 sample1957/c14也退化：0.91432/2.48088/3.33981。','',
        '候选额外参数：ETTh1-H720为308（+0.16110%），Weather-H96为56（+0.35994%）；新增标量运算量相对整模型计数下界的保守比例为0.07030%/0.20033%。这些是低成本证据，不能替代效能收益。',
        '原串行计时有个别native-batch p95门槛失败，因此额外在空闲GPU复测；下面并列结果，原始数据保留。每epoch时间来自并发训练，有共享CPU/GPU负载差异，不以该异常比例直接声称机制训练开销超标或已过门槛。','']
    columns=['setting','batch1_latency_vs_b0','native_batch_latency_vs_b0','native_batch_p95_vs_b0',
             'recheck_batch1_latency_vs_b0','recheck_native_batch_latency_vs_b0','recheck_native_batch_p95_vs_b0']
    lines+=[full.reset_index()[columns].to_markdown(index=False),'',
        '所有变量及去气压Weather汇总见run.yaml中的channel_summaries；不新增变量路由。H2未做正式训练对照，不能给支持/反证评分。','',
        '保留静态双分量b0作为可靠参照。本轮未验证phase条件化新设计，不自动增大k、引入非线性或动态Gate。后续新机制需新的假设与授权，而非放宽本轮门槛。','']
    report+='\n'.join(lines)
    (audit/'objective_error_analysis.md').write_text(report)
    refs=re.findall(r'!\[[^\]]*\]\(([^)]+)\)',report)
    with zipfile.ZipFile(audit/'objective_error_analysis.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['objective_error_analysis.md',*refs]:z.write(audit/name,name)
    with zipfile.ZipFile(audit/'objective_error_analysis.zip') as z:
        assert set(z.namelist())=={'objective_error_analysis.md',*refs}
        for name in z.namelist():assert z.read(name)==(audit/name).read_bytes()
    run['analysis']['finalizer_sha256']=digest(Path(__file__))
    (audit/'run.yaml').write_text(yaml.safe_dump(run,sort_keys=False))
    save_json(scratch/'audit_complete.json',{**run['validation'],
        'artifact_hashes':{str(p.relative_to(audit)):digest(p) for p in audit.rglob('*') if p.is_file()}})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scratch-dir',type=Path,required=True)
    parser.add_argument('--audit-dir',type=Path,required=True)
    args=parser.parse_args()
    finalize(args.scratch_dir,args.audit_dir)
