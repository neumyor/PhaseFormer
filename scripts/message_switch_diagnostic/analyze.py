"""Fit and calibrate on earlier blocks before opening evaluation metrics."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import yaml
from scripts.message_switch_diagnostic.core import FEATURES, fit_selector, score, calibration_choice, summary
from scripts.lowrank_token_interaction.activation_audit import digest, save_json
from scripts.analyze_lowrank_stage2 import pick_cases


def decision(records):
    table=pd.DataFrame(records)
    checks={}
    for baseline in ('native','b0'):
        dm=(table.mse/table[baseline+'_mse']-1)*100
        da=(table.mae/table[baseline+'_mae']-1)*100
        checks[baseline+'_effect']=bool(dm.mean()<=-1 and (dm<0).sum()>=8
                                       and da.mean()<=0 and dm.max()<=2)
    checks['tail']=bool(table.tail_pass.all())
    checks['compensation']=bool((table.lost_benefit_fraction<=.1).all()
                                and (table.lost_extreme_benefit_fraction<=.1).all())
    checks['oracle_recovery']=bool(table.oracle_recovery.mean()>=.2)
    return {'checks':checks,'effect_screen_passed':bool(all(checks.values())),
            'efficiency_status':'Not yet measured; no architecture training authorized by this result alone.'}


def main(args):
    args.audit.mkdir(parents=True,exist_ok=False)
    parts=args.scratch/'parts'
    provenance=[json.loads(p.read_text()) for p in sorted(parts.glob('*.json'))]
    assert len(provenance)==12 and not any(p['smoke'] for p in provenance)
    for p in provenance:
        assert digest(parts/(p['setting']+'.csv'))==p['csv_sha256']
        assert p['roles']['fit']['end']<p['roles']['cal']['start']
        assert p['roles']['cal']['end']<p['roles']['eval']['start']
    # Only fit/cal arrays are used before the frozen selection is written.
    frames=[pd.read_csv(parts/(p['setting']+'.csv')) for p in provenance]
    data=pd.concat(frames,ignore_index=True)
    fit=data[data.role=='fit'].reset_index(drop=True)
    cal=data[data.role=='cal'].reset_index(drop=True)
    selectors={};searches={}
    for name,columns in [('history4',FEATURES[:4]),('history_activation8',FEATURES)]:
        selector=fit_selector(fit,columns)
        chosen,attempts=calibration_choice(cal,score(cal,selector))
        selector['threshold']=chosen['threshold'];selector['threshold_name']=chosen['name']
        selector['cal_relative_mse']=chosen['relative_mse']
        selectors[name]=selector;searches[name]=attempts
    selected=min(selectors,key=lambda n:(selectors[n]['cal_relative_mse'],len(selectors[n]['columns']),n))
    freeze={'selected_selector':selected,'selectors':selectors,'calibration_search':searches,
            'source_parts':provenance,'analysis_sha256':digest(__file__),
            'core_sha256':digest(Path(__file__).with_name('core.py')),
            'evaluation_opened':False}
    save_json(args.scratch/'selector_frozen_before_evaluation.json',freeze)
    eval_frame=data[data.role=='eval'].reset_index(drop=True)
    eval_frame['candidate_config_id']=selected
    eval_frame['baseline_config_id']='b0'
    eval_frame['selector_score']=score(eval_frame,selectors[selected])
    disabled=eval_frame.selector_score.to_numpy()>=selectors[selected]['threshold']
    eval_frame['disabled']=disabled
    eval_frame['native_mse']=eval_frame.candidate_mse
    eval_frame['native_mae']=eval_frame.candidate_mae
    eval_frame['candidate_mse']=np.where(disabled,eval_frame.off_mse,eval_frame.native_mse)
    eval_frame['candidate_mae']=np.where(disabled,eval_frame.off_mae,eval_frame.native_mae)
    eval_frame['delta_mse']=eval_frame.candidate_mse-eval_frame.baseline_mse
    eval_frame['delta_mae']=eval_frame.candidate_mae-eval_frame.baseline_mae
    eval_frame['misclosed_benefit']=np.where(disabled,np.maximum(eval_frame.off_mse-eval_frame.native_mse,0),0)
    eval_frame['missed_harm']=np.where(~disabled,np.maximum(eval_frame.native_mse-eval_frame.off_mse,0),0)
    eval_frame.to_csv(args.audit/'sample_errors.csv',index=False)
    records=[];main_records=[];selections=[]
    for setting,g in data[data.role=='eval'].groupby('setting',sort=True):
        for name in ['native','off','b0','oracle',*selectors]:
            if name=='native':action=np.zeros(len(g),bool)
            elif name=='off':action=np.ones(len(g),bool)
            elif name=='oracle':action=(g.off_mse<g.candidate_mse).to_numpy()
            elif name=='b0':action=np.zeros(len(g),bool)
            else:action=score(g,selectors[name])>=selectors[name]['threshold']
            r=summary(g,action)
            native=summary(g,np.zeros(len(g),bool))
            if name=='b0':
                r['mse']=float(g.baseline_mse.mean());r['mae']=float(g.baseline_mae.mean())
                r['mae_p95']=float(g.baseline_mae.quantile(.95))
                r['mae_top1']=float(g.baseline_mae.nlargest(max(1,int(np.ceil(len(g)*.01)))).mean())
            r.update(setting=setting+'_eval',config_id=name,dataset=g.dataset.iloc[0],
                     horizon=int(g.horizon.iloc[0]),seed=int(g.seed.iloc[0]),split='temporal_validation',
                     model='Frozen PhaseFormer-L TP-Add',selected=name==selected,windows=int(g.sample_id.nunique()),
                     extra_parameters=len(selectors[name]['beta']) if name in selectors else 0,
                     relative_mse_native=(r['mse']/r['native_mse']-1)*100,
                     relative_mse_b0=(r['mse']/r['b0_mse']-1)*100,
                     relative_mae_native=(r['mae']/r['native_mae']-1)*100,
                     relative_mae_b0=(r['mae']/r['b0_mae']-1)*100,
                     tail_pass=bool(r['mae_p95']<=native['mae_p95']*1.02 and r['mae_top1']<=native['mae_top1']*1.02
                         and r['mae_p95']<=g.baseline_mae.quantile(.95)*1.02
                         and r['mae_top1']<=g.baseline_mae.nlargest(max(1,int(np.ceil(len(g)*.01)))).mean()*1.02))
            records.append(r)
            if name==selected:main_records.append(r)
        subset=eval_frame[eval_frame.setting==setting].copy()
        subset['setting']=setting+'_eval'
        selections.extend(pick_cases(subset,int(g.horizon.iloc[0]),top_k=3))
    eval_frame['setting']=eval_frame.setting+'_eval'
    eval_frame.to_csv(args.audit/'sample_errors.csv',index=False)
    pd.DataFrame(records).to_csv(args.audit/'results.csv',index=False)
    save_json(args.scratch/'case_requests.json',selections)
    outcome=decision(main_records)
    run={'experiment_id':args.audit.name,
         'code':{'commit':args.code_commit,'modified_files':['scripts/message_switch_diagnostic/'],
                 'source_model_commit':'dcde3b684f346b14df91d733ad22a159e97393c2'},
         'mechanism':{'description':'Shared linear loss-weighted hard switch on existing TP-Add messages; static output fusion retained.'},
         'experiment':{'baseline':'b0','candidate':selected,'settings':[
             {'setting':p['setting']+'_eval','dataset':p['dataset'],'horizon':p['horizon'],'lookback':720,
              'seed':p['seed'],'split':'support-disjoint validation tail'} for p in provenance],
             'training':'Frozen predictor; selector fitted on original train prefix, threshold calibrated on disjoint train suffix.',
             'metrics':['MSE','MAE','tail','lost compensation','oracle recovery']},
         'execution':{'environment':{'extraction':'torch2.6.0+cu124; highest float32; A800'},
                      'settings':[{'setting':p['setting']+'_eval','runtime':p,'commands':['extract.py -> analyze.py -> collect_cases.py -> report.py']} for p in provenance]},
         'selection':{'source':'calibration','research_direction_test_informed':True,'test_set_selection':False,
                      'selected_selector':selected,'selected_configs':[{'setting':p['setting']+'_eval','config_id':selected,
                        'search_notes':'Shared model and threshold frozen on earlier calibration before evaluation metrics.'} for p in provenance],
                      'selectors':selectors,'searches':searches},
         'analysis':{'ranking_metric':'mae','top_k':3,'dedup_rule':'within category/channel starts separated >=H',
                     'selections':selections,'decision':outcome,
                     'disclosure':'Backbone trained on all original train, checkpoint selected on original val. Temporal transfer diagnostic, not blind test; no new test read.'},
         'validation':{'status':'incomplete','results_checked':True,'ranking_and_cases_checked':False,
                       'report_and_archive_checked':False,'directory_and_settings_checked':False}}
    (args.audit/'run.yaml').write_text(yaml.safe_dump(run,sort_keys=False))
    print(selected,json.dumps(outcome),flush=True)
    print(pd.DataFrame(main_records)[['setting','mse','relative_mse_native','relative_mse_b0',
          'lost_benefit_fraction','lost_extreme_benefit_fraction','oracle_recovery']].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--code-commit',required=True)
    main(parser.parse_args())
