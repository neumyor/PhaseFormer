import unittest

import pandas as pd

from scripts.finalize_parallel_static import decision


class ParallelDecisionTests(unittest.TestCase):
    def table(self):
        rows=[]
        for dataset in ['ETTh1','ETTm2','Weather']:
            for horizon in [96,720]:
                for seed in [2021,2022]:
                    for arm,value in [('components_static',1.),('parallel_components_static',.99)]:
                        rows.append({'setting':f'{dataset}_H{horizon}_S{seed}_val',
                            'dataset':dataset,'horizon':horizon,'seed':seed,'config_id':arm,
                            **{k:value for k in ['mse','mae','mean_candidate_level_mse',
                                'mean_candidate_shape_mse','candidate_mse_p95','candidate_mse_top1_mean']}})
        return pd.DataFrame(rows)

    def test_upgrade_requires_all_gates_and_consistent_tail_protection(self):
        table=self.table()
        self.assertTrue(decision(table)['performance_gate_passed'])
        selected=(table.config_id=='parallel_components_static')&(table.dataset=='Weather')&(table.horizon==720)
        table.loc[selected,'candidate_mse_top1_mean']=1.06
        result=decision(table)
        self.assertFalse(result['performance_gate_passed'])
        self.assertEqual(result['default_candidate'],'components_static')
        self.assertEqual(len(result['blocking_tasks']),1)

    def test_mae_regression_prevents_default_upgrade(self):
        table=self.table();table.loc[table.config_id=='parallel_components_static','mae']=1.001
        result=decision(table)
        self.assertFalse(result['performance_gate_passed'])
        self.assertTrue(result['criteria']['mean_relative_mse_below_zero'])
        self.assertFalse(result['criteria']['mean_relative_mae_nonpositive'])


if __name__=='__main__':unittest.main()
