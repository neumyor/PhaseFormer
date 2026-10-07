import unittest

import pandas as pd

from scripts.analyze_lowrank_stage2 import pick_cases


class CaseSelectionTests(unittest.TestCase):
    def test_dedup_sign_and_ties(self):
        rows = [
            {'sample_id': 0, 'channel': 0, 'baseline_mae': 5., 'delta_mae': 2.},
            {'sample_id': 1, 'channel': 0, 'baseline_mae': 4., 'delta_mae': 1.},
            {'sample_id': 2, 'channel': 1, 'baseline_mae': 3., 'delta_mae': 2.},
            {'sample_id': 5, 'channel': 0, 'baseline_mae': 2., 'delta_mae': -1.},
            {'sample_id': 9, 'channel': 0, 'baseline_mae': 1., 'delta_mae': -2.},
        ]
        selected = pick_cases(pd.DataFrame(rows), horizon=4, top_k=3)
        regression = [r for r in selected if r['category'] == 'candidate_regression']
        self.assertEqual([(r['sample_id'], r['channel']) for r in regression], [(0, 0), (2, 1)])
        improvements = [r['sample_id'] for r in selected if r['category'] == 'candidate_improvement']
        self.assertEqual(improvements, [9, 5])
        self.assertEqual(selected, pick_cases(pd.DataFrame(rows[::-1]), horizon=4, top_k=3))


if __name__ == '__main__':
    unittest.main()
