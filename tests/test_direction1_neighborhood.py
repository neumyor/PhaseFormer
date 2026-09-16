import unittest

import numpy as np

from scripts.compute_direction1_neighborhood_projectors import (
    accumulate_block_moments,
    bootstrap_direction_cloud,
    build_neighborhood_bases,
    contiguous_origin_blocks,
    normalized_direction1,
    sign_align,
    subspace_statistics,
    tangent_pca,
    weighted_moments,
)
from scripts.run_direction1_neighborhood_matrix import (
    arm_command,
    confirm_cells,
    cone_arm,
    sweep_cells,
)
from scripts.select_direction1_neighborhood_width import evaluate_dataset


class Direction1NeighborhoodTest(unittest.TestCase):
    def test_tangent_neighborhood_is_nested_and_contains_reference(self):
        reference = np.array([1.0, 0.0, 0.0, 0.0])
        cloud = np.array(
            [
                [0.99, 0.10, 0.02, 0.00],
                [-0.98, -0.12, 0.00, -0.03],
                [0.99, 0.01, 0.11, 0.00],
                [0.98, -0.02, 0.00, 0.15],
            ],
            dtype=np.float64,
        )
        cloud = np.stack([sign_align(row, reference) for row in cloud])
        tangent, explained, _ = tangent_pca(cloud, reference)
        bases = build_neighborhood_bases(reference, tangent, [1, 2, 4])

        self.assertTrue(np.isclose(explained.sum(), 1.0))
        for width, basis in bases.items():
            self.assertEqual(basis.shape, (4, width))
            np.testing.assert_allclose(basis.T @ basis, np.eye(width), atol=1e-10)
            np.testing.assert_allclose(
                basis @ (basis.T @ reference), reference, atol=1e-10
            )
        np.testing.assert_allclose(
            bases[4] @ (bases[4].T @ bases[2]), bases[2], atol=1e-10
        )

    def test_subspace_predictive_capture_is_monotone_for_nested_bases(self):
        szz = np.diag([2.0, 1.0, 0.5])
        szy = np.array([[2.0, 0.0], [0.0, 1.0], [0.2, 0.1]])
        total_gain = float(
            np.trace(szy.T @ np.linalg.solve(szz + 1e-6 * np.eye(3), szy))
        )
        q1 = np.eye(3)[:, :1]
        q2 = np.eye(3)[:, :2]

        variance1, capture1 = subspace_statistics(
            q1, szz, szy, total_gain, ridge=1e-6
        )
        variance2, capture2 = subspace_statistics(
            q2, szz, szy, total_gain, ridge=1e-6
        )

        self.assertGreater(variance2, variance1)
        self.assertGreaterEqual(capture2, capture1)
        self.assertTrue(0.0 <= capture1 <= 1.0)
        self.assertTrue(0.0 <= capture2 <= 1.0)

    def test_contiguous_blocks_cover_every_origin_once(self):
        blocks = contiguous_origin_blocks(101, 16)
        self.assertEqual(blocks[0][0], 0)
        self.assertEqual(blocks[-1][1], 101)
        self.assertTrue(
            all(left[1] == right[0] for left, right in zip(blocks, blocks[1:]))
        )
        self.assertEqual(sum(stop - start for start, stop in blocks), 101)

    def test_synthetic_block_bootstrap_builds_requested_bases(self):
        rng = np.random.default_rng(7)
        segment = np.cumsum(rng.normal(size=(96, 3)), axis=0)
        szz_blocks, szy_blocks, pair_counts, _ = accumulate_block_moments(
            segment, seq_len=8, pred_len=4, n_blocks=8, chunk=16
        )
        szz, szy = weighted_moments(
            szz_blocks,
            szy_blocks,
            pair_counts,
            np.ones(len(pair_counts)),
        )
        reference = normalized_direction1(szz, szy, ridge=1e-6)
        cloud = bootstrap_direction_cloud(
            szz_blocks,
            szy_blocks,
            pair_counts,
            reference,
            replicates=16,
            ridge=1e-6,
            seed=11,
        )
        tangent, _, _ = tangent_pca(cloud, reference)
        bases = build_neighborhood_bases(reference, tangent, [1, 2, 4])

        self.assertEqual(cloud.shape, (16, 8))
        self.assertEqual(bases[4].shape, (8, 4))
        self.assertTrue(np.all(cloud @ reference >= 0))

    def test_runner_sweep_and_confirm_cells_are_fixed(self):
        sweep = sweep_cells()
        confirm = confirm_cells(
            {"ETTh2": 2, "ETTm2": 4, "Weather": 8, "Electricity": 1}
        )

        self.assertEqual(len(sweep), 42)
        self.assertEqual(
            {cell["dataset"] for cell in sweep},
            {"ETTh2", "ETTm2", "Weather", "Electricity"},
        )
        self.assertEqual(len(confirm), 54)
        self.assertEqual({cell["seed"] for cell in confirm}, {2022, 2023})
        self.assertEqual(
            {
                cell["width"]
                for cell in confirm
                if cell["dataset"] == "ETTm2"
                and cell["arm"].startswith("direction1_neighborhood")
            },
            {1, 4},
        )
        electricity_cones = [
            cell
            for cell in confirm
            if cell["dataset"] == "Electricity"
            and cell["arm"].startswith("direction1_neighborhood")
        ]
        self.assertEqual(len(electricity_cones), 2)
        self.assertEqual({cell["width"] for cell in electricity_cones}, {1})

    def test_confirm_matrix_size_depends_on_selected_width(self):
        all_rank_one = confirm_cells(
            {"ETTh2": 1, "ETTm2": 1, "Weather": 1, "Electricity": 1}
        )
        all_expanded = confirm_cells(
            {"ETTh2": 2, "ETTm2": 4, "Weather": 8, "Electricity": 2}
        )

        self.assertEqual(len(all_rank_one), 42)
        self.assertEqual(len(all_expanded), 56)

    def test_runner_command_installs_basis_and_reads_test(self):
        cell = {
            "dataset": "ETTm2",
            "horizon": 192,
            "seed": 2021,
            "width": 4,
            "arm": cone_arm(4),
        }
        command = arm_command(
            cell,
            "research_runs/example/projectors",
            "research_runs/example",
        )
        self.assertIn("--basis", command)
        self.assertIn(
            "research_runs/example/projectors/ETTm2_192_Qcone4.npy", command
        )
        self.assertIn("--evaluate-test", command)
        overrides = command[command.index("--overrides") + 1]
        self.assertIn("direction1_neighborhood_k4", overrides)

    def test_direct_command_reads_test_without_basis(self):
        command = arm_command(
            {
                "dataset": "Electricity",
                "horizon": 336,
                "seed": 2021,
                "width": None,
                "arm": "direct_nlinear",
            },
            "research_runs/example/projectors",
            "research_runs/example",
        )
        self.assertIn("--evaluate-test", command)
        self.assertNotIn("--basis", command)

    def test_test_selection_chooses_one_width_for_both_dataset_horizons(self):
        rows = {}
        for horizon in (96, 720):
            rows[("ETTh2", horizon, 2021, "direct_nlinear")] = {
                "test_mse": 1.0,
                "test_mae": 1.0,
            }
            rows[("ETTh2", horizon, 2021, "rrr_direction_1_2")] = {
                "test_mse": 1.1,
                "test_mae": 1.1,
            }
            for width, mse, mae in (
                (1, 1.08, 1.06),
                (2, 1.01, 1.02),
                (4, 1.0105, 1.01),
                (8, 1.03, 1.00),
            ):
                rows[("ETTh2", horizon, 2021, cone_arm(width))] = {
                    "test_mse": mse,
                    "test_mae": mae,
                }

        result = evaluate_dataset("ETTh2", [96, 720], rows)

        self.assertTrue(result["complete"])
        self.assertEqual(result["selected_width"], 4)
        self.assertEqual(result["selection_pool"], [2, 4])


if __name__ == "__main__":
    unittest.main()
