"""Unit tests for the low-rank checkpoint information analysis machinery.

These tests cover the parts that can be checked without the remote checkpoints:
the effective map of the factorized head, the semantic dictionaries, the
subspace utilities, the reduced-rank regression solvers and the inverse
problem used by the intervention analysis.
"""

import numpy as np
import torch
import unittest

from scripts.lowrank_checkpoint_core import (
    CenteredMoments,
    adaptive_avg_pool_operator,
    build_groups,
    covariance_correlation,
    effective_map,
    group_explanation,
    group_shapley,
    independent_rrr,
    input_templates,
    orthonormalize,
    orthogonal_projection,
    output_templates,
    principal_angles,
    projection_overlap,
    weighted_rrr_subspace,
)
from scripts.lowrank_checkpoint_inventory import (
    FORMAL_SEEDS,
    FORMAL_SETTINGS,
    LABEL_OF_RELATIVE_RANK,
    max_rank,
    nearest_cell,
    planned_rank,
    rank_ladder,
    rounding_rank,
)
from scripts.evaluate_lowrank_semantic_interventions import (
    arm_metrics,
    random_orthogonal_basis,
    semantic_basis,
)
from src.models.phase_adapters import PooledLowRankWeakPeriodResidualHead


class EffectiveMapTest(unittest.TestCase):
    def test_pooling_operator_matches_torch_for_identity_and_reduction(self):
        for seq_len, pooled_len in ((720, 720), (720, 360), (720, 41), (100, 7)):
            kernel = adaptive_avg_pool_operator(seq_len, pooled_len)
            self.assertEqual(kernel.shape, (pooled_len, seq_len))
            np.testing.assert_allclose(kernel.sum(axis=1), np.ones(pooled_len))
            sample = torch.randn(2, 3, seq_len, dtype=torch.float64)
            expected = torch.nn.functional.adaptive_avg_pool1d(sample, pooled_len)
            actual = np.einsum("ij,bcj->bci", kernel, sample.numpy())
            np.testing.assert_allclose(actual, expected.numpy(), atol=1e-12)

    def test_effective_map_reproduces_the_factorized_head(self):
        seq_len, pred_len, rank = 720, 96, 5
        head = PooledLowRankWeakPeriodResidualHead(
            seq_len, pred_len, pool_factor=1, rank=rank
        ).double()
        with torch.no_grad():
            head.encoder.weight.normal_(0.0, 0.02)
            head.encoder.bias.normal_(0.0, 0.01)
            head.decoder.weight.normal_(0.0, 0.02)
            head.decoder.bias.normal_(0.0, 0.01)
        matrix, _ = effective_map(
            head.encoder.weight.detach().numpy(),
            head.decoder.weight.detach().numpy(),
            head.pooled_len,
        )
        sample = torch.randn(4, seq_len, 3, dtype=torch.float64)
        with torch.no_grad():
            head_output = head(sample)
        last = sample[:, -1:, :]
        centered = (sample - last).permute(0, 2, 1).reshape(-1, seq_len)
        manual = (
            centered.numpy() @ matrix.T
            + head.encoder.bias.detach().numpy() @ head.decoder.weight.detach().numpy().T
            + head.decoder.bias.detach().numpy()
        ).reshape(4, 3, pred_len).transpose(0, 2, 1)
        expected = head_output.detach().numpy() - last.numpy()
        np.testing.assert_allclose(manual, expected, atol=1e-10)

    def test_effective_map_pooling_is_baked_into_the_operator(self):
        head = PooledLowRankWeakPeriodResidualHead(
            720, 96, pool_factor=1, rank=4
        ).double()
        matrix, _ = effective_map(
            head.encoder.weight.detach().numpy(),
            head.decoder.weight.detach().numpy(),
            head.pooled_len,
        )
        self.assertEqual(matrix.shape, (96, 720))


class SemanticDictionaryTest(unittest.TestCase):
    def test_input_dictionary_is_finite_and_does_not_center_templates(self):
        groups = build_groups(input_templates(720, 24))
        self.assertEqual(
            set(groups),
            {
                "recent_level",
                "level_change",
                "local_trend",
                "local_curvature",
                "period_level",
                "period_shape",
                "fast_local_change",
            },
        )
        recent = groups["recent_level"]
        names = {template.name for template in recent.templates}
        self.assertIn("tail_mean_6", names)
        self.assertIn("ema_tau168", names)
        tail_mean = next(
            template for template in recent.templates if template.name == "tail_mean_6"
        )
        # The non-zero sum of a recent-level template *is* the level semantics,
        # so it must survive normalization.
        self.assertGreater(float(tail_mean.vector.sum()), 0.0)
        for group in groups.values():
            self.assertGreaterEqual(group.dimension, 1)
            np.testing.assert_allclose(
                group.basis.T @ group.basis, np.eye(group.dimension), atol=1e-10
            )

    def test_output_dictionary_covers_displacement_and_periodic_shapes(self):
        groups = build_groups(output_templates(96, 24))
        self.assertIn("overall_displacement", groups)
        constant = groups["overall_displacement"].templates[0]
        np.testing.assert_allclose(constant.vector, np.ones(96) / np.sqrt(96))
        self.assertIn("slow_tilt", groups)
        self.assertIn("curvature", groups)
        self.assertIn("periodic", groups)

    def test_group_explanation_and_shapley_are_consistent(self):
        groups = build_groups(input_templates(720, 24))
        order = list(groups)
        direction = groups["recent_level"].basis[:, 0]
        shares = [group_explanation(direction, groups[name]) for name in order]
        # The groups overlap, so their projections are not a partition and the
        # shares are a lower bound that may exceed one; the owning group must
        # still be the largest and the exact explanation must be one.
        self.assertAlmostEqual(max(shares), 1.0, places=8)
        shapley = group_shapley(direction, groups, order)
        self.assertAlmostEqual(sum(shapley.values()), 1.0, places=8)
        # The group that owns the direction must dominate the attribution.
        self.assertEqual(max(shapley, key=shapley.get), "recent_level")
        _, r2 = orthogonal_projection(direction, groups["recent_level"].basis)
        self.assertAlmostEqual(r2, 1.0, places=10)


class SubspaceTest(unittest.TestCase):
    def test_projection_overlap_and_principal_angles(self):
        rng = np.random.default_rng(0)
        basis = orthonormalize(rng.standard_normal((3, 30)))
        self.assertAlmostEqual(projection_overlap(basis, basis), 1.0, places=12)
        np.testing.assert_allclose(principal_angles(basis, basis), 0.0, atol=1e-4)
        other = orthonormalize(np.eye(30)[3:6])
        self.assertLess(projection_overlap(basis, other), 1e-12)

    def test_covariance_correlation_is_scale_invariant(self):
        rng = np.random.default_rng(1)
        z = rng.standard_normal((400, 40))
        moments = CenteredMoments(
            mean=z.mean(axis=0), cov=np.cov(z, rowvar=False), n=z.shape[0]
        )
        direction = rng.standard_normal(40)
        template = rng.standard_normal(40)
        base = covariance_correlation(direction, template, moments)
        scaled = covariance_correlation(direction * 7.0, template * 0.3, moments)
        self.assertAlmostEqual(base, scaled, places=10)
        self.assertLessEqual(abs(base), 1.0)


class ReducedRankRegressionTest(unittest.TestCase):
    def test_independent_rrr_recovers_the_known_rank_one_direction(self):
        # z with two independent coordinates; the target depends on the first
        # only, so the leading RRR direction must be that coordinate.
        rng = np.random.default_rng(2)
        n = 4000
        z = np.zeros((n, 3))
        z[:, 0] = rng.standard_normal(n)
        z[:, 1] = rng.standard_normal(n)
        z[:, 2] = 0.05 * z[:, 0] + 0.02 * rng.standard_normal(n)
        y = 3.0 * z[:, 0][:, None] * np.ones((1, 2)) + 0.01 * rng.standard_normal((n, 2))
        szz = z.T @ z / n
        szy = z.T @ y / n
        basis, values = independent_rrr(szz, szy, 1, ridge=1e-9)
        direction = basis[:, 0] / np.linalg.norm(basis[:, 0])
        self.assertGreater(abs(float(direction[0])), 0.999)
        self.assertGreater(values[0], values[1])

    def test_weighted_rrr_ranks_by_weighted_energy(self):
        rng = np.random.default_rng(3)
        n = 2000
        z = rng.standard_normal((n, 5))
        weights = np.zeros(n)
        weights[: n // 2] = 1.0
        # Only the first coordinate matters in the weighted problem because the
        # second half of the samples has zero weight and carries the signal.
        z[n // 2 :, 0] = 0.0
        target = rng.standard_normal((n, 4))
        m_zz = (z * weights[:, None]).T @ z / n
        m_zy = (z * weights[:, None]).T @ target / n
        basis, values = weighted_rrr_subspace(m_zz, m_zy, 1)
        self.assertEqual(basis.shape, (5, 1))
        # The direction is determined up to sign, so the overlap with the first
        # coordinate is what has to be near one.
        first = np.zeros((5, 1))
        first[0, 0] = 1.0
        self.assertGreater(projection_overlap(basis, first), 0.9)
        self.assertGreater(values[0], values[1])

    def test_weighted_rrr_recovers_the_unweighted_solution(self):
        rng = np.random.default_rng(4)
        n = 1500
        z = rng.standard_normal((n, 4))
        target = z @ rng.standard_normal((4, 3)) + 0.1 * rng.standard_normal((n, 3))
        weights = np.ones(n)
        m_zz = z.T @ z / n
        m_zy = z.T @ target / n
        m_yy = target.T @ target / n
        basis, _ = weighted_rrr_subspace(m_zz, m_zy, 2)
        reference, _ = independent_rrr(m_zz, m_zy, 2, ridge=1e-12)
        self.assertAlmostEqual(projection_overlap(basis, reference), 1.0, places=6)
        del m_yy


class InterventionTest(unittest.TestCase):
    def test_random_basis_is_orthonormal_and_full_rank(self):
        rng = np.random.default_rng(5)
        basis = random_orthogonal_basis(9, 4, rng)
        self.assertEqual(basis.shape, (9, 4))
        np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-12)

    def test_arm_metrics_closed_form_matches_brute_force(self):
        rng = np.random.default_rng(6)
        samples, channels, rank, horizon = 12, 2, 3, 5
        hidden = rng.standard_normal((samples, channels, rank))
        decoder = rng.standard_normal((horizon, rank)) * 0.1
        bias = rng.standard_normal(horizon) * 0.05
        sigma = np.abs(rng.standard_normal((samples, 1, channels))) + 0.5
        gate = np.abs(rng.standard_normal((samples, 1, channels))) * 0.3
        phase = rng.standard_normal((samples, horizon, channels))
        target = rng.standard_normal((samples, horizon, channels))
        anchor = rng.standard_normal((samples, 1, channels))
        basis = orthonormalize(rng.standard_normal((rank, rank - 1)))

        def correction_of(state):
            return np.einsum("ncr,hr->nhc", state, decoder) * sigma

        reference = correction_of(hidden)
        metrics = arm_metrics(
            hidden, decoder, bias, gate, phase, target, reference, anchor,
            basis, "drop",
        )
        projected = np.einsum("ncr,rk->nck", hidden, basis)
        back = np.einsum("nck,rk->ncr", projected, basis)
        dropped = correction_of(hidden - back)
        branch = dropped + bias[None, :, None] + anchor
        fused = (1.0 - gate) * phase + gate * branch
        self.assertAlmostEqual(
            metrics["branch_mse"], float(np.mean((branch - target) ** 2)), places=10
        )
        self.assertAlmostEqual(
            metrics["fused_mse"], float(np.mean((fused - target) ** 2)), places=10
        )
        self.assertAlmostEqual(
            metrics["fused_mae"], float(np.mean(np.abs(fused - target))), places=10
        )
        self.assertAlmostEqual(
            metrics["branch_mae"], float(np.mean(np.abs(branch - target))), places=10
        )

    def test_only_plus_drop_reproduces_the_original_correction(self):
        rng = np.random.default_rng(7)
        hidden = rng.standard_normal((5, 3, 4))
        decoder = rng.standard_normal((6, 4)) * 0.1
        bias = np.zeros(6)
        sigma = np.ones((5, 1, 3))
        gate = np.full((5, 1, 3), 0.4)
        phase = np.zeros((5, 6, 3))
        target = np.zeros((5, 6, 3))
        anchor = np.zeros((5, 1, 3))
        basis = orthonormalize(rng.standard_normal((4, 2)))
        reference = np.einsum("ncr,hr->nhc", hidden, decoder) * sigma
        full = arm_metrics(
            hidden, decoder, bias, gate, phase, target, reference, anchor, None,
            "identity",
        )
        only = arm_metrics(
            hidden, decoder, bias, gate, phase, target, reference, anchor, basis,
            "only",
        )
        drop = arm_metrics(
            hidden, decoder, bias, gate, phase, target, reference, anchor, basis,
            "drop",
        )
        self.assertAlmostEqual(
            only["correction_energy"] + drop["correction_energy"],
            full["correction_energy"],
            places=10,
        )

    def test_semantic_basis_has_the_requested_truncation(self):
        basis = semantic_basis("ETTh2", 8)
        self.assertEqual(basis.shape[1], 8)
        np.testing.assert_allclose(basis.T @ basis, np.eye(8), atol=1e-10)


class RankLadderTest(unittest.TestCase):
    def test_planned_rank_matches_the_registered_plan_table(self):
        expected = {
            96: [24, 12, 6, 3],
            192: [48, 24, 12, 6],
            336: [84, 42, 21, 10],
            720: [180, 90, 45, 22],
        }
        for horizon, values in expected.items():
            actual = [
                planned_rank(1, q, horizon)
                for q in (0.25, 0.125, 0.0625, 0.03125)
            ]
            self.assertEqual(actual, values)

    def test_full_rank_cell_uses_the_factorization_limit(self):
        self.assertEqual(planned_rank(1, 1.0, 96), 96)
        self.assertEqual(max_rank(1, 336), 336)
        # The seed-2021 rounding variant is the one that differs, at horizon 720.
        self.assertEqual(rounding_rank(1, 0.03125, 720), 24)
        self.assertEqual(rounding_rank(1, 0.25, 720), 180)

    def test_rank_ladder_and_nearest_cell(self):
        ladder = rank_ladder(1, 720, [1.0, 0.25, 0.125, 0.0625, 0.03125])
        self.assertEqual(ladder[0.25], 180)
        self.assertEqual(ladder[1.0], 720)
        self.assertAlmostEqual(nearest_cell(0.0305556, [0.25, 0.125, 0.0625, 0.03125]), 0.03125)

    def test_formal_setting_and_seed_sets_match_the_plan(self):
        self.assertEqual(len(FORMAL_SETTINGS), 7)
        self.assertEqual(FORMAL_SEEDS, (2021, 2022, 2023))
        self.assertEqual(LABEL_OF_RELATIVE_RANK[0.25], "q=1/4")


if __name__ == "__main__":
    unittest.main()
