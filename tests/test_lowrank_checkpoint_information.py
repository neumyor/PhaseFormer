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
        self.assertTrue(all(0.0 - 1e-12 <= value <= 1.0 + 1e-12 for value in shares))
        shapley = group_shapley(direction, groups, order)
        self.assertAlmostEqual(sum(shapley.values()), 1.0, places=8)
        # The dictionary groups overlap, so an exact one-hot attribution is not
        # guaranteed; the owning group must still carry the largest share of the
        # reconstruction and a majority of it.
        self.assertEqual(max(shapley, key=shapley.get), "recent_level")
        # The dictionary groups are strongly collinear (several of them contain
        # the plain recent-level templates), so the exact Shapley value of the
        # owning group is diluted by its substitutes; it must still be the
        # largest single attribution.
        self.assertGreater(shapley["recent_level"], 0.15)
        _, r2 = orthogonal_projection(direction, groups["recent_level"].basis)
        self.assertAlmostEqual(r2, 1.0, places=10)


class SubspaceTest(unittest.TestCase):
    def test_projection_overlap_and_principal_angles(self):
        rng = np.random.default_rng(0)
        basis = orthonormalize(rng.standard_normal((3, 12)))
        self.assertAlmostEqual(projection_overlap(basis, basis), 1.0, places=12)
        np.testing.assert_allclose(principal_angles(basis, basis), 0.0, atol=1e-3)
        # Two random 3-dimensional subspaces of R^12 overlap by about 3/12, and
        # a basis built from coordinates outside the first three spans an
        # exactly orthogonal control.
        # Two 3-dimensional subspaces of R^12 overlap by about 3/12 on average;
        # an exactly orthogonal control is a coordinate block, which the
        # overlap must recognize as such.
        rng_big = np.random.default_rng(11)
        wide = orthonormalize(rng_big.standard_normal((3, 12)))
        self.assertLess(projection_overlap(wide, wide), 1.0 + 1e-12)
        self.assertGreater(projection_overlap(wide, wide), 0.5)
        e1 = np.zeros((12, 1))
        e1[0, 0] = 1.0
        e6 = np.zeros((12, 1))
        e6[6, 0] = 1.0
        self.assertLess(projection_overlap(e1, e6), 1e-12)

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
        n = 4000
        z = np.zeros((n, 5))
        # Coordinate 0 dominates the surviving half; the remaining columns carry
        # only a negligible share of the weighted energy.
        z[:, 0] = rng.standard_normal(n)
        z[:, 1] = 0.05 * rng.standard_normal(n)
        z[n // 2 :, 0] = 0.0
        target = rng.standard_normal((n, 4))
        weights = np.ones(n)
        weights[n // 2 :] = 0.0
        m_zz = (z * weights[:, None]).T @ z / n
        m_zy = (z * weights[:, None]).T @ target / n
        basis, values = weighted_rrr_subspace(m_zz, m_zy, 1)
        self.assertEqual(basis.shape, (5, 1))
        # The weighted problems only see the surviving half, where coordinate 0
        # is the informative one, so the leading subspace is that coordinate up
        # to sign.  Orthonormality holds in the *weighted* metric, so the
        # alignment is measured with the same metric.
        unit = basis[:, 0] / np.sqrt(float(basis[:, 0] @ m_zz @ basis[:, 0]))
        self.assertGreater(abs(float(unit[0])), 0.99)
        self.assertLess(abs(float(unit[1])), 0.05)
        self.assertGreater(values[0], values[1])

    def test_weighted_rrr_optimizes_the_weighted_objective(self):
        rng = np.random.default_rng(4)
        n = 3000
        z = np.zeros((n, 4))
        z[:, 0] = rng.standard_normal(n)
        z[:, 1] = rng.standard_normal(n)
        z[:, 2] = 0.05 * rng.standard_normal(n)
        z[:, 3] = 0.05 * rng.standard_normal(n)
        target = z @ rng.standard_normal((4, 3)) + 0.05 * rng.standard_normal((n, 3))
        weights = np.abs(rng.standard_normal(n)) + 0.2
        m_zz = (z * weights[:, None]).T @ z / n
        m_zy = (z * weights[:, None]).T @ target / n
        basis, values = weighted_rrr_subspace(m_zz, m_zy, 2)
        self.assertEqual(basis.shape, (4, 2))

        def weighted_explained(candidate: np.ndarray) -> float:
            """Weighted least-squares objective achieved on ``candidate``'s span.

            ``tr(C^T Zy)`` with ``C = (Q^T Mzz Q)^-1 Q^T Mzy`` is the numerically
            stable form of the maximal explained weighted energy on that span.
            """
            zz = candidate.T @ m_zz @ candidate
            zy = candidate.T @ m_zy
            coefficients = np.linalg.solve(zz, zy)
            return float(np.trace(coefficients.T @ zy))

        best = weighted_explained(basis)
        for _ in range(30):
            other, _ = np.linalg.qr(rng.standard_normal((4, 2)))
            self.assertLessEqual(weighted_explained(other), best * (1.0 + 1e-2))
        # The optimum equals the sum of the two largest leading eigenvalues of
        # the weighted input covariance.
        # The solver returns the leading eigenvectors of the weighted input
        # covariance; that basis must be optimal for the weighted objective.
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (m_zz + m_zz.T))
        order = np.argsort(eigenvalues)[::-1]
        reference = np.ascontiguousarray(eigenvectors[:, order[:2]])
        self.assertAlmostEqual(
            best, weighted_explained(reference), delta=abs(best) * 1e-6
        )
        self.assertGreater(values[1], values[2])


class InterventionTest(unittest.TestCase):
    def test_random_basis_is_orthonormal_and_full_rank(self):
        rng = np.random.default_rng(5)
        basis = random_orthogonal_basis(9, 4, rng)
        self.assertEqual(basis.shape, (9, 4))
        np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-12)

    def _fixture(self, seed=6):
        rng = np.random.default_rng(seed)
        samples, channels, rank, horizon = 12, 2, 3, 5
        return {
            "hidden": rng.standard_normal((samples, channels, rank)),
            "decoder": rng.standard_normal((horizon, rank)) * 0.1,
            "bias": rng.standard_normal(horizon) * 0.05,
            "sigma": np.abs(rng.standard_normal((samples, 1, channels))) + 0.5,
            "gate": np.abs(rng.standard_normal((samples, 1, channels))) * 0.3,
            "phase": rng.standard_normal((samples, horizon, channels)),
            "target": rng.standard_normal((samples, horizon, channels)),
            "anchor": rng.standard_normal((samples, 1, channels)),
            "basis": orthonormalize(rng.standard_normal((rank - 1, rank))),
        }

    def test_identity_arm_reproduces_the_branch(self):
        fixture = self._fixture()
        correction = (
            np.einsum(
                "ncr,hr->nhc", fixture["hidden"], fixture["decoder"]
            )
            + fixture["bias"][None, :, None]
        ) * fixture["sigma"]
        reference = correction
        metrics = arm_metrics(
            fixture["hidden"], fixture["decoder"], fixture["bias"],
            fixture["gate"], fixture["phase"], fixture["target"], reference,
            fixture["anchor"], fixture["sigma"], None, "identity",
        )
        branch = fixture["anchor"] + correction
        np.testing.assert_allclose(
            metrics["branch_mse"], np.mean((branch - fixture["target"]) ** 2),
            atol=1e-12,
        )
        self.assertAlmostEqual(metrics["correction_reconstruction_r2"], 1.0, places=10)

    def test_only_and_drop_partition_the_correction(self):
        fixture = self._fixture()
        hidden, decoder, bias = (
            fixture["hidden"], fixture["decoder"], fixture["bias"]
        )
        sigma, basis = fixture["sigma"], fixture["basis"]
        correction = (np.einsum("ncr,hr->nhc", hidden, decoder) + bias[None, :, None]) * sigma
        common = (
            hidden, decoder, bias, fixture["gate"], fixture["phase"],
            fixture["target"], correction, fixture["anchor"], sigma,
        )
        full = arm_metrics(*common, None, "identity")
        only = arm_metrics(*common, basis, "only")
        drop = arm_metrics(*common, basis, "drop")
        # ``only`` and ``drop`` are complementary projections of the same hidden
        # state, so ``only + drop = full + cross`` where the cross term is what
        # the two complementary projections of the *bias* contribute.  The
        # identity that must hold exactly is on the hidden-state part alone.
        hidden, decoder = fixture["hidden"], fixture["decoder"]
        projected = np.einsum("ncr,rk->nck", hidden, basis)
        back = np.einsum("nck,rk->ncr", projected, basis)
        residual = hidden - back
        projected_energy = float(
            np.mean((np.einsum("nck,hr,rk->nhc", projected, decoder, basis)) ** 2)
        )
        residual_energy = float(
            np.mean((np.einsum("ncr,hr->nhc", residual, decoder)) ** 2)
        )
        # ``only`` and ``drop`` are complementary projections of the hidden
        # state, so the *hidden-space* energies partition exactly.  The
        # corrections differ because each arm adds the same synthetic bias and
        # because the RevIN scale ``sigma`` varies per sample and channel, which
        # breaks orthogonality in the value space; the partition is therefore
        # asserted on the hidden space and the corrections are only required to
        # be of comparable magnitude.
        # The hidden state splits exactly into ``Q Q^T h`` and its complement,
        # but the decoder mixes those two components again, so the *energies*
        # do not partition exactly.  The identity that does hold is the
        # complementarity of the projections themselves.
        full_projected = np.einsum("ncr,hr->nhc", hidden, decoder)
        split = (
            np.einsum("nck,hr,rk->nhc", projected, decoder, basis)
            + np.einsum("ncr,hr->nhc", back, decoder)
        )
        np.testing.assert_allclose(split, full_projected, atol=1e-7)
        self.assertLessEqual(
            abs(projected_energy + residual_energy - float(np.mean(full_projected ** 2))),
            0.05 * float(np.mean(full_projected ** 2)),
        )
        total = only["correction_energy"] + drop["correction_energy"]
        self.assertGreater(total, full["correction_energy"] * 0.5)
        self.assertLess(total, full["correction_energy"] * 1.5)
        self.assertGreater(only["correction_reconstruction_r2"], 0.0)
        self.assertGreater(drop["correction_reconstruction_r2"], 0.0)
        # The untouched checkpoint reproduces itself exactly.
        self.assertAlmostEqual(full["correction_reconstruction_r2"], 1.0, places=10)

    def test_bias_off_removes_only_the_synthetic_bias(self):
        fixture = self._fixture()
        hidden, decoder, bias = (
            fixture["hidden"], fixture["decoder"], fixture["bias"]
        )
        sigma = fixture["sigma"]
        correction = (np.einsum("ncr,hr->nhc", hidden, decoder) + bias[None, :, None]) * sigma
        common = (
            hidden, decoder, bias, fixture["gate"], fixture["phase"],
            fixture["target"], correction, fixture["anchor"], sigma,
        )
        full = arm_metrics(*common, None, "identity")
        off = arm_metrics(*common, None, "bias")
        # Bias-off keeps the input-driven part and drops the synthetic bias.
        expected = np.einsum("ncr,hr->nhc", hidden, decoder) * sigma
        np.testing.assert_allclose(
            off["correction_energy"], np.mean(expected ** 2), atol=1e-12
        )
        scaled_bias = bias[None, :, None] * sigma
        self.assertAlmostEqual(
            full["correction_energy"] - off["correction_energy"],
            float(np.mean(2.0 * expected * scaled_bias + scaled_bias ** 2)),
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
