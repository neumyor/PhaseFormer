import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
import torch

from src.models.PhaseFormer import PhaseFormer
from src.models.phaseformer_presets import (
    ABLATION_MODES,
    PhaseFormerPresetConfig,
    build_hyperparams,
    make_exp_args,
)
from src.models.residual_topology import (
    AdditiveOutputResidualHead,
    LatentResidualPath,
    PhaseSlotResidualHead,
)
from scripts.run_residual_topology import (
    ALL_MODES,
    full_commands,
    screen_commands,
    summarize_full,
    summarize_screen,
)


def make_model(dataset="ETTh1", horizon=336, mode="original", seed=2021, **updates):
    hp = build_hyperparams(dataset, horizon, mode)
    hp.update(updates)
    pl.seed_everything(seed, workers=True)
    args = make_exp_args(dataset, 720, horizon, hp)
    config = PhaseFormerPresetConfig(args, 720, horizon, hp)
    return PhaseFormer(config)


def forward(model, x, marks):
    model.eval()
    with torch.no_grad():
        return model(x, marks, None, None)[0]


class ResidualTopologyModuleTests(unittest.TestCase):
    def test_additive_head_zero_init(self):
        head = AdditiveOutputResidualHead(12, 6)
        correction = head(torch.randn(2, 12, 4))
        torch.testing.assert_close(
            correction, torch.zeros_like(correction), atol=0, rtol=0
        )

    def test_latent_path_zero_init_and_depth(self):
        path = LatentResidualPath(latent_dim=8, num_injections=3)
        anchor = torch.randn(2, 4, 24, 8)
        self.assertEqual(len(path.projections), 3)
        for depth in range(3):
            correction = path(anchor, depth)
            torch.testing.assert_close(
                correction, torch.zeros_like(correction), atol=0, rtol=0
            )

    def test_phase_slot_head_zero_init_and_broadcast(self):
        head = PhaseSlotResidualHead(seq_len=12, num_periods=5, num_slots=24)
        x = torch.randn(2, 12, 4)
        out = head(x)
        self.assertEqual(tuple(out.shape), (2, 4, 24, 5))
        torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)

    def test_phase_slot_head_anchor_broadcast(self):
        head = PhaseSlotResidualHead(seq_len=12, num_periods=5, num_slots=24, anchor=True)
        x = torch.randn(2, 12, 4)
        last_period = torch.randn(2, 4, 5)
        out = head(x, last_period)
        self.assertEqual(tuple(out.shape), (2, 4, 24, 5))
        expected = last_period.unsqueeze(1).expand(-1, 24, -1, -1).permute(0, 2, 1, 3)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)


class PhaseFormerResidualTopologyTests(unittest.TestCase):
    MODES = [
        "residual_output_convex",
        "residual_output_additive",
        "residual_latent_long",
        "residual_latent_layerwise",
        "residual_hybrid",
        "residual_output_layerwise_convex",
        "residual_output_layerwise_additive",
    ]

    def test_modes_build_and_forward(self):
        x = torch.randn(2, 720, 7)
        marks = torch.rand(2, 720, 5)
        for mode in self.MODES:
            self.assertIn(mode, ABLATION_MODES)
            model = make_model(mode=mode)
            y = forward(model, x, marks)
            self.assertEqual(tuple(y.shape), (2, 336, 7), msg=mode)
            self.assertTrue(torch.isfinite(y).all(), msg=mode)

    def test_zero_init_topologies_match_original(self):
        x = torch.randn(2, 720, 7)
        marks = torch.rand(2, 720, 5)
        baseline = forward(make_model(), x, marks)
        for mode in [
            "residual_output_additive",
            "residual_latent_long",
            "residual_latent_layerwise",
            "residual_hybrid",
        ]:
            candidate = forward(make_model(mode=mode), x, marks)
            torch.testing.assert_close(
                candidate, baseline, atol=1e-6, rtol=1e-6, msg=mode
            )

    def test_layerwise_projection_count_matches_depth(self):
        model = make_model(dataset="ETTh1", horizon=336, mode="residual_latent_layerwise")
        self.assertEqual(model.phase_layers, 3)
        self.assertEqual(len(model.latent_residual_path.projections), 3)

    def test_one_layer_long_and_layerwise_are_equivalent(self):
        long_model = make_model(
            dataset="ETTh2", horizon=720, mode="residual_latent_long"
        )
        layer_model = make_model(
            dataset="ETTh2", horizon=720, mode="residual_latent_layerwise"
        )
        self.assertEqual(long_model.phase_layers, 1)
        weight = torch.randn_like(
            long_model.latent_residual_path.projections[0].weight
        )
        long_model.latent_residual_path.projections[0].weight.data.copy_(weight)
        layer_model.latent_residual_path.projections[0].weight.data.copy_(weight)
        x = torch.randn(2, 720, 7)
        marks = torch.rand(2, 720, 5)
        torch.testing.assert_close(
            forward(long_model, x, marks),
            forward(layer_model, x, marks),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_new_parameters_receive_gradients(self):
        model = make_model(mode="residual_hybrid")
        x = torch.randn(2, 720, 7)
        marks = torch.rand(2, 720, 5)
        y = model(x, marks, None, None)[0]
        y.square().mean().backward()
        self.assertIsNotNone(model.additive_output_residual.linear.weight.grad)
        for projection in model.latent_residual_path.projections:
            self.assertIsNotNone(projection.weight.grad)

    def test_zero_init_candidates_move_after_optimizer_step(self):
        x = torch.randn(2, 720, 7)
        marks = torch.rand(2, 720, 5)
        target = torch.randn(2, 336, 7)
        for mode in [
            "residual_output_additive",
            "residual_latent_long",
            "residual_latent_layerwise",
            "residual_hybrid",
        ]:
            model = make_model(mode=mode)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            before = model(x, marks, None, None)[0].detach().clone()
            loss = (model(x, marks, None, None)[0] - target).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            after = model(x, marks, None, None)[0].detach()
            self.assertTrue(torch.isfinite(after).all(), msg=mode)
            self.assertFalse(torch.equal(after, before), msg=mode)

    def test_high_dimensional_hybrid_shape(self):
        model = make_model(
            dataset="Electricity", horizon=336, mode="residual_hybrid"
        )
        x = torch.randn(1, 720, 321)
        marks = torch.rand(1, 720, 5)
        y = forward(model, x, marks)
        self.assertEqual(tuple(y.shape), (1, 336, 321))
        self.assertTrue(torch.isfinite(y).all())

    def test_feature_flags_do_not_shift_shared_initialization(self):
        baseline = make_model()
        prefixes = {
            "residual_output_convex": (
                "topology_output_convex_residual.",
                "topology_output_convex_gate",
            ),
            "residual_output_additive": (
                "additive_output_residual.",
                "additive_output_residual_gate",
            ),
            "residual_latent_long": ("latent_residual_path.",),
            "residual_latent_layerwise": ("latent_residual_path.",),
            "residual_hybrid": (
                "latent_residual_path.",
                "additive_output_residual.",
                "additive_output_residual_gate",
            ),
        }
        base_state = baseline.state_dict()
        for mode, owned_prefixes in prefixes.items():
            candidate = make_model(mode=mode)
            candidate_state = candidate.state_dict()
            shared = {
                key for key in candidate_state
                if not key.startswith(owned_prefixes)
            }
            self.assertEqual(shared, set(base_state), msg=mode)
            for key in shared:
                torch.testing.assert_close(
                    candidate_state[key], base_state[key], atol=0, rtol=0,
                    msg=f"{mode}: {key}",
                )

    def test_master_switch_disables_all_residual_topologies(self):
        model = make_model(mode="residual_hybrid", use_residual_head=False)
        self.assertFalse(model.use_additive_output_residual)
        self.assertFalse(model.use_topology_output_convex_residual)
        self.assertFalse(model.use_latent_long_residual)
        self.assertFalse(model.use_layerwise_latent_residual)
        self.assertFalse(hasattr(model, "additive_output_residual"))
        self.assertFalse(hasattr(model, "latent_residual_path"))

    def test_master_switch_disables_layerwise_output_modes(self):
        for mode in (
            "residual_output_layerwise_convex",
            "residual_output_layerwise_additive",
        ):
            model = make_model(mode=mode, use_residual_head=False)
            self.assertFalse(model.use_layerwise_output_convex, msg=mode)
            self.assertFalse(model.use_layerwise_output_additive, msg=mode)
            self.assertFalse(model.use_topology_output_convex_residual, msg=mode)
            self.assertFalse(model.use_additive_output_residual, msg=mode)
            self.assertIsNone(model.layerwise_convex_residual, msg=mode)
            self.assertIsNone(model.layerwise_additive_residual, msg=mode)

    def test_layerwise_intermediate_heads_only_on_multilayer(self):
        multi = make_model(dataset="ETTh1", horizon=336, mode="residual_output_layerwise_convex")
        self.assertEqual(multi.phase_layers, 3)
        self.assertEqual(len(multi.layerwise_convex_residual), 2)
        one = make_model(dataset="ETTh2", horizon=720, mode="residual_output_layerwise_convex")
        self.assertEqual(one.phase_layers, 1)
        self.assertIsNone(one.layerwise_convex_residual)
        self.assertIsNone(one.layerwise_convex_gates)

    def test_layerwise_convex_reduces_to_convex_on_one_layer(self):
        x = torch.randn(2, 720, 7)
        marks = torch.rand(2, 720, 5)
        base = forward(make_model(dataset="ETTh2", horizon=720, mode="residual_output_convex"), x, marks)
        layerwise = forward(
            make_model(dataset="ETTh2", horizon=720, mode="residual_output_layerwise_convex"),
            x, marks,
        )
        torch.testing.assert_close(layerwise, base, atol=0, rtol=0)

    def test_layerwise_additive_reduces_to_additive_on_one_layer(self):
        x = torch.randn(2, 720, 7)
        marks = torch.rand(2, 720, 5)
        base = forward(make_model(dataset="ETTh2", horizon=720, mode="residual_output_additive"), x, marks)
        layerwise = forward(
            make_model(dataset="ETTh2", horizon=720, mode="residual_output_layerwise_additive"),
            x, marks,
        )
        torch.testing.assert_close(layerwise, base, atol=0, rtol=0)

    def test_layerwise_additive_is_warm_start_on_multilayer(self):
        x = torch.randn(2, 720, 7)
        marks = torch.rand(2, 720, 5)
        baseline = forward(make_model(), x, marks)
        candidate = forward(
            make_model(mode="residual_output_layerwise_additive"), x, marks
        )
        torch.testing.assert_close(candidate, baseline, atol=1e-6, rtol=1e-6)

    def test_layerwise_convex_closed_gates_match_parent_on_multilayer(self):
        # Closing the intermediate convex gates makes the layerwise mode exactly
        # its single-point parent (R1) because the final fusion is identical.
        x = torch.randn(2, 720, 7)
        marks = torch.rand(2, 720, 5)
        parent = forward(make_model(mode="residual_output_convex"), x, marks)
        candidate = make_model(mode="residual_output_layerwise_convex")
        for gate in candidate.layerwise_convex_gates:
            gate.data.fill_(-50.0)  # sigmoid ~ 0 -> intermediate fusion = identity
        torch.testing.assert_close(
            forward(candidate, x, marks), parent, atol=1e-6, rtol=1e-6
        )

    def test_layerwise_additive_intermediate_head_moves_optimizer(self):
        # With an open gate the intermediate correction must actually change the
        # output and receive gradients, proving the path is not dead.
        x = torch.randn(2, 720, 7)
        marks = torch.rand(2, 720, 5)
        target = torch.randn(2, 336, 7)
        model = make_model(mode="residual_output_layerwise_additive")
        model.train()
        for gate in model.layerwise_additive_gates:
            gate.data.fill_(10.0)  # sigmoid ~ 1 -> correction fully applied
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model(x, marks, None, None)[0].detach().clone()
        loss = (model(x, marks, None, None)[0] - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        self.assertIsNotNone(model.layerwise_additive_residual[0].linear.weight.grad)
        self.assertIsNotNone(model.layerwise_additive_gates[0].grad)
        optimizer.step()
        after = model(x, marks, None, None)[0].detach()
        self.assertTrue(torch.isfinite(after).all())
        self.assertFalse(torch.equal(after, before))


class ResidualTopologyRunnerTests(unittest.TestCase):
    def test_dry_run_command_matrices_cover_plan(self):
        args = SimpleNamespace(
            settings="ETTh1:336,ETTh2:720,ETTm1:720,Electricity:336",
            modes=list(ALL_MODES),
            num_workers=0,
            output_dir="research_runs/example",
        )
        screen = screen_commands(args)
        self.assertEqual(len(screen), 32)
        self.assertTrue(all("--evaluate-test" not in cmd for cmd in screen))
        args.modes = ["original", "residual_output_additive"]
        full = full_commands(args)
        self.assertEqual(len(full), 4)
        self.assertTrue(all("benchmark_phaseformer_suite.py" in cmd[1] for cmd in full))

    def test_screen_and_full_summaries_compute_matched_deltas(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temp_dir:
            root = Path(temp_dir)
            screen_run = root / "screen" / "runs" / "one"
            screen_run.mkdir(parents=True)
            screen_fields = [
                "stage", "dataset", "horizon", "mechanism", "val_mae",
                "val_mse", "parameter_count", "elapsed_sec", "run_id",
            ]
            screen_rows = [
                {
                    "stage": "mechanism_screen_1", "dataset": "ETTh1",
                    "horizon": "336", "mechanism": "original",
                    "val_mae": "2.0", "val_mse": "4.0",
                    "parameter_count": "10", "elapsed_sec": "1", "run_id": "b",
                },
                {
                    "stage": "mechanism_screen_1", "dataset": "ETTh1",
                    "horizon": "336", "mechanism": "residual_output_additive",
                    "val_mae": "1.8", "val_mse": "3.6",
                    "parameter_count": "12", "elapsed_sec": "2", "run_id": "c",
                },
            ]
            with (screen_run / "metrics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=screen_fields)
                writer.writeheader()
                writer.writerows(screen_rows)
            summarize_screen(str(root / "screen"))
            with (root / "screen" / "screen_summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            candidate = next(r for r in rows if r["mechanism"] != "original")
            self.assertEqual(candidate["delta_mae_pct"], "10.0000")
            self.assertEqual(candidate["delta_mse_pct"], "10.0000")
            self.assertEqual(candidate["score"], "10.0000")

            full_run = root / "full" / "one"
            full_run.mkdir(parents=True)
            full_fields = [
                "dataset", "horizon", "mode", "test_mae", "test_mse",
                "elapsed_sec", "run_id",
            ]
            full_rows = [
                {
                    "dataset": "ETTh1", "horizon": "336", "mode": "original",
                    "test_mae": "2.0", "test_mse": "4.0",
                    "elapsed_sec": "1", "run_id": "b",
                },
                {
                    "dataset": "ETTh1", "horizon": "336",
                    "mode": "residual_output_additive", "test_mae": "1.8",
                    "test_mse": "3.6", "elapsed_sec": "2", "run_id": "c",
                },
            ]
            with (full_run / "metrics.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=full_fields)
                writer.writeheader()
                writer.writerows(full_rows)
            summarize_full(str(root / "full"))
            with (root / "full" / "full_summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            candidate = next(r for r in rows if r["mode"] != "original")
            self.assertEqual(candidate["delta_mae_pct"], "10.0000")
            self.assertEqual(candidate["delta_mse_pct"], "10.0000")


if __name__ == "__main__":
    unittest.main()
