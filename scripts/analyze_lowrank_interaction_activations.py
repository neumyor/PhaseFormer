#!/usr/bin/env python3
"""Capture validation activations for the corrected PhaseFormer-L interaction matrix.

The script only loads completed validation checkpoints and never constructs a
test loader.  It records one row per validation sample/channel with the
pre/post interaction latent norms, attention summaries, and paired forecast
errors.  A separate JSON summary is emitted so the large row table is not
needed for the first-pass audit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.data_factory import data_provider
from src.models.PhaseFormer import PhaseFormer
from src.models.phaseformer_presets import PhaseFormerPresetConfig, make_exp_args


ARMS = [
    "phase_only",
    "phaseformer_l_i0_late_fusion",
    "phaseformer_l_i1_phase_to_trend",
    "phaseformer_l_i2_trend_to_phase",
    "phaseformer_l_i3_parallel_bidir",
    "phaseformer_l_i6_alternating",
    "phaseformer_l_i7_joint",
]


def parse_setting(text: str) -> tuple[str, int, int]:
    dataset, horizon, seed = text.split(":")
    return dataset, int(horizon), int(seed)


def load_rows(source: Path) -> dict[tuple[str, int, int], dict[str, dict]]:
    grouped: dict[tuple[str, int, int], dict[str, dict]] = defaultdict(dict)
    for path in sorted((source / "search" / "runs").glob("*/metrics.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        row = frame.iloc[-1].to_dict()
        arm = str(row.get("mechanism", ""))
        if arm not in ARMS:
            continue
        config_path = path.parent / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        config["checkpoint"] = row["checkpoint"]
        key = (str(row["dataset"]), int(row["horizon"]), int(row["seed"]))
        grouped[key][arm] = {"row": row, "config": config}
    return grouped


def make_model(item: dict, device: torch.device):
    config = item["config"]
    hp = dict(config["hyperparams"])
    exp_args = make_exp_args(
        config["dataset"], config["lookback"], config["horizon"], hp,
        batch_size=config["batch_size"],
    )
    exp_args.dataset_args.percent = config["percent"]
    exp_args.dataset_args.num_workers = 2
    exp_args.dataset_args.input_hypothesis = config["input_hypothesis"]
    exp_args.dataset_args.input_variant = config["input_variant"]
    exp_args.dataset_args.input_period_len = config["period"]
    exp_args.dataset_args.intervention_seed = config["intervention_seed"]
    exp_args.dataset_args.input_d1_period = config["input_d1_period"]
    exp_args.dataset_args.input_d1_sigma = config["input_d1_sigma"]
    exp_args.dataset_args.input_d2_recent_length = config["input_d2_recent_length"]
    exp_args.dataset_args.input_d3_component = config["input_d3_component"]
    model = PhaseFormer(PhaseFormerPresetConfig(
        exp_args, config["lookback"], config["horizon"], hp
    ))
    checkpoint = Path(config["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, exp_args


def make_loader(item: dict):
    config = item["config"]
    hp = dict(config["hyperparams"])
    exp_args = make_exp_args(
        config["dataset"], config["lookback"], config["horizon"], hp,
        batch_size=config["batch_size"],
    )
    exp_args.dataset_args.percent = config["percent"]
    exp_args.dataset_args.num_workers = 2
    exp_args.dataset_args.input_hypothesis = config["input_hypothesis"]
    exp_args.dataset_args.input_variant = config["input_variant"]
    exp_args.dataset_args.input_period_len = config["period"]
    exp_args.dataset_args.intervention_seed = config["intervention_seed"]
    exp_args.dataset_args.input_d1_period = config["input_d1_period"]
    exp_args.dataset_args.input_d1_sigma = config["input_d1_sigma"]
    exp_args.dataset_args.input_d2_recent_length = config["input_d2_recent_length"]
    exp_args.dataset_args.input_d3_component = config["input_d3_component"]
    return data_provider(exp_args.dataset_args, "val")


def tensor_norm(x: torch.Tensor, dims) -> torch.Tensor:
    return torch.linalg.vector_norm(x, dim=dims)


def cosine(x: torch.Tensor, y: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.nn.functional.cosine_similarity(x, y, dim=dim, eps=1e-8)


def entropy(weights: torch.Tensor, dim: int = -1) -> torch.Tensor:
    p = weights.clamp_min(1e-8)
    return -(p * p.log()).sum(dim=dim)


def evaluate_arm(item: dict, loader, device: torch.device, mode: str):
    model, _ = make_model(item, device)
    branch = model.lowrank_interaction_branch
    hook_calls = {"phase_to_trend": [], "trend_to_phase": [], "joint": []}
    hooks = []

    def add_hook(name, module):
        if module is None:
            return

        def hook(_module, _inputs, output):
            weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            hook_calls[name].append(None if weights is None else weights.detach().cpu())

        hooks.append(module.register_forward_hook(hook))

    add_hook("phase_to_trend", getattr(branch, "phase_to_trend_attention", None))
    add_hook("trend_to_phase", getattr(branch, "trend_to_phase_attention", None))
    add_hook("joint", getattr(branch, "joint_attention", None))

    rows = []
    horizon = int(item["config"]["horizon"])
    batch_size = int(item["config"]["batch_size"])
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch = [x.to(device) if torch.is_tensor(x) else x for x in batch]
            x, y, x_mark, y_mark = batch
            dec = model._build_decoder_input(y.float())
            captured = {}

            def prehook(_module, args):
                captured["x"] = args[0].detach()
                captured["phase"] = args[1].detach()

            pre = branch.register_forward_pre_hook(prehook)
            out, _, _ = model(x.float(), x_mark.float(), dec, y_mark.float())
            pre.remove()
            pred = out[:, -horizon:, :]
            truth = y.float()[:, -horizon:, :]
            phase_pred = model.last_phase_forecast[:, -horizon:, :]
            lowrank_pred = model.last_lowrank_forecast[:, -horizon:, :]

            phase0 = captured["phase"]
            x_branch = captured["x"]
            lowrank0, _ = branch.lowrank_head.encode_latent(x_branch)
            # The branch diagnostics are exposed only through the model call;
            # execute the branch once more directly to obtain the exact tensors.
            forecast_unused, diag = branch(x_branch, phase0)
            trend0 = lowrank0.detach()
            trend1 = diag["rank_latent"].detach()
            phase1 = diag["updated_phase_latent"].detach()
            phase_attn = diag.get("phase_attention")
            joint_attn = diag.get("joint_attention")

            # The direct branch call is only for activations; prediction fields
            # above come from the model call and therefore retain full fusion.
            phase0_n = tensor_norm(phase0, (-1, -2))
            phase1_n = tensor_norm(phase1, (-1, -2))
            phase_delta_n = tensor_norm(phase1 - phase0, (-1, -2))
            trend0_n = tensor_norm(trend0, -1)
            trend1_n = tensor_norm(trend1, -1)
            trend_delta_n = tensor_norm(trend1 - trend0, -1)
            phase_cos = cosine(phase0, phase1, -1).mean(dim=-1)

            # Use the most recent module-hook weights from the branch call.
            pt_weights = hook_calls["phase_to_trend"][-1] if hook_calls["phase_to_trend"] else None
            tp_weights = hook_calls["trend_to_phase"][-1] if hook_calls["trend_to_phase"] else None
            joint_weights = hook_calls["joint"][-1] if hook_calls["joint"] else None
            if pt_weights is not None:
                pt_ent = entropy(pt_weights)
                pt_max = pt_weights.max(dim=-1).values
            else:
                pt_ent = pt_max = torch.zeros_like(trend0_n)
            if tp_weights is not None:
                tp_ent = entropy(tp_weights)
                tp_max = tp_weights.max(dim=-1).values
            else:
                tp_ent = tp_max = torch.zeros_like(trend0_n)
            if joint_weights is not None:
                # [B*C, S+1, S+1] -> trend-to-phase and phase-to-trend mass.
                jw = joint_weights
                jt_phase = jw[:, 0, 1:].mean(dim=-1)
                jp_trend = jw[:, 1:, 0].mean(dim=-1)
                jp_phase = jw[:, 1:, 1:].mean(dim=(-1, -2))
                jt_ent = entropy(jw[:, 0, 1:])
                jt_phase = jt_phase.reshape(x.shape[0], x.shape[2])
                jp_trend = jp_trend.reshape(x.shape[0], x.shape[2])
                jp_phase = jp_phase.reshape(x.shape[0], x.shape[2])
                jt_ent = jt_ent.reshape(x.shape[0], x.shape[2])
            else:
                jt_phase = jp_trend = jp_phase = jt_ent = torch.zeros_like(trend0_n)

            gate = torch.sigmoid(model.lowrank_interaction_gate).expand(x.shape[0], 1, -1)
            for bi in range(pred.shape[0]):
                for ci in range(pred.shape[2]):
                    pe = (phase_pred[bi, :, ci] - truth[bi, :, ci]).detach().cpu().numpy()
                    te = (lowrank_pred[bi, :, ci] - truth[bi, :, ci]).detach().cpu().numpy()
                    fe = (pred[bi, :, ci] - truth[bi, :, ci]).detach().cpu().numpy()
                    rows.append({
                        "mode": mode,
                        "sample_index": batch_index * batch_size + bi,
                        "channel": ci,
                        "fused_mae": float(np.abs(fe).mean()),
                        "fused_mse": float(np.square(fe).mean()),
                        "phase_mae": float(np.abs(pe).mean()),
                        "lowrank_mae": float(np.abs(te).mean()),
                        "gate": float(gate[bi, 0, ci].item()),
                        "trend0_norm": float(trend0_n[bi, ci].item()),
                        "trend1_norm": float(trend1_n[bi, ci].item()),
                        "trend_delta_norm": float(trend_delta_n[bi, ci].item()),
                        "phase0_norm": float(phase0_n[bi, ci].item()),
                        "phase1_norm": float(phase1_n[bi, ci].item()),
                        "phase_delta_norm": float(phase_delta_n[bi, ci].item()),
                        "phase_cosine": float(phase_cos[bi, ci].item()),
                        "pt_entropy": float(pt_ent.reshape(x.shape[0], x.shape[2])[bi, ci].item()),
                        "pt_max": float(pt_max.reshape(x.shape[0], x.shape[2])[bi, ci].item()),
                        "tp_entropy": float(tp_ent.reshape(x.shape[0], x.shape[2])[bi, ci].item()),
                        "tp_max": float(tp_max.reshape(x.shape[0], x.shape[2])[bi, ci].item()),
                        "joint_trend_to_phase_mass": float(jt_phase[bi, ci].item()),
                        "joint_phase_to_trend_mass": float(jp_trend[bi, ci].item()),
                        "joint_phase_to_phase_mass": float(jp_phase[bi, ci].item()),
                        "joint_trend_query_entropy": float(jt_ent[bi, ci].item()),
                    })
            # Keep only per-batch hook state; later summaries use the rows.
            for key in hook_calls:
                hook_calls[key].clear()
    for hook in hooks:
        hook.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--settings", required=True,
                        help="comma-separated dataset:horizon:seed entries")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    source = args.source_dir.resolve()
    grouped = load_rows(source)
    settings = [parse_setting(x) for x in args.settings.split(",")]
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    all_rows = []
    for setting in settings:
        if setting not in grouped:
            raise KeyError(f"setting not found: {setting}")
        missing = [arm for arm in ARMS if arm not in grouped[setting]]
        if missing:
            raise KeyError(f"{setting} missing arms: {missing}")
        _, loader = make_loader(grouped[setting]["phase_only"])
        # Phase-only is the matrix control but has no low-rank branch to
        # instrument.  The activation audit compares all interaction arms to
        # i0; phase-only errors remain available in the completed metrics.
        for arm in ARMS[1:]:
            print(f"[{args.gpu}] {setting} {arm}", flush=True)
            rows = evaluate_arm(grouped[setting][arm], loader, device, arm)
            for row in rows:
                row.update({"dataset": setting[0], "horizon": setting[1], "seed": setting[2]})
            all_rows.extend(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(args.output_dir / "activation_rows.csv", index=False)
    summary = []
    for (dataset, horizon, seed, mode), group in frame.groupby(
        ["dataset", "horizon", "seed", "mode"], sort=True
    ):
        summary.append({
            "dataset": dataset, "horizon": int(horizon), "seed": int(seed), "mode": mode,
            "rows": len(group),
            "fused_mae": float(group.fused_mae.mean()),
            "fused_mse": float(group.fused_mse.mean()),
            "trend_delta_norm": float(group.trend_delta_norm.mean()),
            "phase_delta_norm": float(group.phase_delta_norm.mean()),
            "phase_cosine": float(group.phase_cosine.mean()),
            "pt_entropy": float(group.pt_entropy.mean()),
            "pt_max": float(group.pt_max.mean()),
            "tp_entropy": float(group.tp_entropy.mean()),
            "tp_max": float(group.tp_max.mean()),
            "joint_trend_to_phase_mass": float(group.joint_trend_to_phase_mass.mean()),
            "joint_phase_to_trend_mass": float(group.joint_phase_to_trend_mass.mean()),
            "joint_phase_to_phase_mass": float(group.joint_phase_to_phase_mass.mean()),
            "joint_trend_query_entropy": float(group.joint_trend_query_entropy.mean()),
        })
    (args.output_dir / "activation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(json.dumps({"rows": len(frame), "output": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
