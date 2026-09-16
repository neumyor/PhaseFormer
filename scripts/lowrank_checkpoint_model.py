#!/usr/bin/env python3
"""Model/dataset plumbing and intervention-aware forwards for the low-rank
checkpoint information analysis.

The plan (``docs/PhaseFormer_lowrank_checkpoint_information_analysis_plan.md``)
requires that every intervention touch *only* the private centered input of the
``weak_period_residual`` branch while the Phase backbone keeps reading the
original full input.  ``intervention_forward`` implements exactly that by
replacing the head's ``forward`` for one context and recomputing the output as
``decoder(intervened_hidden) + bias + x_last``, which is bit-identical to the
original head when no intervention is requested.

Nothing in this module trains a model or reads the test split.
"""

from __future__ import annotations

import contextlib
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.dataset.data_factory import data_provider
from src.models.PhaseFormer import PhaseFormer
from src.models.phase_adapters import PooledLowRankWeakPeriodResidualHead
from src.models.phaseformer_presets import PhaseFormerPresetConfig, make_exp_args

# Nominal dominant physical cycle in *steps* for the semantic templates.  The
# plan explicitly forbids writing ETTm2's 24 rows as one day; Electricity is an
# hourly series whose file has no timestamp column, so 24 is a nominal cycle and
# is recorded as such in the report.
DATASET_PERIOD_STEPS = {
    "ETTh2": 24,
    "ETTm2": 96,
    "Weather": 24,
    "Electricity": 24,
}

DATASET_KIND = {
    "ETTh2": "ett_hour",
    "ETTm2": "ett_minute",
    "Weather": "custom",
    "Electricity": "custom",
}


def resolve_root_path(exp_args, repo_root: Path) -> None:
    """Make ``dataset_args.root_path`` absolute relative to the repository."""
    root = Path(exp_args.dataset_args.root_path)
    if not root.is_absolute():
        exp_args.dataset_args.root_path = str(repo_root / root)
    fallback = repo_root / "resources" / "all_datasets" / "ETT-small"
    configured = Path(exp_args.dataset_args.root_path)
    if not configured.exists() and fallback.exists():
        exp_args.dataset_args.root_path = str(fallback)


def build_exp_args(dataset: str, lookback: int, horizon: int, hyperparams: dict, batch_size: int, repo_root: Path):
    exp_args = make_exp_args(dataset, lookback, horizon, hyperparams, batch_size=batch_size)
    exp_args.dataset_args.percent = int(hyperparams.get("percent", 100))
    exp_args.dataset_args.num_workers = 0
    exp_args.training_args.num_workers = 0
    resolve_root_path(exp_args, repo_root)
    return exp_args


def build_loaders(
    dataset: str,
    lookback: int,
    horizon: int,
    hyperparams: dict,
    batch_size: int,
    repo_root: Path,
    splits: tuple[str, ...] = ("train", "val"),
):
    """Return ``(exp_args, {"train": (set, loader), ...})`` with a model config."""
    exp_args = build_exp_args(dataset, lookback, horizon, hyperparams, batch_size, repo_root)
    handles = {}
    for split in splits:
        if split == "train":
            shuffle_set, shuffle_loader = data_provider(exp_args.dataset_args, "train")
            handles["train"] = (shuffle_set, shuffle_loader)
        else:
            data_set, loader = data_provider(exp_args.dataset_args, split)
            handles[split] = (data_set, loader)
    handle = handles.get("train") or handles["val"]
    if hasattr(handle[0], "data_stamp"):
        hyperparams["time_mark_dim"] = int(handle[0].data_stamp.shape[-1])
    return exp_args, handles


def build_model(exp_args, lookback: int, horizon: int, hyperparams: dict) -> PhaseFormer:
    return PhaseFormer(PhaseFormerPresetConfig(exp_args, lookback, horizon, hyperparams))


def load_checkpoint_into(model, checkpoint_path: Path) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "missing_keys": [key for key in missing],
        "unexpected_keys": [key for key in unexpected],
        "epoch": int(payload.get("epoch", -1)) if isinstance(payload, dict) else -1,
    }


@dataclass
class ForwardRecords:
    """Per-batch quantities recorded by :func:`intervention_forward`."""

    x_last: torch.Tensor  # (B, 1, C) normalized last history step
    z: torch.Tensor  # (B, 1, L, C) centered normalized history
    hidden: torch.Tensor  # (B, 1, r, C) head hidden state
    phase: torch.Tensor  # (B, 1, H, C) denormalized phase-only prediction
    residual: torch.Tensor  # (B, 1, H, C) denormalized low-rank prediction
    residual_norm: torch.Tensor  # (B, 1, H, C) normalized low-rank prediction
    mu: torch.Tensor  # (B, 1, C)
    sigma: torch.Tensor  # (B, 1, C)
    gate: torch.Tensor  # (B, 1, C)
    target: torch.Tensor  # (B, 1, H, C) ground truth
    fused: torch.Tensor  # (B, 1, H, C) denormalized fused output


def _residual_head(model):
    head = getattr(model, "weak_period_residual", None)
    if head is None:
        raise RuntimeError("model has no weak_period_residual head")
    return head


def intervention_forward(intervention, audit_math=None):
    """Build a replacement ``forward`` for the residual head.

    ``intervention`` maps ``(z, hidden)`` to an intervened ``hidden``; ``None``
    keeps the original value.  The replacement also records the head's private
    input, the hidden state, and the gate so that later analyses can work
    entirely from cached tensors.

    ``audit_math`` optionally carries the float64 effective map; when given, an
    exact float64 evaluation of the head is recorded alongside the float32 one.
    """

    def forward(self, x):  # noqa: D401 - mirrors the head signature
        last = x[:, -1:, :]
        centered = (x - last).permute(0, 2, 1).contiguous()
        if self.smooth_ratio:
            left = (self.smooth_window - 1) // 2
            right = self.smooth_window - 1 - left
            smoothed = torch.nn.functional.avg_pool1d(
                torch.nn.functional.pad(centered, (left, right), mode="replicate"),
                kernel_size=self.smooth_window,
                stride=1,
            )
            centered = (
                (1.0 - self.smooth_ratio) * centered
                + self.smooth_ratio * smoothed
            )
        pooled = torch.nn.functional.adaptive_avg_pool1d(centered, self.pooled_len)
        hidden = self.encoder(pooled)
        effective = hidden if intervention is None else intervention(centered, hidden)
        delta = self.decoder(effective).permute(0, 2, 1).contiguous()
        self.last_centered = centered
        self.last_pooled = pooled
        self.last_hidden = hidden
        self.last_hidden_used = effective
        self.last_forward_output = delta + last.expand(-1, self.pred_len, -1)
        if audit_math:
            print(
                "AUDIT SHAPES",
                tuple(centered.shape), tuple(pooled.shape), tuple(hidden.shape),
                tuple(self.encoder.weight.shape), tuple(self.encoder.bias.shape),
                tuple(self.decoder.weight.shape), tuple(self.decoder.bias.shape),
                tuple(last.shape), tuple(delta.shape),
                flush=True,
            )
            # Exact float64 evaluation of the *same* head applied to the *same*
            # private input.  The plan's 1e-6 equivalence bound cannot be met by
            # the in-model float32 matmul, which is TF32 on this platform
            # (``torch.set_float32_matmul_precision("medium")`` in the training
            # runner), so the audit uses this bit-faithful path and reports the
            # TF32 deviation separately.
            pooled64 = pooled.double()
            hidden64 = torch.nn.functional.linear(
                pooled64,
                self.encoder.weight.double(),
                self.encoder.bias.double(),
            )
            # All operands are float64 copies of the *checkpoint* tensors; the
            # composed map must be rebuilt here rather than reused from the
            # float32-effective matrix, because ``decoder @ encoder`` computed in
            # float32 and then widened to float64 carries a TF32-level rounding
            # error that would masquerade as an equivalence failure.
            decoder64 = self.decoder.weight.double()
            # The effective affine map of the branch is
            #     r(z) = (decoder @ encoder) z + decoder @ encoder.bias + decoder.bias
            # so the encoder bias has to be mapped through the decoder before it
            # can be compared with ``decoder(encoder(z))``.
            map64 = torch.nn.functional.linear(
                pooled64, decoder64 @ self.encoder.weight.double()
            ) + (decoder64 * self.encoder.bias.double()).sum(dim=1)[None, None, :]
            self.last_audit = {
                "pooled64": pooled64,
                "hidden64": hidden64,
                "map64": map64,
                # The branch's persistence anchor is the *uncentered* last step of
                # its normalized input; the last column of ``centered`` is zero by
                # construction, so it cannot be used here.
                "anchor64": last.permute(0, 2, 1).double(),
                "head64": (
                    torch.nn.functional.linear(
                        hidden64, decoder64, self.decoder.bias.double()
                    ).permute(0, 2, 1)
                    + last.permute(0, 2, 1).double()
                ),
                "shapes": (
                    tuple(pooled64.shape), tuple(hidden64.shape),
                    tuple(decoder64.shape), tuple(self.decoder.bias.shape),
                    tuple(last.shape),
                ),
                "fp32_head": delta + last.expand(-1, self.pred_len, -1),
            }
        return delta + last.expand(-1, self.pred_len, -1)

    return forward


def _capture_forward(module, original_forward):
    """Wrap the model's ``forward`` so the recorded quantities can be collected.

    The RevIN statistics are captured by temporarily replacing the bound
    ``normalize`` / ``normalize_with_stats`` callables of ``module.revin``;
    ``denormalize`` is called as a plain method, so a module hook would never
    fire for it.  The replacement is a pure wrapper: it returns exactly what the
    original returned and does not touch the numerical path.
    """

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        captured: dict = {"denorm": []}
        revin = self.revin
        original_normalize = revin.normalize
        original_with_stats = revin.normalize_with_stats
        original_denormalize = revin.denormalize

        def normalize(x):
            value, stats = original_normalize(x)
            captured["stats"] = stats
            return value, stats

        def normalize_with_stats(x, stats):
            captured["stats"] = stats
            return original_with_stats(x, stats)

        def denormalize(y, stats):
            # Recording the *inputs* of the denormalization gives the exact
            # normalized tensors the model wrote, which is the only way to state
            # the branch decomposition without re-deriving the pre-normalization
            # history.
            captured["denorm"].append((y.detach(), stats))
            return original_denormalize(y, stats)

        module.revin.normalize = normalize
        module.revin.normalize_with_stats = normalize_with_stats
        module.revin.denormalize = denormalize
        try:
            out = original_forward(
                self, x_enc, x_mark_enc, x_dec, x_mark_dec, *args, **kwargs
            )
        finally:
            module.revin.normalize = original_normalize
            module.revin.normalize_with_stats = original_with_stats
            module.revin.denormalize = original_denormalize

        gate = None
        if getattr(self, "weak_period_residual_gate", None) is not None:
            gate = torch.sigmoid(self.weak_period_residual_gate)
        elif getattr(self, "last_weak_residual_alpha", None) is not None:
            gate = self.last_weak_residual_alpha
        if captured.get("stats") is None:
            raise RuntimeError(
                "RevIN statistics were never produced; the analyzer requires "
                "use_revin=True on the audited checkpoints"
            )
        calls = captured["denorm"]
        phase_norm = calls[1][0] if len(calls) > 1 else None
        residual_norm = calls[2][0] if len(calls) > 2 else None
        head = getattr(self, "weak_period_residual", None)
        head_out = getattr(head, "last_forward_output", None) if head else None
        residual_consistency = None
        if head_out is not None and residual_norm is not None:
            residual_consistency = float(
                (head_out.float() - residual_norm).abs().max()
            )
        self.last_lowrank_records = {
            "stats": captured["stats"],
            "gate": gate,
            "hidden": getattr(self.weak_period_residual, "last_hidden", None),
            "z": getattr(self.weak_period_residual, "last_centered", None),
            "phase_norm": phase_norm,
            "residual_norm": residual_norm,
            "head_forward_output": head_out,
            "residual_consistency_max_abs": residual_consistency,
            "denormalize_calls": len(calls),
        }
        return out

    return forward


@contextlib.contextmanager
def instrument_model(model, intervention=None, audit_math=None):
    """Patch the residual head and the top-level forward for one context."""
    head = _residual_head(model)
    if not isinstance(head, PooledLowRankWeakPeriodResidualHead):
        raise TypeError(
            f"expected PooledLowRankWeakPeriodResidualHead, got {type(head).__name__}"
        )
    original_head_forward = head.forward
    original_model_forward = type(model).forward
    head.forward = types.MethodType(
        intervention_forward(intervention, audit_math), head
    )
    type(model).forward = _capture_forward(model, original_model_forward)
    try:
        yield model
    finally:
        head.forward = original_head_forward
        type(model).forward = original_model_forward
        for attribute in (
            "last_centered", "last_pooled", "last_hidden", "last_hidden_used",
            "last_audit", "last_forward_output",
        ):
            if hasattr(head, attribute):
                delattr(head, attribute)


def checkpoints_of(model) -> dict[str, torch.Tensor]:
    """Exact float64 copies of the head's parameters for offline analysis."""
    head = _residual_head(model)
    return {
        "encoder_weight": head.encoder.weight.detach().double().cpu().numpy().copy(),
        "encoder_bias": head.encoder.bias.detach().double().cpu().numpy().copy(),
        "decoder_weight": head.decoder.weight.detach().double().cpu().numpy().copy(),
        "decoder_bias": head.decoder.bias.detach().double().cpu().numpy().copy(),
        "pool_factor": int(head.pool_factor),
        "pooled_len": int(head.pooled_len),
        "rank": int(head.rank),
        "smooth_ratio": float(head.smooth_ratio),
        "gate": (
            torch.sigmoid(model.weak_period_residual_gate)
            .detach()
            .double()
            .cpu()
            .numpy()
            .copy()
            if getattr(model, "weak_period_residual_gate", None) is not None
            else None
        ),
    }


def head_linear_operator(param: dict) -> np.ndarray:
    """``M`` with ``delta = M @ z_flat + (decoder @ encoder.bias)``."""
    from scripts.lowrank_checkpoint_core import effective_map

    matrix, _ = effective_map(
        param["encoder_weight"], param["decoder_weight"], param["pooled_len"]
    )
    return matrix


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
