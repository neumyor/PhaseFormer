#!/usr/bin/env python3
"""Install a Stage-0 frozen projector and launch the direction-retention runner.

This is a thin wrapper around ``scripts/search_phaseformer.py``: it imports that
module (so the training protocol, data split, hyperparameters, checkpoint
selection and metric code are byte-for-byte the runner's own), installs a frozen
``(seq_len, k)`` orthonormal basis into the NLinear branch right after the model
is constructed, and then calls the runner unmodified.

    --basis Q1.npy   -> variant V1 (direction 1 only)
    --basis Q12.npy  -> variant V2 (directions 1+2)

The projector is a frozen data artifact; it is never fitted or updated here.
Immediately after ``trainer.fit`` the wrapper also runs plan Stage-0 check 5 on
one real validation batch: the NLinear branch input under the installed
projector, under the identity projector, and through the head's own linear layer
must all differ — i.e. the projector is active and the variant is not secretly
the direct route.

Example::

    python scripts/run_top2_direction_retention.py \
        --dataset ETTh2 --horizon 96 --seed 2021 \
        --basis research_runs/top2_direction_retention_v1/projectors/ETTh2_96_Q1.npy \
        --output-dir research_runs/top2_direction_retention_v1/stage_a \
        --max-epochs 30 --require-cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.search_phaseformer as runner  # noqa: E402


_STATE: dict = {"tensor": None, "source": None, "audit_path": None}


def _load_basis(path: Path, seq_len: int) -> tuple[np.ndarray, str]:
    basis = np.load(path)
    if basis.ndim != 2 or basis.shape[0] != seq_len:
        raise ValueError(f"basis {path} must be ({seq_len}, k), got {basis.shape}")
    basis = np.ascontiguousarray(basis, dtype=np.float64)
    return basis, hashlib.sha256(basis.tobytes()).hexdigest()


def _projected_model_class(basis, source):
    """A PhaseFormer subclass whose NLinear branch is frozen onto ``basis``.

    ``PhaseFormer.__init__`` takes an optional ``projection_basis`` keyword, so
    the basis rides the real constructor instead of being injected afterwards.
    The subclass adds no parameters, so capacity is identical to the direct
    route; it only removes input information.
    """
    original = runner.PhaseFormer

    class ProjectedPhaseFormer(original):  # type: ignore[misc,valid-type]
        def __init__(self, configs):
            if os.environ.get("TOPDIR_DEBUG"):
                print(
                    json.dumps(
                        {
                            "event": "pre_construct",
                            "use_residual_head": getattr(
                                configs, "use_residual_head", "__absent__"
                            ),
                            "use_weak_period_residual": getattr(
                                configs, "use_weak_period_residual", "__absent__"
                            ),
                            "head_type": getattr(
                                configs, "weak_period_residual_head_type", "__absent__"
                            ),
                            "projection": getattr(
                                configs, "weak_residual_projection", "__absent__"
                            ),
                            "seq_len": getattr(configs, "seq_len", -1),
                        },
                        default=str,
                    ),
                    flush=True,
                )
            original.__init__(self, configs, projection_basis=basis)
            self.projection_basis_source = source
            print(
                json.dumps(
                    {
                        "event": "projection_installed",
                        "rank": int(
                            self.weak_period_residual.projection_basis.shape[1]
                        ),
                        "basis": source,
                        "sha256": _STATE["basis_sha256"],
                    }
                ),
                flush=True,
            )

    return ProjectedPhaseFormer


def _visibility_audit(model, loader) -> dict:
    """Plan Stage-0 check 5 on one real batch.

    Also reports, per retained direction, how much of the centered window's
    energy that single direction carries -- the empirical companion to the
    analytic eigengap record.
    """
    head = model.weak_period_residual
    batch = next(iter(loader))
    x = batch[0][:32].float()
    last = x[:, -1:, :]
    centered = (x - last).permute(0, 2, 1).contiguous()  # (B, C, L)
    basis = head.projection_basis
    identity = torch.eye(basis.shape[0], dtype=basis.dtype, device=basis.device)

    with torch.no_grad():
        projected = (centered @ basis) @ basis.transpose(0, 1)
        via_head = head.linear(projected)
        direct_head = head.linear(centered)
        total_energy = float(centered.pow(2).sum().item())
        # Leave-one-direction-out energy of the retained subspace.
        per_direction = []
        for index in range(basis.shape[1]):
            single = basis[:, index : index + 1]
            energy = float(
                ((centered @ single) @ single.transpose(0, 1)).pow(2).sum().item()
            )
            per_direction.append(energy / max(total_energy, 1e-12))
        # Streaming std of b_i^T z over the batch.
        feature_std = [
            float(((centered @ basis[:, i : i + 1]).flatten()).std().item())
            for i in range(basis.shape[1])
        ]
        return {
            "projection_rank": int(basis.shape[1]),
            "centered_vs_projected_max_abs": float(
                (centered - projected).abs().max().item()
            ),
            "head_input_differs_from_direct": not torch.allclose(
                projected, centered, atol=0.0
            ),
            "head_output_differs_from_direct": not torch.allclose(
                via_head, direct_head, atol=0.0
            ),
            "projected_rank_effective": float(
                torch.linalg.matrix_rank(projected[0].double()).item()
            ),
            "identity_rank_for_contrast": int(identity.shape[0]),
            "centered_energy_retained": float(
                projected.pow(2).sum().item() / max(total_energy, 1e-12)
            ),
            "per_direction_energy_share": per_direction,
            "feature_std": feature_std,
        }


def _patch_trainer_audit() -> None:
    """Run the visibility audit right after fit, before the wrapper returns."""
    original_build_trainer = runner.build_trainer

    def build_trainer(**kwargs):
        trainer, checkpoint = original_build_trainer(**kwargs)
        original_fit = trainer.fit

        def fit(model, *fit_args, **fit_kwargs):
            result = original_fit(model, *fit_args, **fit_kwargs)
            if _STATE["tensor"] is None:
                return result
            loaders = fit_kwargs.get("val_dataloaders") or fit_args[2:3]
            loader = loaders[0] if isinstance(loaders, (list, tuple)) else loaders
            try:
                audit = _visibility_audit(model, loader)
                audit["basis"] = _STATE["source"]
                audit["basis_sha256"] = _STATE["basis_sha256"]
                audit["library"] = "stage0-v1"
                print("TOPDIR_AUDIT " + json.dumps(audit, sort_keys=True), flush=True)
            except Exception as exc:  # pragma: no cover - audit must not be fatal
                print(
                    "TOPDIR_AUDIT " + json.dumps({"error": repr(exc)}), flush=True
                )
            return result

        trainer.fit = fit
        return trainer, checkpoint

    runner.build_trainer = build_trainer


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--basis", default="")
    parser.add_argument("--basis-sha256", default="")
    known, remainder = parser.parse_known_args()

    seq_len = 720
    if "--lookback" in remainder:
        seq_len = int(remainder[remainder.index("--lookback") + 1])

    if known.basis:
        basis_path = Path(known.basis)
        if not basis_path.is_absolute():
            basis_path = REPO_ROOT / basis_path
        if not basis_path.is_file():
            raise FileNotFoundError(f"projection basis not found: {basis_path}")
        basis, digest = _load_basis(basis_path, seq_len)
        if known.basis_sha256 and known.basis_sha256 != digest:
            raise RuntimeError(
                "projection basis sha256 mismatch: expected "
                f"{known.basis_sha256}, got {digest}"
            )
        _STATE["tensor"] = torch.as_tensor(basis, dtype=torch.float32)
        _STATE["source"] = str(basis_path)
        _STATE["basis_sha256"] = digest
        _STATE["audit_path"] = str(basis_path)
        runner.PhaseFormer = _projected_model_class(
            _STATE["tensor"], _STATE["source"]
        )
        _patch_trainer_audit()

    sys.argv = [sys.argv[0], *remainder]
    args = runner.parse_args()
    try:
        runner.execute(args)
    except Exception as exc:
        # Mirror the runner's own failure bookkeeping so a failed run still
        # leaves a status.json behind.
        try:
            spec = runner.build_spec(args)
            path = Path(args.output_dir) / "runs" / runner.run_id(spec)
            path.mkdir(parents=True, exist_ok=True)
            runner.atomic_json(
                path / "status.json",
                {"status": "failed", "failed_at": runner.utc_now(), "error": repr(exc)},
            )
        finally:
            raise


if __name__ == "__main__":
    main()
