#!/usr/bin/env python3
"""Test whether the mapped encoder bias predicts E16's algebra-invariant failures.

Observed on 2026-09-20: cells whose closed-form untouched arm disagrees with the
model's own fused output report a relative gap that grows with
``mapped_encoder_bias_absmax`` (the magnitude of ``decoder_weight @ encoder_bias +
decoder_bias``), the quantity the tool's own error message points at.  Every dense
(``l_main``) cell passes, and the dense head has no encoder bias at all, so the
hypothesis is that this term is where the low-rank path goes wrong.

This probe computes the mapped bias for all 63 cells straight from the
checkpoints and prints it next to the already-observed outcome, so the hypothesis
is tested rather than asserted.  It runs no evaluation and touches no GPU.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import torch

REPO = pathlib.Path.home() / "niuyiming" / "PhaseFormer"
# The checkpoints were pickled with `src` importable, so unpickling needs the repo
# root on sys.path (the dissection tool does the same before loading them).
sys.path.insert(0, str(REPO))
PY = "/home/yyk/yyk03/miniconda3/envs/time/bin/python"
SETTINGS = [("ETTh2", 96), ("ETTh2", 720), ("ETTm2", 96), ("ETTm2", 192),
            ("Weather", 96), ("Weather", 192), ("Electricity", 336)]
ARMS = ("l_main", "l_q1_4", "l_q1_8")
SEEDS = (2021, 2022, 2023)

#: Shards that FAILED the algebra invariant, with the seed named in the error.
FAILED = {
    ("ETTh2", 96, "l_q1_4", 2022), ("ETTh2", 96, "l_q1_8", 2022),
    ("ETTh2", 720, "l_q1_4", 2021), ("ETTh2", 720, "l_q1_8", 2021),
    ("ETTm2", 96, "l_q1_4", 2021), ("ETTm2", 192, "l_q1_4", 2021),
    ("ETTm2", 192, "l_q1_8", 2022),
}
#: Shards that PASSED, so their seeds are known-good.
PASSED_SHARDS = {("ETTh2", 96, "l_main"), ("ETTh2", 720, "l_main"),
                 ("ETTm2", 96, "l_main"), ("ETTm2", 192, "l_main"),
                 ("Weather", 96, "l_main"), ("Weather", 192, "l_main"),
                 ("ETTm2", 96, "l_q1_8"), ("Weather", 96, "l_q1_4"),
                 ("Weather", 96, "l_q1_8"), ("Weather", 192, "l_q1_4"),
                 ("Weather", 192, "l_q1_8")}


def plan(dataset: str, horizon: int) -> list:
    result = subprocess.run(
        [PY, "scripts/phaseformer_L/e16_dissection.py", "--dry-run",
         "--e14-root", "research_runs/phaseformer_L_e14_main_v1",
         "--output-root", "/tmp/e16_bias_probe",
         "--datasets", dataset, "--horizons", str(horizon),
         "--seeds", ",".join(str(s) for s in SEEDS)],
        capture_output=True, text=True, cwd=REPO,
    )
    rows = []
    for line in result.stdout.splitlines():
        match = re.match(
            r"^(l_\w+)\s+(\S+)\s+seed=(\d+)\s+.*ckpt=\S+\s+(\S+/best\.ckpt)", line)
        if match:
            rows.append((match.group(1), int(match.group(3)), match.group(4)))
    return rows


def mapped_bias(path: str):
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    keys = [k for k in state if k.startswith("weak_period_residual.")]
    if any("linear.weight" in k for k in keys):
        return 0.0, "dense"          # dense head has no encoder bias by construction
    enc_b = state["weak_period_residual.encoder.bias"].double()
    dec_w = state["weak_period_residual.decoder.weight"].double()
    dec_b = state["weak_period_residual.decoder.bias"].double()
    return float(torch.abs(dec_w @ enc_b + dec_b).max()), "lowrank"


def main() -> int:
    rows = []
    for dataset, horizon in SETTINGS:
        for arm, seed, ckpt in plan(dataset, horizon):
            bias, kind = mapped_bias(ckpt)
            failed = (dataset, horizon, arm, seed) in FAILED
            known_pass = (dataset, horizon, arm) in PASSED_SHARDS and not failed
            rows.append((bias, dataset, horizon, arm, seed, kind, failed, known_pass))

    rows.sort(reverse=True)
    print(f"{'bias':>10}  {'cell':<28} {'kind':<8} outcome")
    for bias, dataset, horizon, arm, seed, kind, failed, known_pass in rows:
        if failed:
            outcome = "FAILED (observed)"
        elif known_pass:
            outcome = "passed (observed)"
        else:
            outcome = ""
        print(f"{bias:10.4f}  {dataset}-{horizon} {arm} s{seed}  {kind:<8} {outcome}")

    failures = [r for r in rows if r[6]]
    passes = [r for r in rows if r[7]]
    if failures and passes:
        lowest_failure = min(r[0] for r in failures)
        highest_pass = max(r[0] for r in passes)
        print(f"\nlowest observed failure bias : {lowest_failure:.4f}")
        print(f"highest observed passing bias: {highest_pass:.4f}")
        separable = highest_pass < lowest_failure
        print(f"bias separates failures from passes: {separable}")
        if separable:
            threshold = (highest_pass + lowest_failure) / 2
            predicted = [r for r in rows if r[0] > highest_pass and not r[7]]
            print(f"midpoint threshold {threshold:.4f} -> cells predicted to fail: "
                  f"{len(predicted)} of {len(rows)}")
            for bias, dataset, horizon, arm, seed, _kind, failed, known_pass in predicted:
                tag = "FAILED" if failed else "predicted-fail"
                print(f"   {bias:8.4f}  {dataset}-{horizon} {arm} s{seed}  {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
