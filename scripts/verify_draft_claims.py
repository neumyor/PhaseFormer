#!/usr/bin/env python3
"""Verify the numeric claims in the draft findings before they enter the paper."""

from __future__ import annotations

import collections
import csv
import statistics as st
from pathlib import Path

BASE = Path("research_runs/lowrank_functional_rank_v1")


def read(name: str) -> list[dict]:
    with (BASE / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    semantics = read("mode_semantics.csv")
    cells = read("functional_rank_cells.csv")
    pruning = read("mode_pruning.csv")
    alignment = read("dense_alignment.csv")
    sparsity = read("mode_sparsity.csv")
    stability = read("seed_mode_stability.csv")

    print("=== (a) ETTh2-720 q=1/8 seed 2021: top-5 by contribution ===")
    rows = [
        r for r in semantics
        if r["setting"] == "ETTh2-720" and r["cell"] == "q=1/8" and r["seed"] == "2021"
    ]
    rows.sort(key=lambda r: int(r["contribution_rank"]))
    total = 0.0
    for r in rows[:5]:
        total += float(r["share_of_positive"])
        print(f"  mode {r['mode_index']:>3s} pos {int(r['contribution_rank'])} "
              f"share {float(r['share_of_positive']):.4f} "
              f"in {r['input_best_group']}({float(r['input_group_explanation']):.3f}) "
              f"out {r['output_best_group']}({float(r['output_group_explanation']):.3f})")
    print(f"  cumulative top-5 = {total:.4f}")

    print()
    print("=== (a2) beyond the top 8, how many modes are needed for 95%? ===")
    cell = next(r for r in cells if r["setting"] == "ETTh2-720"
                and r["cell"] == "q=1/8" and r["seed"] == "2021")
    print(f"  r95(contribution) = {cell['r95_contribution']}  r90 = {cell['r90_contribution']} "
          f"of rank {cell['rank']}")

    print()
    print("=== (b) ETTh2-720: dropping all negative-contribution modes ===")
    neg = [r for r in pruning if r["setting"] == "ETTh2-720"
           and r["criterion"] == "negative_contribution"]
    by_rank = collections.defaultdict(list)
    for r in neg:
        by_rank[int(r["rank"])].append((int(r["dropped"]), float(r["delta_fused_mse_vs_full"])))
    for rank in sorted(by_rank):
        values = by_rank[rank]
        print(f"  rank {rank:>3d}: dropped {st.mean(v[0] for v in values):5.1f} modes, "
              f"ΔMSE per seed = {[round(v[1], 6) for v in values]}, "
              f"mean {st.mean(v[1] for v in values):+.6f}")

    print()
    print("=== (c) ETTh2-720 negative counts and share ===")
    for rank in sorted({int(r["rank"]) for r in cells if r["setting"] == "ETTh2-720"}):
        values = [r for r in cells if r["setting"] == "ETTh2-720" and int(r["rank"]) == rank]
        counts = [int(v["n_negative_contribution"]) for v in values]
        print(f"  rank {rank:>3d}: {counts}  mean share {st.mean(c / rank for c in counts):.1%}")

    print()
    print("=== (d) q=1/8: r95, top-1 share, negative share per setting ===")
    for setting in ["ETTh2-96", "ETTh2-720", "ETTm2-96", "ETTm2-192", "Weather-96", "Weather-192"]:
        values = [r for r in cells if r["setting"] == setting and r["cell"] == "q=1/8"]
        r95 = [int(v["r95_contribution"]) for v in values]
        top1 = [float(v["contribution_share_of_top1"]) for v in values]
        neg = [int(v["n_negative_contribution"]) for v in values]
        rank = int(values[0]["rank"])
        print(f"  {setting:12s} rank {rank:>3d}  r95 {st.mean(r95):.1f}±{st.pstdev(r95):.1f}  "
              f"top1 {st.mean(top1):.1%}  neg {st.mean(neg):.1f}  "
              f"cells {[round(float(v['total_improvement']), 5) for v in values]}")

    print()
    print("=== (e) dense retention, q=1/8 ===")
    for setting in ["ETTh2-96", "ETTh2-720", "ETTm2-96", "ETTm2-192", "Weather-96", "Weather-192"]:
        values = [r for r in alignment if r["setting"] == setting and r["cell"] == "q=1/8"
                  and r["dense_reference"] == "input_subspace_restriction"]
        line = f"  {setting:12s}"
        for dim in (2, 4, 8):
            numbers = [float(v["dense_improvement_retained"]) for v in values
                       if int(v["subspace_dim"]) == dim]
            line += f"  dim{dim} {st.mean(numbers):.0%}" if numbers else f"  dim{dim} —"
        print(line)

    print()
    print("=== (f) input vs output overlap, q=1/8, dim 8 ===")
    for setting in ["ETTh2-96", "ETTh2-720", "ETTm2-96", "ETTm2-192", "Weather-96", "Weather-192"]:
        values = [r for r in alignment
                  if r["setting"] == setting and r["cell"] == "q=1/8"
                  and r["dense_reference"] == "dense_functional"
                  and int(r["subspace_dim"]) == 8]
        if values:
            print(f"  {setting:12s} input {st.mean(float(v['input_overlap']) for v in values):.3f}  "
                  f"output {st.mean(float(v['output_overlap']) for v in values):.3f}")

    print()
    print("=== (g) seed stability ===")
    print(f"  pairs {len(stability)}  input cos mean {st.mean(float(r['input_cosine_mean']) for r in stability):.3f}"
          f"  output cos mean {st.mean(float(r['output_cosine_mean']) for r in stability):.3f}"
          f"  frac matched>0.7 {st.mean(float(r['fraction_matched_above_0p7']) for r in stability):.3f}")
    for cell in ("q=1/4", "q=1/8", "q=1/16", "q=1/32"):
        values = [r for r in stability if r["cell"] == cell]
        print(f"    {cell:7s} input cos {st.mean(float(r['input_cosine_mean']) for r in values):.3f}")

    print()
    print("=== (h) sparsity ===")
    groups = collections.defaultdict(list)
    for row in sparsity:
        groups[row["variant"]].append(row)
    for name in sorted(groups):
        values = groups[name]
        print(f"  {name:24s} R2 {st.mean(float(v['reconstruction_r2']) for v in values):.3f}  "
              f"dMSE {st.mean(float(v['fused_mse_increase']) for v in values):+.2e}")


if __name__ == "__main__":
    main()
