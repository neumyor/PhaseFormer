#!/usr/bin/env python3
"""Diagnose E16's self-contradictory algebra failure on one real cell.

The failure says two things at once:

* the closed-form untouched arm and the model's own fused output differ by a
  relative fused-MSE gap of ~1e-3..1e-2 (tolerance 1e-3), and
* their element-wise RMS and max difference are exactly 0.000e+00.

Identical elements cannot produce different means, so either the two sides cover
different elements/counts, or the two MSEs are built from different
numerators/denominators.  Reading the code did not settle it, so this probe
MEASURES: it runs one failing cell with the accumulator methods wrapped, and
records, per call, which accumulator grew by how much.  Afterwards it prints a
per-method summary (calls, samples, pairs, elements, sums) plus the two MSEs the
invariant compares.

Nothing in the tool is modified: the methods are wrapped in this process only,
and the run goes to a scratch output root.
"""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path.home() / "niuyiming" / "PhaseFormer"
sys.path.insert(0, str(REPO))

import scripts.phaseformer_L.e16_dissection as E  # noqa: E402

LOG: list = []
_CALLS = {"statistics": 0, "arm": 0, "band": 0}
_SEEN: list = []


def _snapshot(acc):
    return {
        "pairs": getattr(acc, "pairs", 0),
        "elements": getattr(acc, "elements", 0),
        "recorded_fused_sq": getattr(acc, "recorded_fused_sq", 0.0),
        "algebra_samples": getattr(acc, "algebra_samples", 0),
        "algebra_sq": getattr(acc, "algebra_sq", 0.0),
        "algebra_absmax": getattr(acc, "algebra_absmax", 0.0),
    }


def wrap(name: str, original):
    def wrapped(self, *args, **kwargs):
        before = _snapshot(self)
        arm_rows_before = {
            key: dict(row) for key, row in getattr(self, "arm", {}).items()
        }
        result = original(self, *args, **kwargs)
        after = _snapshot(self)
        _CALLS[name] += 1
        row_delta = {}
        for key, row in getattr(self, "arm", {}).items():
            prior = arm_rows_before.get(key, {})
            row_delta[key] = {
                field: float(row.get(field, 0.0)) - float(prior.get(field, 0.0))
                for field in ("fused_sq", "fused_abs", "branch_sq", "recon_sq")
            }
        LOG.append({
            "method": name,
            "d_pairs": after["pairs"] - before["pairs"],
            "d_elements": after["elements"] - before["elements"],
            "d_recorded_fused_sq": after["recorded_fused_sq"] - before["recorded_fused_sq"],
            "d_algebra_samples": after["algebra_samples"] - before["algebra_samples"],
            "d_algebra_sq": after["algebra_sq"] - before["algebra_sq"],
            "row_delta": row_delta,
            "hidden_shape": getattr(self, "_diag_shape", None),
        })
        return result
    return wrapped


def main() -> int:
    E.CellAccumulator.add_statistics_block = wrap(
        "statistics", E.CellAccumulator.add_statistics_block)
    E.CellAccumulator.add_arm_block = wrap("arm", E.CellAccumulator.add_arm_block)
    E.CellAccumulator.add_band_block = wrap("band", E.CellAccumulator.add_band_block)

    sys.argv = [
        "e16_dissection.py", "--fast-einsum",
        "--e14-root", "research_runs/phaseformer_L_e14_main_v1",
        "--output-root", "/tmp/e16_diag_out",
        "--datasets", "ETTh2", "--horizons", "96", "--arms", "l_q1_4",
        "--seeds", "2022", "--gpus", "0",
        "--mem-budget-mb", "2048", "--num-workers", "1",
    ]
    try:
        E.main()
    except SystemExit as exc:
        print(f"\n[tool exited: {exc}]")

    stats = [entry for entry in LOG if entry["method"] == "statistics"]
    arms = [entry for entry in LOG if entry["method"] == "arm"]
    bands = [entry for entry in LOG if entry["method"] == "band"]

    def total(entries, field):
        return sum(entry[field] for entry in entries)

    print("\n================ diagnostic ================")
    print(f"calls: statistics={len(stats)} arm={len(arms)} band={len(bands)}")
    print(f"statistics: sum samples-pairs={total(stats, 'd_pairs')} "
          f"sum elements={total(stats, 'd_elements'):.0f} "
          f"sum recorded_fused_sq={total(stats, 'd_recorded_fused_sq'):.6e}")
    print(f"arm       : sum elements={total(arms, 'd_elements'):.0f} "
          f"sum algebra_samples={total(arms, 'd_algebra_samples'):.0f} "
          f"sum algebra_sq={total(arms, 'd_algebra_sq'):.6e} "
          f"max algebra_absmax={max((e['d_algebra_sq'] for e in arms), default=0):.3e}")

    # Per-identity-arm fused_sq accumulated by the arms pass, keyed by arm name.
    per_arm = {}
    for entry in arms:
        for key, delta in entry["row_delta"].items():
            slot = per_arm.setdefault(key, {"fused_sq": 0.0, "branch_sq": 0.0})
            slot["fused_sq"] += delta["fused_sq"]
            slot["branch_sq"] += delta["branch_sq"]
    print("\nper-arm accumulated fused_sq (numerator of baseline fused_mse):")
    for key in sorted(per_arm):
        print(f"   {key:<22} fused_sq={per_arm[key]['fused_sq']:.6e} "
              f"branch_sq={per_arm[key]['branch_sq']:.6e}")

    elements_stats = total(stats, "d_elements")
    elements_arms = total(arms, "d_elements")
    recorded = total(stats, "d_recorded_fused_sq")
    print(f"\nelements: statistics={elements_stats:.0f}  arm-pass={elements_arms:.0f}  "
          f"equal={elements_stats == elements_arms}")
    if elements_stats:
        print(f"model fused MSE      = {recorded / elements_stats:.10e}")
        for key in sorted(per_arm):
            if key.lower().startswith(("original", "identity", "untouched")):
                print(f"closed-form fused MSE= {per_arm[key]['fused_sq'] / elements_stats:.10e} "
                      f"({key})")
        for key in sorted(per_arm):
            mse = per_arm[key]["fused_sq"] / elements_stats
            rel = abs(mse - recorded / elements_stats) / max(abs(recorded / elements_stats), 1e-12)
            print(f"   arm {key:<22} rel gap vs model = {rel:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
