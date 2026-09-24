#!/usr/bin/env python3
"""Per-mode semantic attribution for the canonical modes.

The functional-rank analysis answers "how many modes" and "how much does each
contribute", but names nothing: it never says *which* temporal component a mode
reads or writes.  This script produces that table.

For every formal cell it ranks the canonical modes by measured predictive
contribution ``I_i`` and attributes each of the top ones against the plan's
semantic dictionaries (section 4 of the checkpoint-information plan):

* the single best-matching template by ``|cos|``;
* the semantic *group* explanation, i.e. the share of the direction's L2 norm
  that lies inside that group's span -- reported for every group, because the
  groups overlap and a single best match hides the mixture;
* the dictionary R^2 against the union of all groups.

Group explanations are the honest quantity: several templates (EMA24, EMA48,
tail_mean_24) are near-collinear, so "best cosine" alone double-counts, whereas
the group span absorbs the collinearity.

Reads exported mode tensors only.  No model, no GPU, no test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if not (REPO_ROOT / "src").is_dir():
    REPO_ROOT = Path.cwd().resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lowrank_checkpoint_core import (  # noqa: E402
    build_groups,
    group_explanation,
    input_templates,
    orthogonal_projection,
    output_templates,
)

#: Dominant physical cycle in steps.  ETTm2 samples every 15 minutes, so 96 rows
#: are one day; writing 24 there would misname the period (plan section 4).
DATASET_PERIOD = {"ETTh2": 24, "ETTm2": 96, "Weather": 24}

INPUT_LABEL = {
    "recent_level": "recent level",
    "level_change": "level change",
    "local_trend": "local trend",
    "local_curvature": "local curvature",
    "period_level": "periodic level",
    "period_shape": "periodic shape",
    "fast_local_change": "fast local change",
}
OUTPUT_LABEL = {
    "overall_displacement": "displacement",
    "slow_tilt": "tilt",
    "curvature": "curvature",
    "periodic": "periodic correction",
    "recent_shape_continuation": "shape continuation",
}


def attribute(direction: np.ndarray, groups: dict, templates: dict) -> dict:
    """Best template, every group's explanation, and the dictionary R^2."""
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return {"best_template": "", "best_cosine": 0.0, "groups": {}, "dictionary_r2": 0.0}
    unit = direction / norm
    best_template, best_cosine = "", 0.0
    for group_name, entries in templates.items():
        for template in entries:
            cosine = abs(float(unit @ template.vector))
            if cosine > best_cosine:
                best_template, best_cosine = template.name, cosine
    explanations = {
        name: group_explanation(direction, group) for name, group in groups.items()
    }
    union = np.concatenate([group.basis for group in groups.values()], axis=1)
    _, dictionary_r2 = orthogonal_projection(direction, union)
    return {
        "best_template": best_template,
        "best_cosine": best_cosine,
        "groups": explanations,
        "dictionary_r2": dictionary_r2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes-dir", default="research_runs/lowrank_functional_rank_v1/modes")
    parser.add_argument("--cells-file", default="research_runs/lowrank_functional_rank_v1/functional_rank_cells.csv")
    parser.add_argument("--output-dir", default="research_runs/lowrank_functional_rank_v1")
    parser.add_argument("--settings", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--top-modes", type=int, default=8)
    parser.add_argument("--only-within-r95", action="store_true",
                        help="keep only modes inside the contribution-ordered r95 prefix")
    parser.add_argument("--repo-root", default="")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    modes_dir = repo_root / args.modes_dir
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = {}
    for row in csv.DictReader((repo_root / args.cells_file).open()):
        if row["dataset"] == "Electricity":
            continue
        cells[(row["setting"], row["seed"], row["cell"])] = row

    wanted_settings = {v for v in args.settings.split(",") if v}
    wanted_seeds = {v for v in args.seeds.split(",") if v}

    rows: list[dict] = []
    for (setting, seed, cell), cell_row in sorted(cells.items()):
        if wanted_settings and setting not in wanted_settings:
            continue
        if wanted_seeds and seed not in wanted_seeds:
            continue
        path = modes_dir / f"{setting}_seed{seed}_{cell.replace('/', '-')}.npz"
        if not path.is_file():
            continue
        payload = np.load(path)
        u, s, vt = payload["u"], payload["s"], payload["vt"]
        contribution = payload["contribution"]
        horizon = int(payload["horizon"])
        lookback = int(payload["lookback"])
        dataset = setting.rsplit("-", 1)[0]
        period = DATASET_PERIOD[dataset]

        in_templates = input_templates(lookback, period)
        out_templates = output_templates(horizon, period)
        in_groups = build_groups(in_templates)
        out_groups = build_groups(out_templates)

        order = np.argsort(-contribution, kind="stable")
        positive_total = float(contribution[contribution > 0].sum())
        net_total = float(contribution.sum())
        limit = int(cell_row["r95_contribution"]) if args.only_within_r95 else args.top_modes
        limit = max(limit, 1)

        for position, index in enumerate(order[:limit], start=1):
            input_stats = attribute(vt[index], in_groups, in_templates)
            output_stats = attribute(u[:, index], out_groups, out_templates)
            input_best_group = max(input_stats["groups"], key=input_stats["groups"].get)
            output_best_group = max(output_stats["groups"], key=output_stats["groups"].get)
            rows.append({
                "setting": setting, "dataset": dataset, "horizon": horizon,
                "seed": int(seed), "cell": cell, "rank": int(cell_row["rank"]),
                "mode_index": int(index),
                "contribution_rank": position,
                "singular_value": float(s[index]),
                "contribution": float(contribution[index]),
                "share_of_positive": (
                    float(contribution[index]) / positive_total if positive_total else float("nan")
                ),
                "share_of_net": (
                    float(contribution[index]) / net_total if net_total else float("nan")
                ),
                "input_best_template": input_stats["best_template"],
                "input_best_cosine": input_stats["best_cosine"],
                "input_best_group": INPUT_LABEL[input_best_group],
                "input_group_explanation": input_stats["groups"][input_best_group],
                "input_dictionary_r2": input_stats["dictionary_r2"],
                "input_group_explanations": json.dumps(
                    {INPUT_LABEL[k]: round(v, 4) for k, v in input_stats["groups"].items()}
                ),
                "output_best_template": output_stats["best_template"],
                "output_best_cosine": output_stats["best_cosine"],
                "output_best_group": OUTPUT_LABEL[output_best_group],
                "output_group_explanation": output_stats["groups"][output_best_group],
                "output_dictionary_r2": output_stats["dictionary_r2"],
                "output_group_explanations": json.dumps(
                    {OUTPUT_LABEL[k]: round(v, 4) for k, v in output_stats["groups"].items()}
                ),
                "paired_mechanism": (
                    f"{INPUT_LABEL[input_best_group]} -> {OUTPUT_LABEL[output_best_group]}"
                ),
            })
        print(f"  {setting} seed={seed} {cell}: {limit} modes attributed", flush=True)

    path = output_dir / "mode_semantics.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}", flush=True)


if __name__ == "__main__":
    main()
