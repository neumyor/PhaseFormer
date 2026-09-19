#!/usr/bin/env python3
"""E16 - minipaper section 4.4: dissection of the trained head plus the
intervention table for PhaseFormer-L and the low-rank probe.

Produces the two section 4.4 tables of ``docs/PhaseFormer_L_minipaper.md`` for
the **seven test-selected settings x three seeds** of
``docs/PhaseFormer_L_execution_schedule.md`` section 2.1:

* ``dissection_table.csv`` -- one row per (setting, arm, seed): the leading
  canonical mode's input semantic group and explanation rate, its output group
  and explanation rate, its share of the correction energy, the cross-seed
  ``leading4`` subspace overlap, and the stable-semantics verdict;
* ``intervention_table.csv`` -- one row per (setting, arm, seed, intervention
  arm) carrying **both** the branch's own error and the fused error, the
  dimension bookkeeping of every control, and the null bands.

Three branch families are dissected (``--arms``):

=========  ==========================  ===============================
arm        head                        effective map
=========  ==========================  ===============================
``l_main`` dense ``shared`` NLinear    ``W`` (H x 720) itself
``l_q1_4`` ``pooled_lowrank`` q=1/4   ``W_dec @ W_enc`` (H x 720)
``l_q1_8`` ``pooled_lowrank`` q=1/8   ``W_dec @ W_enc`` (H x 720)
=========  ==========================  ===============================

The dense head is handled by the *same* algebra as the low-rank probe: with
``encoder = I``, ``encoder_bias = 0``, ``decoder = linear.weight`` and
``decoder_bias = linear.bias`` the branch's private input *is* its hidden state,
so the effective map, the canonical modes, the semantic attribution and every
intervention arm are the registered ones and no second code path is introduced.
``scripts/lowrank_checkpoint_core.effective_map`` supplies the spectrum, the
analyzer's ``align_direction`` / dictionary machinery the semantic columns and
``scripts/evaluate_lowrank_semantic_interventions`` the arm algebra and the
bands, so a probe row recomputed here is directly comparable with the already
filled ``research_runs/lowrank_checkpoint_information_v1`` tables.

Test split: **never read.**  ``--evaluation-split`` accepts only ``val`` /
``validation``, the validation loader is the only loader built, and the train
split is read through ``e15_dimension.load_split``, which parses the CSV with
``nrows = validation border`` -- rows at or after the test border are never
loaded.  No other file is opened, and no column of any product is a test
quantity.

Memory: the dense head's hidden state is the 720-wide centered input, so a
cell's ``(samples, channels, 720)`` float64 tensor is exactly the allocation
this script must not make (Electricity-336 needs ~2.9 GiB for the hidden state
alone; the 13.9 GiB ``(17344, 336, 321)`` allocation that OOM'd an earlier
attempt is larger again).  Everything is therefore streamed over
(batch x channel block), with

* ``--mem-budget-mb`` bounding the float64 block that is materialised -- the
  hidden block, its reference correction and, for the bands, the
  ``(samples, channels, rank, arms)`` contraction;
* ``random_drop_band``'s own ``block_elements`` bound on the band transient;
* the random bands drawn **once per cell** and replayed on every block, so the
  weighted average of the block means is each arm's own metric;
* scalar / small-matrix accumulators only: the mode energies are assembled from
  ``s_k^2 * mean(score_k^2 sigma^2)`` (algebraically identical to the analyzer's
  ``mean(contribution**2)``), the latent variances and the PCA controls from the
  streamed hidden second moment, and the arm metrics from per-block calls to the
  registered ``arm_metrics``.

Two validation passes are made per cell: pass 1 accumulates the statistics a
first-order basis needs (mode energies, the hidden second moment for the PCA
controls, the reference energy and the output-side residual moments), pass 2
evaluates every arm and band with all bases fixed.  A single pass would have to
retain the per-sample hidden state, which is the allocation the budget forbids.

Usage (the real run, after E14 has written ``stage_a_manifest.json``)::

    CUDA_VISIBLE_DEVICES=0 python scripts/phaseformer_L/e16_dissection.py \
        --output-root research_runs/phaseformer_L_e16_dissection_v1 \
        --e14-root research_runs/phaseformer_L_e14_main_v1 \
        --num-workers 4 --mem-budget-mb 512

    # stage 2/3 of the six-stage flow: resolve every cell, run nothing
    python scripts/phaseformer_L/e16_dissection.py --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_lowrank_checkpoint_information import (  # noqa: E402
    INPUT_GROUP_LABELS,
    MECHANISM_LABELS,
    OUTPUT_GROUP_LABELS,
    SettingDictionary,
    align_direction,
    matrix_rank_tolerance,
    principal_angle_gap,
)
from scripts.evaluate_lowrank_semantic_interventions import (  # noqa: E402
    RANDOM_SEED,
    affine_bias,
    arm_metrics,
    band_summary,
    latent_image,
    legacy_random_band_dimension,
    random_drop_band,
    random_orthogonal_basis,
    random_rrr_basis,
    semantic_basis,
)
from scripts.lowrank_checkpoint_core import (  # noqa: E402
    CenteredMoments,
    build_groups,
    effective_map,
    independent_rrr,
    input_templates,
    orthonormality_error,
    output_templates,
    projection_overlap,
)
from scripts.lowrank_checkpoint_model import (  # noqa: E402
    DATASET_PERIOD_STEPS,
    _capture_forward,
    build_loaders,
    build_model,
    intervention_forward,
    set_seed,
)
from scripts.phaseformer_L.e14_main_matrix import (  # noqa: E402
    SEEDS as E14_SEEDS,
)
from scripts.phaseformer_L.e14_main_matrix import _arm_match  # noqa: E402
from scripts.phaseformer_L.e15_dimension import (  # noqa: E402
    build_registry,
    load_dataset_info,
    load_split,
    resolve_csv_path,
)
from src.dataset.data_factory import data_provider  # noqa: E402
from src.models.phase_adapters import (  # noqa: E402
    PooledLowRankWeakPeriodResidualHead,
    WeakPeriodResidualHead,
)

# ---------------------------------------------------------------------------
# Protocol constants (identical to E14 and to the low-rank analysis)
# ---------------------------------------------------------------------------
LOOKBACK = 720
PERIOD = 24
REVIN_EPS = 1e-5  # src/models/phase_adapters.py::RevIN.normalize defaults

# Draw streams.  They are chosen so that a probe cell's *registered* bands are
# bit-identical to the ones ``evaluate_lowrank_semantic_interventions`` writes
# for the same checkpoint: the legacy whole-latent-space band uses
# ``RANDOM_SEED``, the RandomRRR band ``RANDOM_SEED + 1`` and the arm's own
# random reference draw ``RANDOM_SEED + 2``.
BAND_SEED_AMBIENT_MATCHED = RANDOM_SEED + 3
# Placeholder generator for calls that pass pre-drawn ``bases`` (never consumed).
_PLACEHOLDER_RNG = np.random.default_rng(0)

# The seven test-selected settings of the schedule (section 2.1).  They are
# disclosed as test-set selection wherever they appear; E16 reports
# train/validation quantities only.
TEST_SELECTED_SETTINGS = (
    ("ETTh2", 96),
    ("ETTh2", 720),
    ("ETTm2", 96),
    ("ETTm2", 192),
    ("Weather", 96),
    ("Weather", 192),
    ("Electricity", 336),
)

# E14 arm -> dissectable head.  ``phase_only`` has no residual branch, and
# ``l_rcrf``/``a1`` are deliberately absent: the rcrf arm fuses through a
# reliability map instead of the plain two-way gate this algebra assumes, and
# ``a1`` has no local artifacts at all.
SUPPORTED_ARMS = {
    "l_main": {"head": "dense", "rank_div": None, "q_label": "direct"},
    "l_q1_4": {"head": "pooled_lowrank", "rank_div": 4, "q_label": "q=1/4"},
    "l_q1_8": {"head": "pooled_lowrank", "rank_div": 8, "q_label": "q=1/8"},
}
DEFAULT_ARMS = ("l_main", "l_q1_4", "l_q1_8")

# Section 6.1 of the low-rank plan plus the control minipaper section 4.4 adds.
CRITERION_INPUT_EXPLANATION = 0.50
CRITERION_OUTPUT_EXPLANATION = 0.80
CRITERION_SUFFICIENT_TOLERANCE = 0.005  # +0.5 % on both fused metrics
CRITERION_SEED_MAJORITY = 2  # out of three seeds


def parse_list(raw, cast=str) -> list:
    return [cast(item) for item in str(raw).split(",") if str(item).strip()]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="e16_dissection.py",
        description=(
            "E16 / minipaper 4.4: dissect the PhaseFormer-L dense head and the "
            "low-rank probe (7 test-selected settings x 3 seeds) and evaluate "
            "the intervention arms, including the new RandomRRR-drop control. "
            "Reads the validation split only; the test split is not offered."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-root",
        default="research_runs/phaseformer_L_e16_dissection_v1",
        help="single root for every product of this unit (the schedule forbids "
        "sharing an output root between E units)",
    )
    parser.add_argument(
        "--e14-root",
        default="research_runs/phaseformer_L_e14_main_v1",
        help="E14 stage-A output root holding stage_a_manifest.json and runs/",
    )
    parser.add_argument(
        "--checkpoints",
        default="",
        help="optional explicit comma-separated checkpoint list; every file must "
        "live inside a run directory with config.json, which then supplies "
        "dataset/horizon/seed/arm.  Overrides the --e14-root resolution",
    )
    parser.add_argument(
        "--arms",
        default=",".join(DEFAULT_ARMS),
        help="E14 arm rows to dissect; only l_main / l_q1_4 / l_q1_8 describe a "
        "linear residual branch",
    )
    parser.add_argument(
        "--datasets", default="",
        help="default: the datasets of the 7 test-selected settings",
    )
    parser.add_argument(
        "--horizons", default="",
        help="default: the horizons of the 7 test-selected settings",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in E14_SEEDS),
        help="training seeds; the cross-seed column needs at least two",
    )
    parser.add_argument(
        "--evaluation-split",
        default="val",
        choices=("val", "validation"),
        help="only the validation split exists for this unit; the test split is "
        "deliberately not selectable",
    )
    parser.add_argument(
        "--random-repeats", type=int, default=100,
        help="draws of each whole-space band (legacy dimension and the "
        "Semantic-drop dimension)",
    )
    parser.add_argument(
        "--random-rrr-repeats", type=int, default=100,
        help="draws of the RandomRRR-drop band (random subspaces of the "
        "RRR-achievable subspace at the Semantic-drop dimension)",
    )
    parser.add_argument(
        "--random-rrr", dest="random_rrr", action="store_true", default=True,
        help="evaluate the RandomRRR-drop arm and band (default: on)",
    )
    parser.add_argument(
        "--no-random-rrr", dest="random_rrr", action="store_false",
        help="disable the RandomRRR-drop arm and band",
    )
    parser.add_argument(
        "--rrr-pool-space", default="revin", choices=("revin", "standardized"),
        help="space of the pool E16 fits itself; 'revin' reproduces "
        "scripts/compute_phase_conditional_rrr.py (per-window RevIN "
        "normalization, i.e. a 1/sigma^2 weighted fit of the standardized "
        "windows), which is also the space of the Stage 3 subspace files",
    )
    parser.add_argument("--semantic-rank", type=int, default=8,
                        help="dimension of the Semantic8 arms (plan section 4)")
    parser.add_argument("--modes", type=int, default=8,
                        help="canonical modes reported per checkpoint")
    parser.add_argument(
        "--mem-budget-mb", type=float, default=512.0,
        help="budget for the float64 block materialised per (batch x channel "
        "block) and for the random-band transient",
    )
    parser.add_argument(
        "--channel-block", type=int, default=0,
        help="force the channel block size (0 = derive it from --mem-budget-mb)",
    )
    parser.add_argument(
        "--max-batches", type=int, default=0,
        help="cap the validation batches per pass (0 = all); a smoke/debug knob "
        "that changes every number it touches",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpus", default="0")
    parser.add_argument(
        "--data-root", default="",
        help="optional override root replacing the leading 'all_datasets/' part "
        "of the registered dataset root (empty: use the registered path)",
    )
    parser.add_argument(
        "--reference-root",
        default="research_runs/lowrank_checkpoint_information_v1",
        help="registered low-rank analysis root, used for the parity report and "
        "for the Stage 3 subspace files of the probe cells",
    )
    parser.add_argument("--skip-reference-parity", action="store_true")
    parser.add_argument(
        "--allow-missing-cells", action="store_true",
        help="continue when a wanted cell has no checkpoint yet (default: fail, "
        "so a partially trained E14 cannot silently shrink the table)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="resolve every cell and print the plan, run nothing",
    )
    return parser


# ---------------------------------------------------------------------------
# Block sizing (memory budget)
# ---------------------------------------------------------------------------
def resolve_block_sizes(
    budget_mb: float,
    batch_samples: int,
    channels: int,
    rank_dim: int,
    horizon: int,
    band_repeats: int,
    forced_channel_block: int = 0,
) -> tuple[int, int]:
    """``(channel_block, band_block_elements)`` for one batch size.

    Two transients have to fit the budget:

    * the statistics/arm block, ``(batch, channel_block, rank_dim)`` float64 plus
      its reference correction and error tensors, over-estimated here as
      ``2 * rank_dim + 4 * horizon`` float64 per (sample, channel) pair;
    * the band contraction, whose dominant term is
      ``(batch, channel_block, rank, arms)``, i.e.
      ``max(rank_dim, horizon) * repeats`` float64 per pair.

    ``band_block_elements`` is handed to :func:`random_drop_band` so that its own
    ``(samples, horizon, channels, arms)`` block stays bounded as well.
    """
    budget_bytes = max(1.0, float(budget_mb)) * 1024 * 1024
    if forced_channel_block > 0:
        return max(1, min(int(forced_channel_block), channels)), max(
            1, int(budget_bytes * 0.25 // 8)
        )
    batch_samples = max(1, int(batch_samples))
    statistics_pair = max(1, 2 * int(rank_dim) + 4 * int(horizon)) * 8
    band_pair = max(1, max(int(rank_dim), int(horizon)) * max(1, int(band_repeats))) * 8
    channel_block = min(
        channels,
        max(1, int(budget_bytes * 0.5 // (batch_samples * statistics_pair))),
        max(1, int(budget_bytes * 0.5 // (batch_samples * band_pair))),
    )
    return max(1, channel_block), max(1, int(budget_bytes * 0.25 // 8))


def iter_channel_blocks(channels: int, block: int):
    step = max(1, int(block))
    for start in range(0, int(channels), step):
        yield start, min(start + step, int(channels))


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------
def load_e14_manifest(e14_root: Path) -> dict:
    path = e14_root / "stage_a_manifest.json"
    if not path.is_file():
        raise SystemExit(
            f"E14 manifest not found: {path}\n"
            "E16 depends on E14 (schedule section 4.1).  Run "
            "scripts/phaseformer_L/e14_main_matrix.py --stage a first, or point "
            "--e14-root at the root that holds stage_a_manifest.json."
        )
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f"cannot parse {path}: {error}") from error
    if not isinstance(manifest.get("cells"), list):
        raise SystemExit(f"{path} has no 'cells' list")
    return manifest


def checkpoint_in_run_dir(run_dir: Path) -> tuple[Path | None, str]:
    """Best-validation checkpoint of a run, and the rule that found it.

    ``metrics.csv`` records the repo-relative path of the checkpoint the run
    itself restored for its validation pass, so that path is authoritative: a
    later retry attempt can leave a *newer* file behind that did not produce the
    recorded ``val_mse``, and the untouched-arm invariant is checked against that
    very number.  The registered inventory's ``attempts/*/checkpoints`` glob is
    the fallback.
    """
    metrics_path = run_dir / "metrics.csv"
    recorded = ""
    if metrics_path.is_file():
        with metrics_path.open(newline="") as handle:
            first = next(csv.DictReader(handle), None) or {}
        recorded = str(first.get("checkpoint", "")).strip()
    if recorded:
        raw = Path(recorded)
        candidates = [REPO_ROOT / raw, run_dir / raw]
        parts = raw.parts
        for index, part in enumerate(parts):
            if part == "runs" and index + 1 < len(parts):
                # the runs are relocatable, so the recorded prefix may not exist
                candidates.append(run_dir / Path(*parts[index + 1:]))
                break
        candidates.append(run_dir / "checkpoints" / raw.name)
        candidates.extend(
            sorted((run_dir / "attempts").glob(f"*/checkpoints/{raw.name}"))
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate, "metrics_csv"
    candidates = sorted(
        run_dir.glob("attempts/*/checkpoints/best.ckpt"),
        key=lambda path: path.parent.parent.name,
    )
    if candidates:
        return candidates[-1], "attempts_glob"
    return None, ""


def read_run_metrics(run_dir: Path) -> dict:
    """``val_mse`` / ``val_mae`` of the run, plus its test-read marker.

    ``test_read`` is recorded only so the audit can show that E14 had already
    taken its single test read (legacy E8 runs keep it in an external summary);
    E16 itself never reads it.
    """
    path = run_dir / "metrics.csv"
    row = {"val_mse": None, "val_mae": None, "test_mse_recorded": False}
    if not path.is_file():
        return row
    with path.open(newline="") as handle:
        first = next(csv.DictReader(handle), None)
    if not first:
        return row
    for key in ("val_mse", "val_mae"):
        raw = str(first.get(key, "")).strip()
        row[key] = float(raw) if raw else None
    row["test_mse_recorded"] = bool(str(first.get("test_mse", "")).strip())
    return row


def config_of_run_dir(run_dir: Path) -> dict:
    path = run_dir / "config.json"
    if not path.is_file():
        raise SystemExit(f"run directory without config.json: {run_dir}")
    return json.loads(path.read_text())


def find_e14_run_dir(
    e14_root: Path, dataset: str, horizon: int, seed: int, arm: str
) -> tuple[Path | None, list[str]]:
    """Resolve the E14 run directory of one cell.

    ``e14_main_matrix._arm_match`` is imported rather than reimplemented so that
    "which run implements l_q1_4" is defined in exactly one place.
    """
    matches: list[Path] = []
    runs_dir = e14_root / "runs"
    if runs_dir.is_dir():
        for config_path in sorted(runs_dir.glob("*/config.json")):
            try:
                config = json.loads(config_path.read_text())
            except json.JSONDecodeError:
                continue
            if (
                str(config.get("dataset")) == dataset
                and int(config.get("horizon", -1)) == int(horizon)
                and int(config.get("seed", -1)) == int(seed)
                and _arm_match(config, arm)
            ):
                matches.append(config_path.parent)

    def sort_key(run_dir: Path):
        metrics = read_run_metrics(run_dir)
        val_mse = metrics["val_mse"]
        return (
            0 if (run_dir / "metrics.csv").is_file() else 1,
            val_mse if val_mse is not None else float("inf"),
            str(run_dir),
        )

    matches.sort(key=sort_key)
    alternatives = [str(path.relative_to(REPO_ROOT)) for path in matches]
    return (matches[0] if matches else None), alternatives


def resolve_explicit_checkpoints(raw: str, arms: list[str]) -> list[dict]:
    """Cells named by an explicit ``--checkpoints`` list.

    Every checkpoint must sit inside a run directory that carries
    ``config.json`` (the requirement the registered inventory also imposes),
    because dataset/horizon/seed/arm are read from the run spec instead of being
    guessed from the path.
    """
    cells: list[dict] = []
    for item in parse_list(raw):
        checkpoint = Path(item)
        if not checkpoint.is_absolute():
            checkpoint = REPO_ROOT / checkpoint
        if not checkpoint.is_file():
            raise SystemExit(f"--checkpoints entry is not a file: {checkpoint}")
        run_dir = None
        for parent in [checkpoint.parent, *checkpoint.parents]:
            if (parent / "config.json").is_file():
                run_dir = parent
                break
        if run_dir is None:
            raise SystemExit(
                f"--checkpoints entry {checkpoint} has no run directory with "
                "config.json; dataset/horizon/seed/arm cannot be resolved"
            )
        config = config_of_run_dir(run_dir)
        arm = next((name for name in arms if _arm_match(config, name)), None)
        if arm is None:
            raise SystemExit(
                f"{run_dir} implements none of {arms} "
                f"(mechanism={config.get('mechanism')!r}, head="
                f"{config.get('hyperparams', {}).get('weak_period_residual_head_type')!r})"
            )
        cells.append(
            {
                "arm": arm,
                "dataset": str(config["dataset"]),
                "horizon": int(config["horizon"]),
                "seed": int(config["seed"]),
                "e14_status": "explicit",
                "run_dir": run_dir,
                "checkpoint": checkpoint,
                "config_hash": config.get("config_hash"),
                "checkpoint_source": "explicit",
                "n_alternative_run_dirs": 0,
                "alternative_run_dirs": [],
            }
        )
    return cells


def finish_cell_record(cell: dict) -> dict:
    config = config_of_run_dir(cell["run_dir"])
    hyper = dict(config.get("hyperparams", {}))
    cell["hyperparams"] = hyper
    cell["batch_size"] = int(
        config.get("batch_size") or hyper.get("batch_size") or 256
    )
    cell["config_hash"] = cell.get("config_hash") or config.get("config_hash")
    cell["mechanism"] = str(config.get("mechanism"))
    cell["head_type"] = str(hyper.get("weak_period_residual_head_type", "shared"))
    cell["loss"] = str(config.get("loss"))
    cell["max_epochs"] = config.get("max_epochs")
    cell["percent"] = config.get("percent")
    cell["lookback"] = config.get("lookback")
    cell["period"] = config.get("period")
    cell["learning_rate"] = hyper.get("learning_rate")
    cell["gate_init"] = hyper.get("weak_period_residual_gate_init")
    cell["smooth_ratio"] = hyper.get("weak_period_residual_smooth_ratio", 0.0)
    metrics = read_run_metrics(cell["run_dir"])
    cell["selected_val_mse"] = metrics["val_mse"]
    cell["selected_val_mae"] = metrics["val_mae"]
    cell["run_mse_csv_has_test_metric"] = metrics["test_mse_recorded"]
    cell["setting"] = f"{cell['dataset']}-{cell['horizon']}"
    cell["q_label"] = SUPPORTED_ARMS[cell["arm"]]["q_label"]
    rank = hyper.get("weak_period_residual_rank")
    cell["lowrank_rank"] = int(rank) if rank not in (None, "") else 0
    cell["run_dir_rel"] = str(cell["run_dir"].relative_to(REPO_ROOT))
    cell["checkpoint_rel"] = str(cell["checkpoint"].relative_to(REPO_ROOT))
    return cell


def build_cell_plan(args) -> list[dict]:
    """Every (arm, dataset, horizon, seed) of the run, with its checkpoint."""
    wanted_arms = parse_list(args.arms)
    for arm in wanted_arms:
        if arm not in SUPPORTED_ARMS:
            raise SystemExit(
                f"--arms {arm!r} is not dissectable; E16 supports "
                f"{sorted(SUPPORTED_ARMS)}.  phase_only has no residual branch, "
                "and l_rcrf/a1 do not use the plain two-way gate this algebra "
                "assumes."
            )
    wanted_seeds = parse_list(args.seeds, int)
    datasets = (
        set(parse_list(args.datasets))
        if args.datasets
        else {dataset for dataset, _ in TEST_SELECTED_SETTINGS}
    )
    horizons = (
        set(parse_list(args.horizons, int))
        if args.horizons
        else {horizon for _, horizon in TEST_SELECTED_SETTINGS}
    )
    if not wanted_seeds:
        raise SystemExit("--seeds is empty")

    cells: list[dict] = []
    if args.checkpoints:
        cells = [
            cell
            for cell in resolve_explicit_checkpoints(args.checkpoints, wanted_arms)
            if cell["arm"] in wanted_arms
            and cell["dataset"] in datasets
            and cell["horizon"] in horizons
            and cell["seed"] in wanted_seeds
        ]
    else:
        e14_root = REPO_ROOT / args.e14_root
        manifest = load_e14_manifest(e14_root)
        manifest_cells = {
            (
                str(cell.get("arm")),
                str(cell.get("dataset")),
                int(cell.get("horizon", -1)),
                int(cell.get("seed", -1)),
            ): cell
            for cell in manifest["cells"]
        }
        missing: list[str] = []
        for arm in wanted_arms:
            for dataset in sorted(datasets):
                for horizon in sorted(horizons):
                    for seed in wanted_seeds:
                        entry = manifest_cells.get((arm, dataset, horizon, seed))
                        run_dir = None
                        status = "not_in_manifest"
                        alternatives: list[str] = []
                        if entry is not None:
                            status = str(entry.get("status", ""))
                            source = entry.get("source") or {}
                            if status == "reused" and source.get("run_dir"):
                                run_dir = REPO_ROOT / str(source["run_dir"])
                            else:
                                run_dir, alternatives = find_e14_run_dir(
                                    e14_root, dataset, horizon, seed, arm
                                )
                        else:
                            run_dir, alternatives = find_e14_run_dir(
                                e14_root, dataset, horizon, seed, arm
                            )
                        checkpoint, checkpoint_source = (
                            checkpoint_in_run_dir(run_dir)
                            if run_dir is not None
                            else (None, "")
                        )
                        if checkpoint is None:
                            missing.append(
                                f"{arm} {dataset}-{horizon} seed={seed}"
                                + (f" (run dir {run_dir})" if run_dir else " (no run found)")
                            )
                            continue
                        source = (entry or {}).get("source") or {}
                        cells.append(
                            {
                                "arm": arm,
                                "dataset": dataset,
                                "horizon": horizon,
                                "seed": seed,
                                "e14_status": status,
                                "run_dir": run_dir,
                                "checkpoint": checkpoint,
                                "config_hash": source.get("config_hash"),
                                "checkpoint_source": checkpoint_source,
                                "n_alternative_run_dirs": len(alternatives),
                                "alternative_run_dirs": alternatives[:8],
                            }
                        )
        if missing:
            report = "cells without a resolvable checkpoint:\n  " + "\n  ".join(missing)
            if not args.allow_missing_cells:
                raise SystemExit(
                    report
                    + "\nRe-run once E14 has produced the affected cells, or pass "
                    "--allow-missing-cells to write an explicitly partial table."
                )
            print(f"[warn] {report}", flush=True)

    cells = [finish_cell_record(cell) for cell in cells]
    cells.sort(key=lambda cell: (cell["arm"], cell["dataset"], cell["horizon"], cell["seed"]))
    if not cells:
        raise SystemExit(
            "no cell resolved; check --arms/--datasets/--horizons/--seeds and --e14-root"
        )
    return cells


# ---------------------------------------------------------------------------
# Train-split statistics (one streaming pass per setting)
# ---------------------------------------------------------------------------
def train_statistics(dataset: str, horizon: int, args, registry: dict) -> dict:
    """Streaming second moments of the train split for one (dataset, horizon).

    One pass over (window block x channel block) yields everything the
    dissection takes from the train split: the centered input moments of the
    semantic dictionary metric, and the moments that reproduce
    ``scripts/compute_phase_conditional_rrr.py``'s independent-RRR fit.  The
    RevIN weighting matters because the branch's own ``z`` is
    ``(x - x_last) / sigma_window``: the registered fit is the
    dataset-standardized fit reweighted by ``1 / sigma_window^2``.
    """
    train_seg, _, meta = load_split(
        resolve_csv_path(dataset, registry, args.data_root),
        registry[dataset]["kind"],
        LOOKBACK,
    )
    channels = int(train_seg.shape[1])
    total = len(train_seg) - LOOKBACK - horizon + 1
    if total <= 0:
        raise SystemExit(
            f"{dataset}-{horizon}: the train split has {len(train_seg)} rows but a "
            f"window needs {LOOKBACK + horizon}"
        )
    budget_bytes = max(1.0, float(args.mem_budget_mb)) * 1024 * 1024
    window_chunk = 4096
    # one pair costs z (L) + d (H) + the window statistics (2 L) in float64
    per_pair = (3 * LOOKBACK + horizon) * 8
    if args.channel_block > 0:
        channel_block = max(1, min(args.channel_block, channels))
    else:
        channel_block = max(
            1,
            min(
                channels,
                int(budget_bytes * 0.5 // max(1, window_chunk * per_pair)),
            ),
        )

    x_windows = np.lib.stride_tricks.sliding_window_view(train_seg, LOOKBACK, axis=0)[:total]
    y_windows = np.lib.stride_tricks.sliding_window_view(
        train_seg[LOOKBACK:], horizon, axis=0
    )[:total]

    count = 0
    weight_sum = 0.0
    sum_z = np.zeros(LOOKBACK)
    gram_zz = np.zeros((LOOKBACK, LOOKBACK))
    gram_zd = np.zeros((LOOKBACK, horizon))
    weighted_gram_zz = np.zeros((LOOKBACK, LOOKBACK))
    weighted_gram_zd = np.zeros((LOOKBACK, horizon))
    for w0 in range(0, total, window_chunk):
        w1 = min(w0 + window_chunk, total)
        for c0, c1 in iter_channel_blocks(channels, channel_block):
            x = np.ascontiguousarray(x_windows[w0:w1, c0:c1], dtype=np.float64)
            y = np.ascontiguousarray(y_windows[w0:w1, c0:c1], dtype=np.float64)
            last = x[:, :, -1:]
            z = x - last
            d = y - last
            mu = x.mean(axis=2, keepdims=True)
            var = ((x - mu) ** 2).mean(axis=2, keepdims=True)
            sigma = np.sqrt(var + REVIN_EPS)  # PhaseFormer RevIN.normalize
            weights = 1.0 / (sigma[:, :, 0] ** 2)
            z_flat = z.reshape(-1, LOOKBACK)
            d_flat = d.reshape(-1, horizon)
            w_flat = weights.reshape(-1)
            count += int(z_flat.shape[0])
            weight_sum += float(w_flat.sum())
            sum_z += z_flat.sum(axis=0)
            gram_zz += z_flat.T @ z_flat
            gram_zd += z_flat.T @ d_flat
            weighted = z_flat * w_flat[:, None]
            weighted_gram_zz += weighted.T @ z_flat
            weighted_gram_zd += weighted.T @ d_flat
            del x, y, z, d, mu, var, sigma, weights, z_flat, d_flat, weighted

    mean_z = sum_z / count
    # ddof=1, exactly as analyze_lowrank_checkpoint_information.centered_moments
    covariance = (gram_zz - count * np.outer(mean_z, mean_z)) / max(count - 1, 1)
    print(
        f"[train] {dataset}-{horizon}: pairs={count} channels={channels} "
        f"channel_block={channel_block} windows={total} "
        f"rows parsed={meta['rows_read']} (test border at row "
        f"{meta['test_border_row']}; test rows never read)",
        flush=True,
    )
    return {
        "dataset": dataset,
        "horizon": horizon,
        "moments": CenteredMoments(mean=mean_z, cov=covariance, n=int(count)),
        "szz": gram_zz / count,
        "szy": gram_zd / count,
        "weighted_szz": weighted_gram_zz / count,
        "weighted_szy": weighted_gram_zd / count,
        "train_pairs": int(count),
        "train_windows": int(total),
        "channels": channels,
        "channel_block": int(channel_block),
        "weight_sum": float(weight_sum),
        "csv_rows_parsed": int(meta["rows_read"]),
        "test_border_row": int(meta["test_border_row"]),
        "test_split_read": False,
    }


def rrr_pool_from_train(stats: dict, rank: int, space: str) -> np.ndarray:
    """Independent-RRR input basis of the train fit, ``(720, <= rank)``."""
    if space == "revin":
        szz, szy = stats["weighted_szz"], stats["weighted_szy"]
    else:
        szz, szy = stats["szz"], stats["szy"]
    basis, _ = independent_rrr(szz, szy, int(rank), ridge=1e-6)
    return np.ascontiguousarray(basis)


# ---------------------------------------------------------------------------
# Head instrumentation
# ---------------------------------------------------------------------------
def dense_head_forward():
    """Replacement ``forward`` for the dense ``shared`` NLinear residual head.

    Mirrors ``scripts/lowrank_checkpoint_model.intervention_forward`` for
    ``WeakPeriodResidualHead``, whose private input is the centered normalized
    history ``(B, C, L)`` and whose decoder is ``linear``: with that head the
    hidden state *is* the branch's private input, so the same arm algebra
    applies and the recorded anchor is the head's own uncentered last step.
    """

    def forward(self, x):  # noqa: D401 - mirrors the head signature
        last = x[:, -1:, :]
        centered = (x - last).permute(0, 2, 1).contiguous()  # (B, C, L)
        delta = self.linear(centered).permute(0, 2, 1).contiguous()  # (B, H, C)
        self.last_centered = centered
        self.last_hidden = centered
        self.last_forward_output = delta + last.expand(
            -1, self.linear.out_features, -1
        )
        self.last_anchor64 = last.double()
        return self.last_forward_output

    return forward


@contextlib.contextmanager
def instrument_branch(model, head_kind: str):
    """Patch the residual head and the top-level forward for one context.

    ``lowrank_checkpoint_model.instrument_model`` cannot be reused as is: it
    hard-requires ``PooledLowRankWeakPeriodResidualHead`` and E16 must also
    cover the dense ``shared`` head.  Its ``_capture_forward`` *is* reused, so
    E16 observes exactly the quantities the registered evaluator caches (RevIN
    statistics, gate, phase-only forecast, model output, anchor, hidden state).
    """
    head = model.weak_period_residual
    original_head_forward = head.forward
    original_model_forward = type(model).forward
    if head_kind == "pooled_lowrank":
        head.forward = types.MethodType(intervention_forward(None, None), head)
    else:
        head.forward = types.MethodType(dense_head_forward(), head)
    type(model).forward = _capture_forward(model, original_model_forward)
    try:
        yield model
    finally:
        head.forward = original_head_forward
        type(model).forward = original_model_forward
        for attribute in (
            "last_centered", "last_pooled", "last_hidden", "last_hidden_used",
            "last_audit", "last_forward_output", "last_anchor64",
        ):
            if hasattr(head, attribute):
                delattr(head, attribute)


def describe_branch(model, state: dict) -> dict:
    """Guards without which the closed-form arm algebra would be wrong."""
    head = model.weak_period_residual
    if getattr(model.args, "use_adaptive_weak_period_gate", False) or getattr(
        model.args, "use_adaptive_residual_gate", False
    ):
        raise SystemExit(
            "the dissected cells must use the fixed sigmoid gate; an adaptive "
            "gate makes the fused metric input dependent"
        )
    for flag in ("use_rcrf_fusion", "use_dual_reliability_fusion"):
        if getattr(model, flag, False):
            raise SystemExit(
                f"{flag} is set: that route fuses through a reliability/alpha map "
                "instead of the plain two-way gate this algebra assumes"
            )
    if getattr(model, "weak_residual_asymmetric_component", "none") not in ("none", ""):
        raise SystemExit(
            "weak_residual_asymmetric_component is set: the residual branch then "
            "reads a component of the input, not the RevIN-normalized history"
        )
    if not getattr(model, "use_revin", False):
        raise SystemExit("the dissected cells must use RevIN")
    if getattr(model.revin, "affine", False):
        raise SystemExit("an affine RevIN would add a per-channel scale the arms ignore")
    if float(head.smooth_ratio) != 0.0:
        raise SystemExit(
            f"smooth_ratio={float(head.smooth_ratio)} != 0; smoothing is a linear "
            "operator on the branch input that the effective map does not contain, "
            "so every dissected cell must have smooth_ratio=0"
        )
    if isinstance(head, PooledLowRankWeakPeriodResidualHead):
        if int(head.pool_factor) != 1:
            raise SystemExit(
                f"pool_factor={int(head.pool_factor)} != 1: the effective map would "
                "have to fold the pooling operator in, and no cell of the section "
                "4.4 scope uses pooling"
            )
        return {
            "head_kind": "pooled_lowrank",
            "encoder_weight": state["weak_period_residual.encoder.weight"].detach().double().cpu().numpy(),
            "encoder_bias": state["weak_period_residual.encoder.bias"].detach().double().cpu().numpy(),
            "decoder_weight": state["weak_period_residual.decoder.weight"].detach().double().cpu().numpy(),
            "decoder_bias": state["weak_period_residual.decoder.bias"].detach().double().cpu().numpy(),
            "pooled_len": int(head.pooled_len),
            "pool_factor": int(head.pool_factor),
            "rank_dim": int(head.rank),
            "hidden_kind": "latent_bottleneck",
        }
    if isinstance(head, WeakPeriodResidualHead):
        if head.projection_basis is not None:
            raise SystemExit(
                "the dense head carries a frozen projection basis: that is the E8 "
                "frozen-subspace arm, not the PhaseFormer-L main row"
            )
        weight = state["weak_period_residual.linear.weight"].detach().double().cpu().numpy()
        bias = state["weak_period_residual.linear.bias"].detach().double().cpu().numpy()
        ambient = int(weight.shape[1])
        return {
            "head_kind": "dense_shared",
            "encoder_weight": np.eye(ambient),
            "encoder_bias": np.zeros(ambient),
            "decoder_weight": weight,
            "decoder_bias": bias,
            "pooled_len": ambient,
            "pool_factor": 1,
            "rank_dim": ambient,
            "hidden_kind": "centered_input",
        }
    raise SystemExit(f"unsupported residual head for E16: {type(head).__name__}")


# ---------------------------------------------------------------------------
# Streaming accumulators
# ---------------------------------------------------------------------------
class CellAccumulator:
    """Scalar / small-matrix accumulators of one cell.

    No ``(samples, horizon, channels)`` tensor and no hidden block survives a
    batch: every quantity below is a sum over pairs, so the reported means,
    energies and bands do not depend on the blocking (up to float64 association).
    """

    def __init__(self, rank_dim: int, horizon: int, modes: int):
        self.rank_dim = int(rank_dim)
        self.horizon = int(horizon)
        self.modes = int(modes)
        self.elements = 0  # pairs x horizon: the denominator of every mean
        self.pairs = 0
        self.hidden_sum = np.zeros(self.rank_dim)
        self.hidden_gram = np.zeros((self.rank_dim, self.rank_dim))
        self.score_sigma_sq = np.zeros(self.modes)
        self.sigma_sum = 0.0
        self.gate_sum = 0.0
        self.correction_sq = 0.0
        self.reference_sum = 0.0
        self.reference_sq = 0.0
        self.recorded_fused_sq = 0.0
        self.recorded_fused_abs = 0.0
        self.residual_sum = np.zeros(self.horizon)
        self.residual_gram = np.zeros((self.horizon, self.horizon))
        self.arm: dict[str, dict] = {}
        self.band: dict[str, dict] = {}

    def declare_arms(self, arm_names) -> None:
        for name in arm_names:
            self.arm[name] = {
                "branch_sq": 0.0, "branch_abs": 0.0, "fused_sq": 0.0,
                "fused_abs": 0.0, "recon_sq": 0.0, "correction_sq": 0.0,
            }

    def declare_bands(self, band_names) -> None:
        for name in band_names:
            self.band[name] = {"mse": None, "mae": None}

    def add_statistics_block(
        self,
        hidden: np.ndarray,
        sigma: np.ndarray,
        gate: np.ndarray,
        phase: np.ndarray,
        target: np.ndarray,
        fused: np.ndarray,
        decoder_weight: np.ndarray,
        decoder_bias: np.ndarray,
        encoder_bias: np.ndarray,
        decoder_frame: np.ndarray,
    ) -> None:
        """Pass 1: the quantities a first-order basis needs."""
        samples, channels, _ = hidden.shape
        horizon = self.horizon
        bias_term = affine_bias(encoder_bias, decoder_weight, decoder_bias)
        mapped = np.einsum("ncr,hr->nhc", hidden, decoder_weight)
        correction = (mapped + bias_term[None, :, None]) * sigma
        pairs = int(samples * channels)
        self.pairs += pairs
        self.elements += pairs * horizon
        flat_hidden = hidden.reshape(-1, self.rank_dim)
        self.hidden_sum += flat_hidden.sum(axis=0)
        self.hidden_gram += flat_hidden.T @ flat_hidden
        scale = sigma.reshape(pairs)
        for index in range(min(self.modes, decoder_frame.shape[1])):
            score = flat_hidden @ decoder_frame[index]
            self.score_sigma_sq[index] += float(np.sum((score ** 2) * (scale ** 2)))
        self.sigma_sum += float(sigma.sum())
        self.gate_sum += float(gate.sum())
        self.correction_sq += float(np.sum(correction ** 2))
        self.reference_sum += float(correction.sum())
        self.reference_sq += float(np.sum(correction ** 2))
        recorded = fused - target
        self.recorded_fused_sq += float(np.sum(recorded ** 2))
        self.recorded_fused_abs += float(np.sum(np.abs(recorded)))
        residual = (target - phase).reshape(-1, horizon)
        self.residual_sum += residual.sum(axis=0)
        self.residual_gram += residual.T @ residual
        del mapped, correction, flat_hidden, scale, score, recorded, residual

    def add_arm_block(
        self,
        hidden: np.ndarray,
        sigma: np.ndarray,
        gate: np.ndarray,
        phase: np.ndarray,
        target: np.ndarray,
        last_abs: np.ndarray,
        decoder_weight: np.ndarray,
        decoder_bias: np.ndarray,
        encoder_bias: np.ndarray,
        arms,
    ) -> None:
        """Pass 2: every intervention arm, through the registered algebra."""
        correction_reference = (
            np.einsum("ncr,hr->nhc", hidden, decoder_weight)
            + affine_bias(encoder_bias, decoder_weight, decoder_bias)[None, :, None]
        ) * sigma
        for arm_name, basis, mode in arms:
            metrics = arm_metrics(
                hidden, decoder_weight, decoder_bias, encoder_bias, gate,
                phase, target, correction_reference, last_abs, sigma, basis, mode,
            )
            row = self.arm[arm_name]
            count = metrics["pair_count"]
            row["branch_sq"] += metrics["branch_mse"] * count
            row["branch_abs"] += metrics["branch_mae"] * count
            row["fused_sq"] += metrics["fused_mse"] * count
            row["fused_abs"] += metrics["fused_mae"] * count
            row["recon_sq"] += (metrics["correction_rmse"] ** 2) * count
            row["correction_sq"] += metrics["correction_energy"] * count
        del correction_reference

    def add_band_block(
        self,
        hidden: np.ndarray,
        sigma: np.ndarray,
        gate: np.ndarray,
        phase: np.ndarray,
        target: np.ndarray,
        last_abs: np.ndarray,
        decoder_weight: np.ndarray,
        decoder_bias: np.ndarray,
        encoder_bias: np.ndarray,
        bands: dict,
        block_elements: int,
        chunk: int = 32,
    ) -> None:
        """Pass 2: the null bands, with the cell's pre-drawn bases.

        The bases are drawn once per cell (see :func:`build_bases`); replaying
        the *same* arms on every block is what makes the weighted average of the
        block means each arm's own metric.  Drawing per block would mix arms
        across blocks and understate the band's spread.
        """
        pairs = int(hidden.shape[0] * hidden.shape[1])
        weight = pairs / max(self.pairs, 1)
        for name, payload in bands.items():
            bases = payload["bases"]
            mse, mae = random_drop_band(
                hidden, decoder_weight, decoder_bias, encoder_bias, sigma,
                last_abs, gate, phase, target, int(hidden.shape[-1]),
                int(bases.shape[0]), _PLACEHOLDER_RNG, chunk=chunk,
                block_elements=block_elements, bases=bases,
            )
            if self.band[name]["mse"] is None:
                self.band[name]["mse"] = weight * mse
                self.band[name]["mae"] = weight * mae
            else:
                self.band[name]["mse"] += weight * mse
                self.band[name]["mae"] += weight * mae

    # -- finalisation ------------------------------------------------------
    def statistics(self) -> dict:
        elements = max(self.elements, 1)
        pairs = max(self.pairs, 1)
        mean_hidden = self.hidden_sum / pairs
        covariance = (
            self.hidden_gram - pairs * np.outer(mean_hidden, mean_hidden)
        ) / max(pairs - 1, 1)
        residual_mean = self.residual_sum / pairs
        residual_covariance = (
            self.residual_gram - pairs * np.outer(residual_mean, residual_mean)
        ) / max(pairs - 1, 1)
        scale = float(np.trace(residual_covariance)) / max(self.horizon, 1)
        residual_covariance = residual_covariance + np.eye(self.horizon) * max(
            scale, 1e-12
        ) * 1e-6
        return {
            "pairs": int(self.pairs),
            "elements": int(elements),
            "sigma_mean": self.sigma_sum / pairs,
            "gate_mean": self.gate_sum / pairs,
            "total_correction_energy": self.correction_sq / elements,
            "reference_sum": self.reference_sum,
            "reference_sq": self.reference_sq,
            "recorded_fused_mse": self.recorded_fused_sq / elements,
            "recorded_fused_mae": self.recorded_fused_abs / elements,
            "hidden_mean": mean_hidden,
            "hidden_covariance": covariance,
            "output_moments": CenteredMoments(
                mean=residual_mean, cov=residual_covariance, n=int(self.pairs)
            ),
        }

    def mode_energy(self, index: int, singular_value: float) -> float:
        """``mean(contribution**2)`` of canonical mode ``index``.

        ``contribution[n, h, c] = (u_k[h] * s_k) * score_k[n, c] * sigma[n, c]``
        with ``|u_k| = 1``, so summing its square over the horizon gives
        ``s_k**2 * sum(score_k**2 sigma**2)``: the analyzer's
        ``np.mean(contribution**2)`` without ever materialising the
        ``(samples, horizon, channels)`` tensor.
        """
        return float(singular_value ** 2 * self.score_sigma_sq[index]) / max(
            self.elements, 1
        )

    def arm_metrics(self, arm_name: str) -> dict:
        row = self.arm[arm_name]
        elements = max(self.elements, 1)
        mean_reference = self.reference_sum / elements
        centered_reference = self.reference_sq - elements * mean_reference ** 2
        return {
            "branch_mse": row["branch_sq"] / elements,
            "branch_mae": row["branch_abs"] / elements,
            "fused_mse": row["fused_sq"] / elements,
            "fused_mae": row["fused_abs"] / elements,
            "correction_rmse": float(np.sqrt(row["recon_sq"] / elements)),
            "correction_energy": row["correction_sq"] / elements,
            "correction_reconstruction_r2": (
                float(1.0 - row["recon_sq"] / centered_reference)
                if centered_reference > 0
                else 0.0
            ),
        }

    def band_values(self, name: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        payload = self.band.get(name)
        if payload is None:
            return None, None
        return payload["mse"], payload["mae"]


# ---------------------------------------------------------------------------
# Bases of one cell
# ---------------------------------------------------------------------------
def pca_basis_from_covariance(covariance: np.ndarray, rank: int) -> np.ndarray:
    """Top ``rank`` principal directions of a covariance matrix.

    The same eigen-decomposition as the registered
    ``latent_input_pca_basis`` (symmetrized ``eigh``, descending, leading
    ``rank``), applied to the streamed covariance because the full hidden tensor
    is never materialised.
    """
    values, vectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    order = np.argsort(values)[::-1]
    keep = max(1, min(int(rank), vectors.shape[1]))
    return np.ascontiguousarray(vectors[:, order[:keep]])


def build_bases(
    cell: dict,
    spec: dict,
    hidden_covariance: np.ndarray,
    train_stats: dict,
    subspaces_path: Path,
    args,
) -> dict:
    """Every subspace this cell's arms and bands need, plus its provenance."""
    rank_dim = int(spec["rank_dim"])
    encoder = spec["encoder_weight"]
    z_semantic = semantic_basis(cell["dataset"], rank_dim)
    semantic_full = latent_image(z_semantic, encoder, rank_dim)
    semantic_small = np.ascontiguousarray(
        semantic_full[:, : min(int(args.semantic_rank), semantic_full.shape[1])]
    )
    pca = pca_basis_from_covariance(hidden_covariance, rank_dim)
    semantic_dimension = int(semantic_full.shape[1])
    pca_matched = (
        pca_basis_from_covariance(hidden_covariance, semantic_dimension)
        if semantic_dimension < rank_dim
        else pca
    )

    conditional = None
    pool = None
    pool_source = ""
    if subspaces_path.is_file():
        payload = np.load(subspaces_path)
        if "conditional_basis" in payload.files:
            conditional = latent_image(
                payload["conditional_basis"].astype(np.float64), encoder, rank_dim
            )
        if "independent_basis" in payload.files:
            pool = latent_image(
                payload["independent_basis"].astype(np.float64), encoder, rank_dim
            )
            pool_source = "stage3_independent_rrr_subspace_file"
    if pool is None or pool.shape[1] < 1:
        # No registered Stage 3 subspace file for this cell (the dense head has
        # none): fit the same independent RRR on the train split instead.  The
        # estimator is the one compute_phase_conditional_rrr.py uses -- the fit
        # is just done in float64 from the standardized windows instead of from
        # float32 model records.
        pool_rank = int(cell["lowrank_rank"]) or int(min(cell["horizon"], LOOKBACK))
        pool = latent_image(
            rrr_pool_from_train(train_stats, pool_rank, args.rrr_pool_space),
            encoder,
            rank_dim,
        )
        pool_source = f"e16_train_split_independent_rrr_{args.rrr_pool_space}"

    pool_dimension = int(pool.shape[1]) if pool is not None else 0
    rrr_dimension = max(1, min(semantic_dimension, pool_dimension)) if pool_dimension else 0
    repeats = max(1, int(args.random_repeats))
    rng = np.random.default_rng(RANDOM_SEED)
    legacy_dimension = legacy_random_band_dimension(rank_dim)
    bands: dict[str, dict] = {
        "random": {
            "bases": np.stack(
                [random_orthogonal_basis(rank_dim, legacy_dimension, rng)
                 for _ in range(repeats)]
            ),
            "dimension": legacy_dimension,
            "repeats": repeats,
            "family": "whole_latent_space",
            "seed": RANDOM_SEED,
        },
        "random_ambient_matched": {
            "bases": np.stack(
                [random_orthogonal_basis(rank_dim, semantic_dimension, rng)
                 for _ in range(repeats)]
            ),
            "dimension": semantic_dimension,
            "repeats": repeats,
            "family": "whole_latent_space",
            "seed": BAND_SEED_AMBIENT_MATCHED,
        },
    }
    reference = None
    if bool(args.random_rrr) and int(rrr_dimension) >= 1:
        rrr_repeats = max(1, int(args.random_rrr_repeats))
        rng_rrr = np.random.default_rng(RANDOM_SEED + 1)
        bands["random_rrr"] = {
            "bases": np.stack(
                [random_rrr_basis(pool, rrr_dimension, rng_rrr)
                 for _ in range(rrr_repeats)]
            ),
            "dimension": int(rrr_dimension),
            "repeats": rrr_repeats,
            "family": "rrr_achievable_subspace",
            "seed": RANDOM_SEED + 1,
        }
        reference = random_rrr_basis(
            pool, rrr_dimension, np.random.default_rng(RANDOM_SEED + 2)
        )
    return {
        "semantic_full": semantic_full,
        "semantic_small": semantic_small,
        "semantic_dimension": semantic_dimension,
        "pca": pca,
        "pca_matched": pca_matched,
        "conditional": conditional,
        "pool": pool,
        "pool_dimension": pool_dimension,
        "pool_source": pool_source,
        "pool_covers_ambient": bool(pool_dimension >= rank_dim),
        "rrr_dimension": int(rrr_dimension),
        "rrr_reference": reference,
        "bands": bands,
    }


def build_arm_plan(bases: dict) -> list[tuple[str, np.ndarray | None, str]]:
    """The section 4.4 intervention arms of one cell.

    Names, order and semantics are the registered ones; ``PCA-matched-*`` and
    ``RandomRRR-drop`` are the two additions that make the same-dimension
    question answerable (in the existing data ``Semantic-drop`` equals
    ``PCA-drop`` in 57/57 dimension-matched cells, while ``PCA-drop`` keeps
    ``rank`` dimensions and ``Semantic-drop`` only as many as the dictionary
    allows).
    """
    arms: list[tuple[str, np.ndarray | None, str]] = [
        ("Original", None, "identity"),
        ("Semantic-only", bases["semantic_full"], "only"),
        ("Semantic-drop", bases["semantic_full"], "drop"),
        ("Semantic8-only", bases["semantic_small"], "only"),
        ("Semantic8-drop", bases["semantic_small"], "drop"),
        ("Bias-off", None, "bias"),
        ("PCA-only", bases["pca"], "only"),
        ("PCA-drop", bases["pca"], "drop"),
    ]
    if bases["pca_matched"].shape[1] != bases["pca"].shape[1]:
        arms.append(("PCA-matched-only", bases["pca_matched"], "only"))
        arms.append(("PCA-matched-drop", bases["pca_matched"], "drop"))
    if bases["pool"] is not None:
        arms.append(("Independent-RRR-only", bases["pool"], "only"))
    if bases["conditional"] is not None:
        arms.append(("Conditional-RRR-only", bases["conditional"], "only"))
    if bases["rrr_reference"] is not None:
        arms.append(("RandomRRR-drop", bases["rrr_reference"], "drop"))
    return arms


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------
def run_cell(
    cell: dict,
    spec: dict,
    model,
    val_loader,
    device,
    dictionary: SettingDictionary,
    train_stats: dict,
    args,
) -> dict:
    """Two streamed validation passes; returns the cell's dissection payload."""
    dataset = cell["dataset"]
    horizon = int(cell["horizon"])
    rank_dim = int(spec["rank_dim"])
    decoder_weight = spec["decoder_weight"]
    decoder_bias = spec["decoder_bias"]
    encoder_bias = spec["encoder_bias"]
    encoder_weight = spec["encoder_weight"]
    matrix, _ = effective_map(encoder_weight, decoder_weight, spec["pooled_len"])
    output_basis, singular, input_basis = np.linalg.svd(matrix, full_matrices=False)
    dec_u, dec_s, dec_vt = np.linalg.svd(decoder_weight, full_matrices=False)
    mode_count = int(min(int(args.modes), singular.size, dec_s.size))
    # The registered arm algebra adds the mapped encoder bias to a hidden
    # state that already contains the encoder bias, so the closed-form
    # untouched arm is offset from the model's own output by this term.  It is
    # recorded instead of silently corrected, because the probe intervention
    # table is already filled with the registered convention; the term is
    # exactly zero for the dense head (encoder = I, no encoder bias).
    mapped_encoder_bias_absmax = float(
        np.abs(decoder_weight @ encoder_bias).max()
    )

    set_seed(20260916)
    accumulator = CellAccumulator(rank_dim, horizon, mode_count)
    block_sizes = {"channel_block": None, "band_block_elements": None}

    def iterate(pass_name: str) -> int:
        batches = 0
        with torch.inference_mode():
            for batch_index, batch in enumerate(val_loader):
                if args.max_batches and batch_index >= args.max_batches:
                    break
                batch = [
                    item.to(device) if torch.is_tensor(item) else item
                    for item in batch
                ]
                x, y, x_mark, y_mark = batch
                dec = model._build_decoder_input(y.float())
                with instrument_branch(model, spec["head_kind"]) as instrumented:
                    fused, _, _ = instrumented(
                        x.float(), x_mark.float(), dec, y_mark.float()
                    )
                    records = instrumented.last_lowrank_records
                    phase = instrumented.last_phase_forecast
                hidden = records["hidden"]
                anchor = records["anchor64"]
                if hidden is None or anchor is None or phase is None:
                    raise SystemExit(
                        "the instrumented head did not record its hidden state / "
                        "anchor / phase forecast; refusing to report unsupported numbers"
                    )
                mu, sigma = records["stats"]
                gate = records["gate"]
                if int(gate.numel()) != int(hidden.shape[1]):
                    raise SystemExit(
                        "the gate is not a per-channel scalar; E16 supports only "
                        "the fixed sigmoid gate"
                    )
                samples, channels = int(hidden.shape[0]), int(hidden.shape[1])
                target = y.float()[:, -horizon:, :]
                # Band repeats are known before the bases exist (pass 1 sizes its
                # blocks the same way pass 2 does); the bands use exactly these
                # two counts, so the bound is the same in both passes.
                band_repeats = max(
                    int(args.random_repeats),
                    int(args.random_rrr_repeats) if args.random_rrr else 0,
                )
                channel_block, band_block = resolve_block_sizes(
                    args.mem_budget_mb, samples, channels, rank_dim, horizon,
                    band_repeats, args.channel_block,
                )
                block_sizes["channel_block"] = channel_block
                block_sizes["band_block_elements"] = band_block
                gate_vector = gate.reshape(-1)
                for c0, c1 in iter_channel_blocks(channels, channel_block):
                    hidden_block = hidden[:, c0:c1, :].double().cpu().numpy()
                    sigma_block = sigma[:, :, c0:c1].double().cpu().numpy()
                    mu_block = mu[:, :, c0:c1].double().cpu().numpy()
                    # ``gate`` is a per-channel constant tensor; it is broadcast
                    # to the cache's ``(samples, 1, channels)`` convention here
                    # because ``random_drop_band`` slices it per sample.
                    gate_block = np.broadcast_to(
                        gate_vector[c0:c1].double().cpu().numpy().reshape(1, 1, -1),
                        (samples, 1, c1 - c0),
                    )
                    phase_block = phase[:, :, c0:c1].double().cpu().numpy()
                    target_block = target[:, :, c0:c1].double().cpu().numpy()
                    anchor_block = anchor[:, :, c0:c1].double().cpu().numpy()
                    last_abs_block = anchor_block * sigma_block + mu_block
                    if pass_name == "statistics":
                        accumulator.add_statistics_block(
                            hidden_block, sigma_block, gate_block, phase_block,
                            target_block, fused[:, :, c0:c1].double().cpu().numpy(),
                            decoder_weight, decoder_bias, encoder_bias, dec_vt,
                        )
                    else:
                        accumulator.add_arm_block(
                            hidden_block, sigma_block, gate_block, phase_block,
                            target_block, last_abs_block, decoder_weight,
                            decoder_bias, encoder_bias, cell["arms"],
                        )
                        accumulator.add_band_block(
                            hidden_block, sigma_block, gate_block, phase_block,
                            target_block, last_abs_block, decoder_weight,
                            decoder_bias, encoder_bias, cell["bases"]["bands"],
                            band_block,
                        )
                    del (hidden_block, sigma_block, mu_block, gate_block,
                         phase_block, target_block, anchor_block, last_abs_block)
                del x, y, x_mark, y_mark, dec, fused, records, phase, hidden, anchor
                batches += 1
        return batches

    batches_first = iterate("statistics")
    statistics = accumulator.statistics()
    dictionary.output_moments = statistics["output_moments"]
    bases = build_bases(
        cell, spec, statistics["hidden_covariance"], train_stats,
        REPO_ROOT / args.reference_root / "subspaces"
        / f"{cell['setting']}_seed{cell['seed']}_{cell['q_label'].replace('/', '-')}.npz",
        args,
    )
    cell["bases"] = bases
    cell["arms"] = build_arm_plan(bases)
    accumulator.declare_arms([arm[0] for arm in cell["arms"]])
    accumulator.declare_bands(list(bases["bands"]))
    batches_second = iterate("arms")

    # Semantic attribution of the canonical modes: weights + train statistics
    # only, no validation quantity enters it.
    semantic_rows = []
    for index in range(mode_count):
        input_alignment = align_direction(
            input_basis[index], dictionary.input_templates,
            dictionary.input_groups, dictionary.input_group_order,
            dictionary.moments,
        )
        output_alignment = align_direction(
            output_basis[:, index], dictionary.output_templates,
            dictionary.output_groups, dictionary.output_group_order,
            dictionary.output_moments,
        )
        semantic_rows.append(
            {
                "setting": cell["setting"],
                "dataset": dataset,
                "horizon": horizon,
                "seed": cell["seed"],
                "cell": cell["q_label"],
                "arm": cell["arm"],
                "rank": cell["lowrank_rank"],
                "mode_index": index,
                "singular_value": float(singular[index]),
                "input_best_template": input_alignment["best_template"],
                "input_best_template_abs_cos": input_alignment["best_template_abs_cos"],
                "input_best_group": input_alignment["best_group"],
                "input_group_explanation": input_alignment["best_group_explanation"],
                "input_second_group": input_alignment["second_group"],
                "input_second_group_explanation": input_alignment["second_group_explanation"],
                "input_dictionary_r2": input_alignment["dictionary_r2"],
                "input_covariance_correlation_max": input_alignment["covariance_correlation_max"],
                "input_group_explanation_json": json.dumps(input_alignment["group_explanation"]),
                "input_shapley_json": json.dumps(input_alignment["shapley_r2"]),
                "output_best_template": output_alignment["best_template"],
                "output_best_template_abs_cos": output_alignment["best_template_abs_cos"],
                "output_best_group": output_alignment["best_group"],
                "output_group_explanation": output_alignment["best_group_explanation"],
                "output_second_group": output_alignment["second_group"],
                "output_second_group_explanation": output_alignment["second_group_explanation"],
                "output_dictionary_r2": output_alignment["dictionary_r2"],
                "output_covariance_correlation_max": output_alignment["covariance_correlation_max"],
                "output_group_explanation_json": json.dumps(output_alignment["group_explanation"]),
                "paired_mechanism": MECHANISM_LABELS.get(
                    (input_alignment["best_group"], output_alignment["best_group"]),
                    "未匹配预注册机制",
                ),
                "test_split_read": False,
            }
        )

    # Canonical modes: the registered analyzer's columns plus the arm tag.
    singular_energy = singular ** 2
    energy_total = float(singular_energy.sum())
    energy_shares = singular_energy / energy_total if energy_total else np.zeros_like(singular_energy)
    cumulative = np.cumsum(energy_shares)
    gaps = principal_angle_gap(singular)
    tolerance = matrix_rank_tolerance(singular)
    participation = (
        float(energy_total ** 2 / np.sum(singular_energy ** 2)) if energy_total else 0.0
    )
    total_correction_energy = statistics["total_correction_energy"]
    # Score variances live in the decoder's own latent frame; for the dense head
    # the decoder *is* the effective map, for the probe it is the bottleneck.
    # ``np.var`` in the registered analyzer is the population variance, while the
    # streamed hidden covariance is built with ``ddof=1`` (to match
    # ``centered_moments``), so the ratio converts one into the other exactly.
    pairs = max(statistics["pairs"], 1)
    score_variance = (
        np.diag(dec_vt @ statistics["hidden_covariance"] @ dec_vt.T)
        * (pairs - 1)
        / pairs
    )
    variance_total = float(score_variance.sum())
    canonical_rows = []
    for index in range(mode_count):
        # The modal *energy* lives in the decoder's own latent frame, exactly as
        # in ``analyze_lowrank_checkpoint_information``: the reported spectrum is
        # the effective map's (``singular``), but the contribution of mode
        # ``index`` is ``(dec_u[:, index] * dec_s[index]) <dec_vt[index], h>
        # sigma``.  For the dense head the two frames coincide (decoder ==
        # effective map), for the probe they do not and mixing them would
        # misstate every energy share.
        energy = accumulator.mode_energy(index, float(dec_s[index]))
        canonical_rows.append(
            {
                "setting": cell["setting"],
                "dataset": dataset,
                "horizon": horizon,
                "seed": cell["seed"],
                "cell": cell["q_label"],
                "arm": cell["arm"],
                "rank": cell["lowrank_rank"],
                "mode_index": index,
                "singular_value": float(singular[index]),
                "singular_value_share": float(energy_shares[index]),
                "cumulative_singular_share": float(cumulative[index]),
                "singular_gap_to_next": float(gaps[index]) if index < gaps.size else 0.0,
                "latent_variance": float(score_variance[index]),
                "latent_variance_share": (
                    float(score_variance[index] / variance_total) if variance_total else 0.0
                ),
                "latent_variance_std": float(np.sqrt(max(score_variance[index], 0.0))),
                "correction_energy": energy,
                "correction_energy_share": (
                    float(energy / total_correction_energy) if total_correction_energy else 0.0
                ),
                # The registered analyzer writes 0.0 here (the synthetic bias is
                # not part of any canonical mode); kept for column parity.
                "bias_energy": 0.0,
                "total_correction_energy": total_correction_energy,
                "participation_ratio": participation,
                "numerical_rank": int(np.sum(singular > tolerance)),
                "input_basis_orthonormality_error": orthonormality_error(
                    input_basis[: singular.size].T
                ),
                "output_basis_orthonormality_error": orthonormality_error(
                    output_basis[:, : singular.size]
                ),
                "phase_gate_mean": statistics["gate_mean"],
                "sigma_mean": statistics["sigma_mean"],
                "hidden_kind": spec["hidden_kind"],
                "rank_dim": rank_dim,
                "mapped_encoder_bias_absmax": mapped_encoder_bias_absmax,
                "validation_pairs": statistics["pairs"],
                "validation_batches": batches_first,
                "test_split_read": False,
            }
        )

    baseline = accumulator.arm_metrics("Original")
    recorded_val_mse = cell.get("selected_val_mse")
    if recorded_val_mse in (None, ""):
        recorded_gap = float("nan")
    else:
        recorded_gap = abs(baseline["fused_mse"] - float(recorded_val_mse))
    recorded_fused_gap = abs(statistics["recorded_fused_mse"] - baseline["fused_mse"])
    reproduces = bool(
        recorded_val_mse not in (None, "")
        and recorded_gap <= max(1e-3, 1e-3 * abs(float(recorded_val_mse)))
    )

    intervention_rows = []
    for arm_name, basis, _mode in cell["arms"]:
        metrics = accumulator.arm_metrics(arm_name)
        record = {
            "setting": cell["setting"],
            "dataset": dataset,
            "horizon": horizon,
            "seed": cell["seed"],
            "arm": cell["arm"],
            "q_or_r": cell["q_label"],
            "lowrank_rank": cell["lowrank_rank"],
            "rank_dim": rank_dim,
            "intervention_arm": arm_name,
            "subspace_dimension": int(basis.shape[1]) if basis is not None else 0,
            "branch_mse": metrics["branch_mse"],
            "branch_mae": metrics["branch_mae"],
            "fused_mse": metrics["fused_mse"],
            "fused_mae": metrics["fused_mae"],
            "delta_branch_mse_vs_checkpoint": metrics["branch_mse"] - baseline["branch_mse"],
            "delta_branch_mae_vs_checkpoint": metrics["branch_mae"] - baseline["branch_mae"],
            "delta_fused_mse_vs_checkpoint": metrics["fused_mse"] - baseline["fused_mse"],
            "delta_fused_mae_vs_checkpoint": metrics["fused_mae"] - baseline["fused_mae"],
            "correction_reconstruction_r2": metrics["correction_reconstruction_r2"],
            "correction_rmse": metrics["correction_rmse"],
            "correction_energy": metrics["correction_energy"],
            "baseline_branch_mse": baseline["branch_mse"],
            "baseline_branch_mae": baseline["branch_mae"],
            "baseline_fused_mse": baseline["fused_mse"],
            "baseline_fused_mae": baseline["fused_mae"],
            "recorded_run_val_mse": "" if recorded_val_mse is None else recorded_val_mse,
            "untouched_arm_reproduces_run_metric": reproduces,
            "untouched_arm_gap_vs_run_metric": recorded_gap,
            "untouched_arm_gap_vs_recorded_fused": recorded_fused_gap,
            "mapped_encoder_bias_absmax": mapped_encoder_bias_absmax,
            "validation_pairs": statistics["pairs"],
            "semantic_dimension": int(bases["semantic_dimension"]),
            "semantic8_dimension": int(bases["semantic_small"].shape[1]),
            "pca_dimension": int(bases["pca"].shape[1]),
            "random_band_dimension": int(bases["bands"]["random"]["dimension"]),
            "random_repeats": int(bases["bands"]["random"]["repeats"]),
            "random_rrr_available": bool("random_rrr" in bases["bands"]),
            "random_rrr_repeats": int(bases["bands"].get("random_rrr", {}).get("repeats", 0)),
            "random_rrr_dimension": int(bases["rrr_dimension"]),
            "random_rrr_pool_dimension": int(bases["pool_dimension"]),
            "random_rrr_pool_source": bases["pool_source"],
            "random_rrr_pool_covers_ambient": bool(bases["pool_covers_ambient"]),
            "semantic_pca_dimension_match": bool(
                int(bases["pca"].shape[1]) == int(bases["semantic_dimension"])
            ),
            "semantic8_pca_dimension_match": bool(
                int(bases["pca"].shape[1]) == int(bases["semantic_small"].shape[1])
            ),
            "semantic_random_band_dimension_match": bool(
                int(bases["bands"]["random"]["dimension"]) == int(bases["semantic_dimension"])
            ),
            "random_rrr_dimension_match": bool(
                "random_rrr" in bases["bands"]
                and int(bases["rrr_dimension"]) == int(bases["semantic_dimension"])
            ),
            "same_dimension_controls_available": bool(
                int(bases["pca"].shape[1]) == int(bases["semantic_dimension"])
                or (
                    "random_rrr" in bases["bands"]
                    and int(bases["rrr_dimension"]) == int(bases["semantic_dimension"])
                )
            ),
            "head_kind": spec["head_kind"],
            "hidden_kind": spec["hidden_kind"],
            "e14_status": cell["e14_status"],
            "checkpoint_path": cell["checkpoint_rel"],
            "run_dir": cell["run_dir_rel"],
            "test_split_read": False,
        }
        for band_name, band in bases["bands"].items():
            band_mse, band_mae = accumulator.band_values(band_name)
            record.update(
                band_summary(arm_name, band_name, band_mse, band_mae, metrics)
            )
        intervention_rows.append(record)

    return {
        "cell": cell,
        "spec": spec,
        "canonical_rows": canonical_rows,
        "semantic_rows": semantic_rows,
        "intervention_rows": intervention_rows,
        "statistics": statistics,
        "bases": bases,
        "svd": {
            "singular": singular,
            "input_basis": input_basis,
            "output_basis": output_basis,
            "batches_first_pass": batches_first,
            "batches_second_pass": batches_second,
            "channel_block": block_sizes["channel_block"],
            "band_block_elements": block_sizes["band_block_elements"],
        },
        "invariant": {
            "untouched_arm_reproduces_run_metric": reproduces,
            "untouched_arm_gap_vs_run_metric": recorded_gap,
            "untouched_arm_gap_vs_recorded_fused": recorded_fused_gap,
            "mapped_encoder_bias_absmax": mapped_encoder_bias_absmax,
            "validation_pairs": statistics["pairs"],
        },
    }


# ---------------------------------------------------------------------------
# Cross-seed stability
# ---------------------------------------------------------------------------
def cross_seed_rows(payloads: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for payload in payloads:
        cell = payload["cell"]
        groups.setdefault((cell["setting"], cell["arm"]), []).append(payload)
    rows = []
    for (setting, arm), group in sorted(groups.items()):
        group = sorted(group, key=lambda item: item["cell"]["seed"])
        for dimension, label in ((4, "leading4"), (8, "leading8")):
            for first in range(len(group)):
                for second in range(first + 1, len(group)):
                    a, b = group[first], group[second]
                    dim = min(
                        dimension,
                        a["svd"]["singular"].size,
                        b["svd"]["singular"].size,
                    )
                    if dim < 1:
                        continue
                    input_a = np.ascontiguousarray(a["svd"]["input_basis"][:dim].T)
                    input_b = np.ascontiguousarray(b["svd"]["input_basis"][:dim].T)
                    output_a = np.ascontiguousarray(a["svd"]["output_basis"][:, :dim])
                    output_b = np.ascontiguousarray(b["svd"]["output_basis"][:, :dim])
                    rows.append(
                        {
                            "setting": setting,
                            "dataset": a["cell"]["dataset"],
                            "horizon": a["cell"]["horizon"],
                            "arm": arm,
                            "cell": a["cell"]["q_label"],
                            "q_or_r": a["cell"]["q_label"],
                            "seed_a": a["cell"]["seed"],
                            "seed_b": b["cell"]["seed"],
                            "scope": label,
                            "dimension": int(dim),
                            "input_subspace_overlap": float(
                                projection_overlap(input_a, input_b)
                            ),
                            "output_subspace_overlap": float(
                                projection_overlap(output_a, output_b)
                            ),
                            "rank_a": a["cell"]["lowrank_rank"],
                            "rank_b": b["cell"]["lowrank_rank"],
                            "test_split_read": False,
                        }
                    )
    return rows


def leading4_overlap(payloads: list[dict], setting: str, arm: str, seed: int) -> dict:
    """Mean ``leading4`` overlap of one seed with the other seeds of its cell."""
    payload = next(
        item
        for item in payloads
        if item["cell"]["setting"] == setting
        and item["cell"]["arm"] == arm
        and item["cell"]["seed"] == seed
    )
    inputs, outputs = [], []
    for other in payloads:
        cell = other["cell"]
        if cell["setting"] != setting or cell["arm"] != arm or cell["seed"] == seed:
            continue
        dim = min(4, payload["svd"]["singular"].size, other["svd"]["singular"].size)
        if dim < 1:
            continue
        inputs.append(
            projection_overlap(
                np.ascontiguousarray(payload["svd"]["input_basis"][:dim].T),
                np.ascontiguousarray(other["svd"]["input_basis"][:dim].T),
            )
        )
        outputs.append(
            projection_overlap(
                np.ascontiguousarray(payload["svd"]["output_basis"][:, :dim]),
                np.ascontiguousarray(other["svd"]["output_basis"][:, :dim]),
            )
        )
    return {
        "input": float(np.mean(inputs)) if inputs else float("nan"),
        "output": float(np.mean(outputs)) if outputs else float("nan"),
        "pairs": len(inputs),
    }


# ---------------------------------------------------------------------------
# Dissection table
# ---------------------------------------------------------------------------
def majority_flag(flags: list[bool]) -> bool:
    if not flags:
        return False
    return sum(1 for flag in flags if flag) >= min(
        CRITERION_SEED_MAJORITY, len(flags)
    )


def dissection_rows(
    payloads: list[dict], intervention_by_key: dict
) -> list[dict]:
    """The section 4.4 dissection table, one row per (setting, arm, seed)."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for payload in payloads:
        cell = payload["cell"]
        groups.setdefault((cell["setting"], cell["arm"]), []).append(payload)

    rows: list[dict] = []
    for (setting, arm), group in sorted(groups.items()):
        group = sorted(group, key=lambda item: item["cell"]["seed"])
        leading = [payload["semantic_rows"][0] for payload in group]
        counts: dict[str, int] = {}
        for row in leading:
            counts[row["input_best_group"]] = counts.get(row["input_best_group"], 0) + 1
        majority, votes = max(counts.items(), key=lambda item: (item[1], item[0]))
        paired_outputs = sorted(
            {output for (input_group, output) in MECHANISM_LABELS if input_group == majority}
        )
        input_explanations = [
            float(json.loads(row["input_group_explanation_json"]).get(majority, np.nan))
            for row in leading
        ]
        output_explanations = []
        for row in leading:
            explanation = json.loads(row["output_group_explanation_json"])
            output_explanations.append(
                max((float(explanation.get(name, 0.0)) for name in paired_outputs), default=0.0)
                if paired_outputs
                else 0.0
            )
        drop_rows = [
            intervention_by_key.get((setting, arm, payload["cell"]["seed"], "Semantic-drop"))
            for payload in group
        ]
        only_rows = [
            intervention_by_key.get((setting, arm, payload["cell"]["seed"], "Semantic-only"))
            for payload in group
        ]
        criterion_1 = votes >= CRITERION_SEED_MAJORITY
        criterion_2 = bool(
            input_explanations
            and float(np.nanmean(input_explanations)) >= CRITERION_INPUT_EXPLANATION
        )
        criterion_3 = bool(
            output_explanations
            and float(np.mean(output_explanations)) >= CRITERION_OUTPUT_EXPLANATION
        )
        criterion_4 = majority_flag(
            [bool(row and row.get("worse_than_random_95pct_fused_mse")) for row in drop_rows]
        )
        criterion_6 = majority_flag(
            [bool(row and row.get("worse_than_random_rrr_95pct_fused_mse")) for row in drop_rows]
        )
        sufficient = []
        for row in only_rows:
            if not row:
                sufficient.append(False)
                continue
            tolerance_mse = CRITERION_SUFFICIENT_TOLERANCE * abs(row["baseline_fused_mse"])
            tolerance_mae = CRITERION_SUFFICIENT_TOLERANCE * abs(row["baseline_fused_mae"])
            sufficient.append(
                bool(
                    row["delta_fused_mse_vs_checkpoint"] <= tolerance_mse
                    and row["delta_fused_mae_vs_checkpoint"] <= tolerance_mae
                )
            )
        criterion_5 = majority_flag(sufficient)

        if not (criterion_1 and criterion_2 and criterion_3 and criterion_5):
            verdict = "not_supported"
        elif criterion_4 and criterion_6:
            verdict = "stable_specific"
        elif criterion_4 and not criterion_6:
            verdict = "stable_not_specific"
        else:
            verdict = "stable_without_necessity"

        for payload in group:
            cell = payload["cell"]
            seed = int(cell["seed"])
            leading_row = payload["semantic_rows"][0]
            canonical_row = payload["canonical_rows"][0]
            overlaps = leading4_overlap(payloads, setting, arm, seed)
            rows.append(
                {
                    "setting": setting,
                    "dataset": cell["dataset"],
                    "horizon": cell["horizon"],
                    "seed": seed,
                    "arm": arm,
                    "q_or_r": cell["q_label"],
                    "lowrank_rank": cell["lowrank_rank"],
                    "rank_dim": payload["spec"]["rank_dim"],
                    "head_kind": payload["spec"]["head_kind"],
                    "leading_mode_index": 0,
                    "leading_input_best_template": leading_row["input_best_template"],
                    "leading_input_best_template_abs_cos": leading_row["input_best_template_abs_cos"],
                    "leading_input_group": leading_row["input_best_group"],
                    "leading_input_group_label": INPUT_GROUP_LABELS.get(
                        leading_row["input_best_group"], ""
                    ),
                    "leading_input_group_explanation": leading_row["input_group_explanation"],
                    "leading_input_dictionary_r2": leading_row["input_dictionary_r2"],
                    "leading_output_best_template": leading_row["output_best_template"],
                    "leading_output_group": leading_row["output_best_group"],
                    "leading_output_group_label": OUTPUT_GROUP_LABELS.get(
                        leading_row["output_best_group"], ""
                    ),
                    "leading_output_group_explanation": leading_row["output_group_explanation"],
                    "leading_output_dictionary_r2": leading_row["output_dictionary_r2"],
                    "leading_paired_mechanism": leading_row["paired_mechanism"],
                    "leading_singular_value": canonical_row["singular_value"],
                    "leading_singular_value_share": canonical_row["singular_value_share"],
                    "leading_correction_energy": canonical_row["correction_energy"],
                    "leading_correction_energy_share": canonical_row["correction_energy_share"],
                    "total_correction_energy": canonical_row["total_correction_energy"],
                    "participation_ratio": canonical_row["participation_ratio"],
                    "numerical_rank": canonical_row["numerical_rank"],
                    "cross_seed_leading4_input_overlap": overlaps["input"],
                    "cross_seed_leading4_output_overlap": overlaps["output"],
                    "cross_seed_pairs": overlaps["pairs"],
                    # Setting-level grouping verdict, repeated on every seed row.
                    "majority_input_group": majority,
                    "majority_input_group_votes": int(votes),
                    "mean_input_group_explanation": (
                        float(np.nanmean(input_explanations)) if input_explanations else float("nan")
                    ),
                    "mean_output_group_explanation": (
                        float(np.mean(output_explanations)) if output_explanations else float("nan")
                    ),
                    "criterion_1_group_stable": bool(criterion_1),
                    "criterion_2_input_explanation_ge_0p5": bool(criterion_2),
                    "criterion_3_output_explanation_ge_0p8": bool(criterion_3),
                    "criterion_4_drop_beyond_random_95pct": bool(criterion_4),
                    "criterion_5_only_within_0p5pct": bool(criterion_5),
                    "criterion_6_drop_beyond_random_rrr_95pct": bool(criterion_6),
                    "stable_semantics_verdict": verdict,
                    "e14_status": cell["e14_status"],
                    "checkpoint_path": cell["checkpoint_rel"],
                    "run_dir": cell["run_dir_rel"],
                    "seeds_available": len(group),
                    "test_split_read": False,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Reference parity (audit only)
# ---------------------------------------------------------------------------
def _same_checkpoint(reference_path: str, mine_path: str) -> bool:
    reference_path = str(reference_path).replace("\\", "/").strip()
    mine_path = str(mine_path).replace("\\", "/").strip()
    if not reference_path or not mine_path:
        return False
    if reference_path == mine_path:
        return True
    return (
        reference_path.split("/attempts/")[0] == mine_path.split("/attempts/")[0]
    )


def reference_parity(payloads: list[dict], args, output_dir: Path) -> dict:
    """Compare the recomputed probe cells with the registered artifacts.

    Reuse is deliberately *not* taken for the probe columns: E14's reuse
    resolution and the low-rank inventory pick one checkpoint per cell with
    different tie-breaks, so the dissected artifact can legitimately differ.  The
    value comparison therefore only runs where the checkpoint (or at least its
    run directory) matches, and the mismatching cells are listed instead of being
    hidden.
    """
    reference_root = REPO_ROOT / args.reference_root
    report = {
        "reference_root": str(args.reference_root),
        "reference_available": reference_root.is_dir(),
        "probe_cells": 0,
        "cells_value_compared": 0,
        "checkpoint_path_mismatches": [],
        "fields": {},
        "passed": True,
    }
    if not reference_root.is_dir():
        return report
    inventory: dict[tuple, str] = {}
    inventory_path = reference_root / "checkpoint_inventory.csv"
    if inventory_path.is_file():
        with inventory_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                inventory[
                    (row.get("setting"), str(row.get("seed")), row.get("cell"))
                ] = row.get("checkpoint_path", "")
    mine = {
        (payload["cell"]["setting"], str(payload["cell"]["seed"]), payload["cell"]["q_label"]): payload
        for payload in payloads
        if payload["spec"]["head_kind"] == "pooled_lowrank"
    }
    seen_mismatch: set[tuple] = set()
    for filename, fields, source in (
        ("canonical_modes.csv",
         ["singular_value", "singular_value_share", "correction_energy_share", "latent_variance"],
         "canonical_rows"),
        ("semantic_alignment.csv",
         ["input_group_explanation", "output_group_explanation", "input_dictionary_r2",
          "output_dictionary_r2"],
         "semantic_rows"),
    ):
        path = reference_root / filename
        if not path.is_file():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("mode_index", "0")) != "0":
                    continue
                lookup = (row["setting"], str(row["seed"]), row["cell"])
                payload = mine.get(lookup)
                if payload is None:
                    continue
                report["probe_cells"] += 1
                expected = inventory.get(lookup, "")
                actual = payload["cell"]["checkpoint_rel"]
                if expected and not _same_checkpoint(expected, actual):
                    if lookup not in seen_mismatch:
                        seen_mismatch.add(lookup)
                        report["checkpoint_path_mismatches"].append(
                            {
                                "setting": lookup[0],
                                "seed": lookup[1],
                                "cell": lookup[2],
                                "reference": expected,
                                "e16": actual,
                            }
                        )
                    continue
                origin = payload[source][0]
                report["cells_value_compared"] += 1
                for field in fields:
                    if field not in origin:
                        continue
                    try:
                        diff = abs(float(origin[field]) - float(row[field]))
                    except (TypeError, ValueError):
                        diff = float("inf")
                    entry = report["fields"].setdefault(
                        field, {"max_abs_diff": 0.0, "comparisons": 0, "tolerance_abs": 1e-6}
                    )
                    entry["max_abs_diff"] = max(entry["max_abs_diff"], diff)
                    entry["comparisons"] += 1
    for entry in report["fields"].values():
        if entry["max_abs_diff"] > entry["tolerance_abs"]:
            report["passed"] = False
    (output_dir / "reference_parity.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def serialise(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()
    if args.evaluation_split not in ("val", "validation"):
        raise SystemExit("only the validation split can be evaluated by E16")
    args.evaluation_split = "val"

    cells = build_cell_plan(args)
    if args.dry_run:
        by_arm: dict[str, int] = {}
        by_status: dict[str, int] = {}
        for cell in cells:
            by_arm[cell["arm"]] = by_arm.get(cell["arm"], 0) + 1
            by_status[cell["e14_status"]] = by_status.get(cell["e14_status"], 0) + 1
        print(
            json.dumps(
                {
                    "event": "plan",
                    "cells": len(cells),
                    "by_arm": by_arm,
                    "by_e14_status": by_status,
                    "seeds": sorted({cell["seed"] for cell in cells}),
                    "settings": sorted({cell["setting"] for cell in cells}),
                    "evaluation_split": "val",
                    "test_split_read": False,
                    "random_repeats": int(args.random_repeats),
                    "random_rrr_repeats": int(args.random_rrr_repeats),
                    "random_rrr": bool(args.random_rrr),
                    "mem_budget_mb": float(args.mem_budget_mb),
                },
                ensure_ascii=False,
            )
        )
        for cell in cells:
            print(
                f"{cell['arm']:<8} {cell['setting']:<16} seed={cell['seed']} "
                f"status={cell['e14_status']:<14} rank={cell['lowrank_rank']:<4} "
                f"{cell['checkpoint_rel']}",
                flush=True,
            )
        return

    output_dir = REPO_ROOT / args.output_root
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        gpu = (parse_list(args.gpus, int) or [0])[0]
        torch.cuda.set_device(gpu)
        device = torch.device("cuda", gpu)
    print(f"device: {device}", flush=True)

    registry = build_registry(load_dataset_info())
    unknown = sorted({cell["dataset"] for cell in cells} - set(registry))
    if unknown:
        raise SystemExit(f"datasets not present in DATASET_INFO: {unknown}")

    print(
        f"[scope] {len(cells)} cells = {sorted({c['arm'] for c in cells})} x "
        f"{sorted({c['setting'] for c in cells})} x {sorted({c['seed'] for c in cells})}"
        " (settings are the test-selected set of the schedule; disclosed, not blind)",
        flush=True,
    )

    started = time.time()
    train_stats: dict[tuple[str, int], dict] = {}
    dictionaries: dict[tuple[str, int], SettingDictionary] = {}
    payloads: list[dict] = []
    # Models are keyed by arm as well: l_main carries the dense head while
    # l_q1_4/l_q1_8 carry the pooled one, so their state dicts differ (a strict
    # load into the wrong model would fail) and D-2 gives each arm its own frozen
    # (gate, lr).  Loaders are shared per (setting, batch size).
    models: dict[tuple[str, int, int, str], object] = {}
    loaders: dict[tuple[str, int, int], dict] = {}
    cached_setting: tuple[str, int] | None = None

    for cell in cells:
        dataset = cell["dataset"]
        horizon = int(cell["horizon"])
        seed = int(cell["seed"])
        setting_key = (dataset, horizon)
        if setting_key not in train_stats:
            train_stats[setting_key] = train_statistics(dataset, horizon, args, registry)
            period = DATASET_PERIOD_STEPS[dataset]
            input_groups = build_groups(input_templates(LOOKBACK, period))
            output_groups = build_groups(output_templates(horizon, period))
            dictionaries[setting_key] = SettingDictionary(
                setting=f"{dataset}-{horizon}",
                dataset=dataset,
                horizon=horizon,
                period=period,
                input_groups=input_groups,
                output_groups=output_groups,
                input_group_order=list(input_groups),
                output_group_order=list(output_groups),
                input_templates=[t for group in input_groups.values() for t in group.templates],
                output_templates=[t for group in output_groups.values() for t in group.templates],
                input_shapley={},
                output_shapley={},
                moments=train_stats[setting_key]["moments"],
                output_moments=None,
            )
        if cached_setting != setting_key:
            models.clear()  # keep at most one setting's artefacts alive
            loaders.clear()
            cached_setting = setting_key
        loader_key = (dataset, horizon, int(cell["batch_size"]))
        if loader_key not in loaders:
            # ``build_loaders`` writes ``time_mark_dim`` into the hyperparams dict
            # it is given, so the *same object* has to reach ``build_model`` --
            # which is exactly what the registered evaluator does.
            hyperparams = dict(cell["hyperparams"])
            exp_args, handles = build_loaders(
                dataset, LOOKBACK, horizon, hyperparams, cell["batch_size"],
                REPO_ROOT, splits=(args.evaluation_split,),
            )
            if int(args.num_workers) != int(getattr(exp_args.dataset_args, "num_workers", 0)):
                # ``build_loaders`` forces num_workers=0; rebuild the loader with
                # the requested worker count instead of ignoring the flag.
                exp_args.dataset_args.num_workers = int(args.num_workers)
                exp_args.training_args.num_workers = int(args.num_workers)
                handles[args.evaluation_split] = data_provider(
                    exp_args.dataset_args, args.evaluation_split
                )
            loaders[loader_key] = {
                "exp_args": exp_args,
                "hyperparams": hyperparams,
                "loader": handles[args.evaluation_split][1],
            }
        bundle = loaders[loader_key]
        model_key = (dataset, horizon, seed, cell["arm"])
        if model_key not in models:
            models[model_key] = build_model(
                bundle["exp_args"], LOOKBACK, horizon, bundle["hyperparams"]
            )
        model = models[model_key]
        state = torch.load(cell["checkpoint"], map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        model.load_state_dict(state, strict=True)
        spec = describe_branch(model, state)
        model.to(device).eval()
        cell_started = time.time()
        payload = run_cell(
            cell, spec, model, bundle["loader"], device,
            dictionaries[setting_key], train_stats[setting_key], args,
        )
        payloads.append(payload)
        del state
        print(
            f"[cell] {cell['arm']:<8} {cell['setting']:<16} seed={seed} "
            f"head={spec['head_kind']:<16} rank_dim={spec['rank_dim']:<4} "
            f"pairs={payload['statistics']['pairs']} "
            f"invariant={'ok' if payload['invariant']['untouched_arm_reproduces_run_metric'] else 'FAILED'} "
            f"gap={payload['invariant']['untouched_arm_gap_vs_run_metric']:.2e} "
            f"({time.time() - cell_started:.1f}s)",
            flush=True,
        )

    cross_rows = cross_seed_rows(payloads)
    intervention_by_key: dict[tuple, dict] = {}
    intervention_rows: list[dict] = []
    for payload in payloads:
        cell = payload["cell"]
        for row in payload["intervention_rows"]:
            intervention_by_key[
                (cell["setting"], cell["arm"], int(cell["seed"]), row["intervention_arm"])
            ] = row
            intervention_rows.append(row)
    dissection = dissection_rows(payloads, intervention_by_key)
    canonical_rows = [row for payload in payloads for row in payload["canonical_rows"]]
    semantic_rows = [row for payload in payloads for row in payload["semantic_rows"]]

    write_csv(output_dir / "dissection_table.csv", dissection)
    write_csv(output_dir / "intervention_table.csv", intervention_rows)
    write_csv(output_dir / "canonical_modes.csv", canonical_rows)
    write_csv(output_dir / "semantic_alignment.csv", semantic_rows)
    write_csv(output_dir / "cross_seed_alignment.csv", cross_rows)

    parity = (
        {"skipped": True}
        if args.skip_reference_parity
        else reference_parity(payloads, args, output_dir)
    )

    verdicts: dict[str, int] = {}
    for row in dissection:
        verdicts[row["stable_semantics_verdict"]] = (
            verdicts.get(row["stable_semantics_verdict"], 0) + 1
        )
    invariant_failures = [
        f"{payload['cell']['arm']}|{payload['cell']['setting']}|{payload['cell']['seed']}"
        for payload in payloads
        if not payload["invariant"]["untouched_arm_reproduces_run_metric"]
    ]
    summary = {
        "experiment": "E16 (minipaper 4.4 dissection + intervention table)",
        "reads_test": False,
        "evaluation_split": args.evaluation_split,
        "scope": {
            "settings": sorted({cell["setting"] for cell in cells}),
            "seeds": sorted({cell["seed"] for cell in cells}),
            "arms": sorted({cell["arm"] for cell in cells}),
            "test_selected_settings": [[dataset, horizon] for dataset, horizon in TEST_SELECTED_SETTINGS],
            "cells": len(cells),
        },
        "protocol": {
            "lookback": sorted({cell["lookback"] for cell in cells}, key=str),
            "period": sorted({cell["period"] for cell in cells}, key=str),
            "loss": sorted({cell["loss"] for cell in cells}),
            "max_epochs": sorted({cell["max_epochs"] for cell in cells}, key=str),
            "percent": sorted({cell["percent"] for cell in cells}, key=str),
            "checkpoint": "best validation loss (attempts/*/checkpoints/best.ckpt)",
            "gate": "fixed sigmoid (adaptive gates are rejected)",
            "smooth_ratio": sorted({cell["smooth_ratio"] for cell in cells}, key=str),
            "head_type": sorted({cell["head_type"] for cell in cells}),
            "mechanism": sorted({cell["mechanism"] for cell in cells}),
            "learning_rate": sorted({str(cell["learning_rate"]) for cell in cells}),
            "gate_init": sorted({str(cell["gate_init"]) for cell in cells}),
        },
        "random_controls": {
            "random_band": {
                "family": "whole latent space",
                "dimension_rule": "legacy max(1, min(rank_dim, 6))",
                "repeats": int(args.random_repeats),
                "seed": RANDOM_SEED,
            },
            "random_ambient_matched_band": {
                "family": "whole latent space",
                "dimension_rule": "Semantic-drop dimension",
                "repeats": int(args.random_repeats),
                "seed": BAND_SEED_AMBIENT_MATCHED,
            },
            "random_rrr_band": {
                "enabled": bool(args.random_rrr),
                "family": "RRR-achievable subspace (Stage 3 independent RRR, or the "
                          "E16 train-split fit where no subspace file exists)",
                "dimension_rule": "min(Semantic-drop dimension, pool dimension)",
                "repeats": int(args.random_rrr_repeats),
                "seed": RANDOM_SEED + 1,
                "arm_reference_seed": RANDOM_SEED + 2,
                "pool_space_for_e16_fits": args.rrr_pool_space,
                "ridge": 1e-6,
            },
        },
        "memory": {
            "mem_budget_mb": float(args.mem_budget_mb),
            "forced_channel_block": int(args.channel_block),
            "channel_block_per_cell": {
                f"{payload['cell']['arm']}|{payload['cell']['setting']}|{payload['cell']['seed']}":
                    payload["svd"]["channel_block"]
                for payload in payloads
            },
            "band_block_elements_per_cell": {
                f"{payload['cell']['arm']}|{payload['cell']['setting']}|{payload['cell']['seed']}":
                    payload["svd"]["band_block_elements"]
                for payload in payloads
            },
        },
        "train_statistics": {
            f"{key[0]}-{key[1]}": {
                "pairs": stats["train_pairs"],
                "windows": stats["train_windows"],
                "channels": stats["channels"],
                "channel_block": stats["channel_block"],
                "csv_rows_parsed": stats["csv_rows_parsed"],
                "test_border_row": stats["test_border_row"],
                "test_split_read": False,
            }
            for key, stats in sorted(train_stats.items())
        },
        "counts": {
            "cells": len(cells),
            "canonical_mode_rows": len(canonical_rows),
            "semantic_alignment_rows": len(semantic_rows),
            "intervention_rows": len(intervention_rows),
            "dissection_rows": len(dissection),
            "cross_seed_rows": len(cross_rows),
            "intervention_arms_per_cell": sorted({len(payload["intervention_rows"]) for payload in payloads}),
            "verdicts": verdicts,
        },
        "invariants": {
            "untouched_arm_failures": invariant_failures,
            "per_cell": {
                f"{payload['cell']['arm']}|{payload['cell']['setting']}|{payload['cell']['seed']}":
                    payload["invariant"]
                for payload in payloads
            },
        },
        "cells": [
            {
                "arm": payload["cell"]["arm"],
                "setting": payload["cell"]["setting"],
                "dataset": payload["cell"]["dataset"],
                "horizon": payload["cell"]["horizon"],
                "seed": payload["cell"]["seed"],
                "q_or_r": payload["cell"]["q_label"],
                "lowrank_rank": payload["cell"]["lowrank_rank"],
                "head_kind": payload["spec"]["head_kind"],
                "hidden_kind": payload["spec"]["hidden_kind"],
                "rank_dim": payload["spec"]["rank_dim"],
                "e14_status": payload["cell"]["e14_status"],
                "run_dir": payload["cell"]["run_dir_rel"],
                "checkpoint": payload["cell"]["checkpoint_rel"],
                "config_hash": payload["cell"]["config_hash"],
                "checkpoint_source": payload["cell"]["checkpoint_source"],
                "n_alternative_run_dirs": payload["cell"]["n_alternative_run_dirs"],
                "alternative_run_dirs": payload["cell"]["alternative_run_dirs"],
                "selected_val_mse": payload["cell"]["selected_val_mse"],
                "run_mse_csv_has_test_metric": payload["cell"]["run_mse_csv_has_test_metric"],
                "rrr_pool_source": payload["bases"]["pool_source"],
                "rrr_pool_dimension": payload["bases"]["pool_dimension"],
                "rrr_pool_covers_ambient": payload["bases"]["pool_covers_ambient"],
                "rrr_dimension": payload["bases"]["rrr_dimension"],
                "semantic_dimension": payload["bases"]["semantic_dimension"],
                "intervention_arms": [row["intervention_arm"] for row in payload["intervention_rows"]],
            }
            for payload in payloads
        ],
        "reference_parity": parity,
        "disclosures": [
            "The seven settings are test-set selected (schedule section 2.1); no "
            "number in this unit is a blind estimate.",
            "The test split is never read: --evaluation-split only accepts val, "
            "only the validation loader is built, and the train split is parsed up "
            "to the validation border.",
            "l_main rows reuse the E3-lineage Stage-0 frozen (gate, lr) under D-2, "
            "so the dense arm and the probe arms are not one hyperparameter "
            "agreement.",
            "RandomRRR-drop is the control minipaper section 4.4 adds; the legacy "
            "whole-space band is kept and is NOT dimension-matched to "
            "Semantic-drop.",
            "Probe cells take their RRR pool from the Stage 3 independent-RRR "
            "subspace file (the branch's RevIN-normalized space); dense cells have "
            "no such file and get an E16 train-split fit with the same estimator "
            "and ridge in float64.",
            "A pooled RRR of the branch's own rank spans its whole latent space, so "
            "for every rank-bottlenecked probe cell random_rrr_pool_covers_ambient "
            "is true and the RRR family coincides with the ambient one; the "
            "discriminating cells are the dense ones.",
            "The arm algebra adds the mapped encoder bias (W_dec @ encoder_bias) "
            "to a hidden state that already contains it, exactly as the registered "
            "evaluator does; the resulting offset of the untouched arm from the "
            "model's own output is reported per cell as "
            "untouched_arm_gap_vs_recorded_fused and is zero for the dense head.",
            "The output-side covariance metric is estimated from this cell's own "
            "validation residuals rather than from the per-setting pooled residuals "
            "of the registered analyzer; only output_covariance_correlation_max is "
            "affected.",
        ],
        "elapsed_seconds": time.time() - started,
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
    }
    (output_dir / "e16_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, default=serialise) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cell_plan.json").write_text(
        json.dumps(
            [
                {
                    key: serialise(value)
                    for key, value in cell.items()
                    if key not in ("hyperparams", "arms", "bases")
                }
                for cell in cells
            ],
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=serialise,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "finished",
                "cells": len(cells),
                "intervention_rows": len(intervention_rows),
                "dissection_rows": len(dissection),
                "verdicts": verdicts,
                "invariant_failures": len(invariant_failures),
                "reference_parity_passed": parity.get("passed", None),
                "elapsed_seconds": round(summary["elapsed_seconds"], 1),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
