#!/usr/bin/env python3
"""E18 row 3: SVD truncation vs rank-constrained training on all 28 settings.

Implements the third row of minipaper §4.6 ("SVD 截断 vs 秩约束训练", the row that
`docs/PhaseFormer_L_execution_schedule.md` §2.3 tasks with extending the
comparison from 7 to **28 settings**).  No training happens here: the script
reads checkpoints that E14 already produced.

What is compared, per setting and per rank ``r``:

* **truncated** — take the *full-rank trained* head's weight
  ``weak_period_residual.linear.weight`` (``H x 720``) from E14's ``l_main``
  cell, replace it by its best rank-``r`` SVD approximation
  ``U[:, :r] @ diag(S[:r]) @ Vh[:r, :]`` (the transformation Experiment 2 of
  ``scripts/analyze_weak_residual_svd_truncation_eval.py`` used), load it into a
  freshly built model, and evaluate.  No gradient step.
* **trained low-rank** — the head E14 actually trained under the rank
  constraint, from its ``l_q1_4`` / ``l_q1_8`` cells
  (``pooled_lowrank``, ``pool_factor=1``, ``rank = H/4`` or ``H/8``).

A small gap means the full-rank solution was already close to rank ``r``; a
large gap means rank-constrained *training* finds a solution that plain
truncation cannot reach.  Electricity-336 at ``r=10`` is the registered
counterexample (truncated +29% vs trained +0.7%).

Evaluation split
----------------

**This script only evaluates the validation split.**  ``--evaluation-split``
accepts ``val``/``validation`` and rejects anything else, and no test loader is
ever constructed.  That matters because the published E11 numbers behind
``docs/PhaseFormer_lowrank_mechanism_analysis.md`` §3.2 are **test-based**: that
analysis evaluated the real test set
(``scripts/analyze_weak_residual_svd_truncation_eval.py`` docstring: "evaluate
real test-set MSE/MAE").  The 28-setting table therefore carries
validation-based numbers and is *not* numerically interchangeable with E11's 7
test-based rows; the difference is disclosed in every output (``split`` column,
``test_split_read: false``, and the ``e11_comparability`` block of the summary).

Truncation ranks
----------------

``--ranks`` defaults to ``10``, E11's anchor rank: Electricity-336's
counterexample was measured at ``r=10``, so the fixed rank keeps the one
quantity the minipaper quotes comparable while covering all 28 settings.  The
tooling rank ``pool_factor=1`` makes the trained head's rank the number the
minipaper quotes (``q=1/4`` -> ``H/4``, ``q=1/8`` -> ``H/8``), so each setting
also reports its own trained ranks and a ``comparison_rank`` chosen for the
closest match to the primary rank; pass ``--ranks 10,<native>`` to evaluate more
than the anchor.

Rank-limited (non-shared) full-rank heads
-----------------------------------------

Only the dense ``shared`` head carries a single ``H x 720`` weight.  If a cell
resolves to another head type the run is reported as
``unsupported_head_type`` instead of being truncated silently.

Usage::

    # static check: resolve the 28 settings, evaluate nothing
    python scripts/phaseformer_L/e18_svd_truncation.py \
        --e14-root research_runs/phaseformer_L_e14_main_v1 \
        --output-root research_runs/phaseformer_L_e18_negative_v1 --dry-run

    # the 28-setting analysis (validation split only)
    python scripts/phaseformer_L/e18_svd_truncation.py \
        --e14-root research_runs/phaseformer_L_e14_main_v1 \
        --output-root research_runs/phaseformer_L_e18_negative_v1 \
        --ranks 10 --seeds 2021
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# stdlib-only sibling module: importing it keeps "which run is l_main / l_q1_8"
# defined in exactly one place (the §4.2 matrix owner).
from scripts.phaseformer_L.e14_main_matrix import (  # noqa: E402
    ARM_DATASET_EXCLUSIONS,
    ARMS,
    HORIZONS,
    LOOKBACK,
    LOSS,
    MAIN_DATASETS,
    MAX_EPOCHS,
    PERCENT,
    PERIOD,
    SEEDS,
    TRAFFIC_DATASETS,
    _arm_match,
    parse_list,
)

# The 28 settings of the §4.6 row-3 extension: 7 datasets x 4 horizons.
# Traffic MUST be included: minipaper §4.6 row 3 says "全 28 setting", and the
# 24-setting main table plus the 4-setting Traffic appendix is exactly the 28.
# Defaulting to MAIN_DATASETS alone silently produced 24 cells, which would have
# been reported as "all 28" while omitting a whole dataset.
ALL_DATASETS = tuple(MAIN_DATASETS) + tuple(TRAFFIC_DATASETS)
SETTINGS = tuple(
    (dataset, horizon) for dataset in ALL_DATASETS for horizon in HORIZONS
)
FULL_RANK_ARM = "l_main"
LOW_RANK_ARMS = ("l_q1_4", "l_q1_8")
# Absolute-rank controls that must never be confused with the relative grid.
ABSOLUTE_RANK_OVERRIDE_NOTE = (
    "l_q1_4/l_q1_8 give rank = H/4 / H/8; the E18 row-5 ablation uses absolute "
    "rank 1/2 and lives in e18_negative.py, not here"
)

ALLOWED_SPLITS = ("val", "validation")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--e14-root", default="research_runs/phaseformer_L_e14_main_v1",
                        help="root holding E14's stage_a_manifest.json and runs/")
    parser.add_argument("--manifest", default="",
                        help="explicit stage_a_manifest.json (overrides --e14-root)")
    parser.add_argument("--output-root",
                        default="research_runs/phaseformer_L_e18_negative_v1")
    parser.add_argument("--datasets", default=",".join(ALL_DATASETS))
    parser.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    parser.add_argument("--seeds", default=str(SEEDS[0]),
                        help="E11's anchor seed is 2021; more seeds multiply "
                             "the number of validation passes")
    parser.add_argument("--ranks", default="10",
                        help="absolute truncation ranks to evaluate; E11's "
                             "anchor is 10")
    parser.add_argument("--evaluation-split", default="val",
                        choices=list(ALLOWED_SPLITS),
                        help="only val/validation; the test split is never read")
    parser.add_argument("--max-eval-samples", type=int, default=0,
                        help="cap validation samples per setting (0 = all)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="resolve every cell's checkpoint before evaluating")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# E14 manifest and run directories
# --------------------------------------------------------------------------

def load_manifest(args) -> tuple:
    path = (Path(args.manifest) if args.manifest
            else ROOT / args.e14_root / "stage_a_manifest.json")
    if not path.is_file():
        raise SystemExit(
            f"E14 stage-A manifest not found: {path}\n"
            "E18 row 3 depends on E14 (schedule section 4.1).  Run "
            "scripts/phaseformer_L/e14_main_matrix.py --stage a first, or point "
            "--e14-root/--manifest at the root that holds stage_a_manifest.json."
        )
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SystemExit(f"cannot parse {path}: {error}") from error
    if not isinstance(manifest.get("cells"), list):
        raise SystemExit(f"{path} has no 'cells' list")
    return manifest, path


def repo_relative(path: Path) -> str:
    """Repo-relative path when the file is inside the repo, else absolute.

    Run directories and manifests can sit outside the working copy (a relocated
    ``--e14-root``, a test fixture), so a bare ``relative_to`` must never be able
    to abort the analysis.
    """
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics.csv"
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        return next(csv.DictReader(handle), {}) or {}


def read_config(run_dir: Path) -> dict:
    path = run_dir / "config.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def arm_cells(manifest: dict, arm: str) -> dict:
    """Index a manifest's cells for one arm by ``(dataset, horizon, seed)``.

    A manifest entry counts only when its own command does not carry
    ``--evaluate-test`` (stage A must not have touched test) and its recorded
    arm is the requested one.
    """
    index: dict = {}
    for entry in manifest["cells"]:
        if entry.get("arm") != arm:
            continue
        argv = entry.get("command") or []
        if "--evaluate-test" in argv:
            continue
        key = (entry["dataset"], int(entry["horizon"]), int(entry["seed"]))
        index.setdefault(key, entry)
    return index


def resolve_run_dir(entry: dict, e14_root: Path) -> tuple:
    """Run directory of a manifest cell, and how it was found.

    ``reused`` cells point at an external lineage run dir (``entry['source']``),
    ``new`` cells point at a run under E14's own output root (the manifest
    carries no run dir for them, because the run id embeds a config hash that is
    only known after the run).  A run directory is only accepted when its
    ``config.json`` reproduces the arm fingerprint ``_arm_match`` defines.
    """
    candidates: list = []
    source = entry.get("source") or {}
    if isinstance(source, dict) and source.get("run_dir"):
        recorded = Path(source["run_dir"])
        candidates.append(recorded if recorded.is_absolute() else ROOT / recorded)
    argv = entry.get("command") or []
    out_root = e14_root
    if "--output-dir" in argv:
        recorded = Path(argv[argv.index("--output-dir") + 1])
        out_root = recorded if recorded.is_absolute() else ROOT / recorded
    for runs_dir in (out_root / "runs", e14_root / "runs"):
        if not runs_dir.is_dir():
            continue
        for config_path in sorted(runs_dir.glob("*/config.json")):
            config = read_config(config_path.parent)
            if not config:
                continue
            if (config.get("dataset") != entry["dataset"]
                    or int(config.get("horizon", -1)) != int(entry["horizon"])
                    or int(config.get("seed", -1)) != int(entry["seed"])):
                continue
            if _arm_match(config, entry["arm"]):
                candidates.append(config_path.parent)
    seen = set()
    unique = []
    for candidate in candidates:
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    for candidate in unique:
        if (candidate / "metrics.csv").is_file():
            from_source = bool(
                isinstance(source, dict) and source.get("run_dir")
                and candidate.resolve() == (
                    ROOT / source["run_dir"]).resolve())
            return candidate, ("manifest_source" if from_source
                               else "e14_runs_glob")
    for candidate in unique:
        if (candidate / "config.json").is_file():
            return candidate, "config_only"
    return None, ""


def checkpoint_of(run_dir: Path, metrics: dict) -> tuple:
    """Best-validation checkpoint of a run, from its own ``metrics.csv``.

    The recorded repo-relative path is authoritative; the ``attempts/*`` and
    ``checkpoints/`` globs are fallbacks for relocated run directories.
    """
    recorded = str(metrics.get("checkpoint", "")).strip()
    if recorded:
        raw = Path(recorded)
        candidates = [ROOT / raw]
        if raw.is_absolute():
            candidates.append(raw)
        parts = raw.parts
        for index, part in enumerate(parts):
            if part == "runs" and index + 1 < len(parts):
                candidates.append(run_dir / Path(*parts[index + 1:]))
                break
        candidates.append(run_dir / "checkpoints" / raw.name)
        candidates.extend(sorted((run_dir / "attempts").glob(
            f"*/checkpoints/{raw.name}")))
        for candidate in candidates:
            if candidate.is_file():
                return candidate, "metrics_csv"
    globbed = sorted(run_dir.glob("attempts/*/checkpoints/best.ckpt"))
    if globbed:
        return globbed[-1], "attempts_glob"
    return None, ""


def build_plan(args) -> tuple:
    """Resolve, for every (setting, seed), its full-rank and low-rank cells."""
    manifest, manifest_path = load_manifest(args)
    e14_root = manifest_path.parent
    datasets = parse_list(args.datasets)
    horizons = parse_list(args.horizons, int)
    seeds = parse_list(args.seeds, int)
    ranks = parse_list(args.ranks, int)

    full_index = arm_cells(manifest, FULL_RANK_ARM)
    low_index = {arm: arm_cells(manifest, arm) for arm in LOW_RANK_ARMS}

    plan: list = []
    problems: list = []
    for dataset, horizon in SETTINGS:
        if dataset not in datasets or horizon not in horizons:
            continue
        if dataset in ARM_DATASET_EXCLUSIONS.get(FULL_RANK_ARM, set()):
            continue
        for seed in seeds:
            key = (dataset, horizon, seed)
            entry = full_index.get(key)
            record = {
                "dataset": dataset,
                "horizon": horizon,
                "seed": seed,
                "full_rank_arm": FULL_RANK_ARM,
                "full_rank_run_dir": None,
                "full_rank_source": None,
                "full_rank_checkpoint": None,
                "full_rank_checkpoint_rule": None,
                "full_rank_head_type": None,
                "full_rank_gate_init": None,
                "full_rank_learning_rate": None,
                "full_rank_cell_status": None,
                "low_rank": [],
                "problems": [],
            }
            if entry is None:
                record["problems"].append("missing_full_rank_cell")
            else:
                run_dir, rule = resolve_run_dir(entry, e14_root)
                record["full_rank_cell_status"] = entry.get("status")
                if run_dir is None:
                    record["problems"].append("full_rank_run_dir_unresolved")
                else:
                    metrics = read_metrics(run_dir)
                    config = read_config(run_dir)
                    hyper = config.get("hyperparams", {}) or {}
                    head = hyper.get("weak_period_residual_head_type", "shared")
                    record["full_rank_run_dir"] = repo_relative(run_dir)
                    record["full_rank_source"] = rule
                    record["full_rank_head_type"] = head
                    record["full_rank_gate_init"] = hyper.get(
                        "weak_period_residual_gate_init")
                    record["full_rank_learning_rate"] = hyper.get("learning_rate")
                    record["full_rank_val_mse"] = metrics.get("val_mse", "")
                    record["full_rank_val_mae"] = metrics.get("val_mae", "")
                    if head != "shared":
                        record["problems"].append(
                            f"unsupported_head_type:{head}")
                    checkpoint, ckpt_rule = checkpoint_of(run_dir, metrics)
                    if checkpoint is None:
                        record["problems"].append("full_rank_checkpoint_missing")
                    else:
                        record["full_rank_checkpoint"] = repo_relative(checkpoint)
                        record["full_rank_checkpoint_rule"] = ckpt_rule
            for arm in LOW_RANK_ARMS:
                low_entry = low_index[arm].get(key)
                if low_entry is None:
                    record["problems"].append(f"missing_{arm}_cell")
                    continue
                low_dir, low_rule = resolve_run_dir(low_entry, e14_root)
                if low_dir is None:
                    record["problems"].append(f"{arm}_run_dir_unresolved")
                    continue
                low_metrics = read_metrics(low_dir)
                low_config = read_config(low_dir)
                low_hyper = low_config.get("hyperparams", {}) or {}
                low_rank = low_hyper.get("weak_period_residual_rank")
                low_record = {
                    "arm": arm,
                    "rank": (int(low_rank) if low_rank is not None else None),
                    "run_dir": repo_relative(low_dir),
                    "source": low_rule,
                    "cell_status": low_entry.get("status"),
                    "head_type": low_hyper.get("weak_period_residual_head_type"),
                    "pool_factor": low_hyper.get("weak_period_residual_pool_factor"),
                    "gate_init": low_hyper.get("weak_period_residual_gate_init"),
                    "learning_rate": low_hyper.get("learning_rate"),
                    "val_mse": low_metrics.get("val_mse", ""),
                    "val_mae": low_metrics.get("val_mae", ""),
                }
                if low_record["rank"] is None:
                    record["problems"].append(f"{arm}_rank_unresolved")
                record["low_rank"].append(low_record)
            record["ranks"] = list(ranks)
            plan.append(record)
            problems.extend({"setting": f"{dataset}-{horizon}", "seed": seed,
                             "problem": problem}
                            for problem in record["problems"])
    return plan, problems, manifest_path, seeds, ranks


def comparison_rank(low_record: dict, primary_rank: int) -> tuple:
    """Trained low-rank cell whose rank is closest to the primary truncation rank."""
    candidates = [item for item in low_record
                  if isinstance(item.get("rank"), int)]
    if not candidates:
        return None, ""
    best = min(candidates, key=lambda item: (abs(item["rank"] - primary_rank),
                                             item["rank"]))
    return best, best["arm"]


# --------------------------------------------------------------------------
# Evaluation (torch is imported lazily so --dry-run/--verify need no GPU stack)
# --------------------------------------------------------------------------

def pick_device(choice: str):
    import torch  # noqa: F401

    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def svd_truncate(weight, rank: int):
    """Best rank-``rank`` SVD approximation, verbatim from Experiment 2.

    Same algebra as ``scripts/analyze_weak_residual_svd_truncation_eval.py:41-44``.
    """
    import torch

    u, s, vh = torch.linalg.svd(weight.double(), full_matrices=False)
    truncated = u[:, :rank] @ torch.diag(s[:rank]) @ vh[:rank, :]
    return truncated.to(weight.dtype)


def build_head_and_loader(run_dir: Path, config: dict, split: str,
                          num_workers: int, max_eval_samples: int):
    """Rebuild one run's model and one split's loader.

    ``split`` is one of ``ALLOWED_SPLITS`` (validation only).  The batch size is
    the run's own recorded batch size, so the recomputed metric follows the same
    aggregation the training loop used for its best-checkpoint selection.
    """
    from src.dataset.data_factory import data_provider
    from src.models.PhaseFormer import PhaseFormer
    from src.models.phaseformer_presets import (
        PhaseFormerPresetConfig,
        make_exp_args,
    )

    hp = dict(config["hyperparams"])
    metrics = read_metrics(run_dir)
    batch_size = int(metrics.get("batch_size") or config.get("batch_size")
                     or 0) or None
    exp_args = make_exp_args(config["dataset"], config["lookback"],
                             config["horizon"], hp, batch_size=batch_size)
    exp_args.dataset_args.num_workers = num_workers
    dataset, loader = data_provider(exp_args.dataset_args, split)
    if max_eval_samples:
        from torch.utils.data import DataLoader, Subset

        dataset = Subset(dataset, range(min(max_eval_samples, len(dataset))))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=False)
    model = PhaseFormer(PhaseFormerPresetConfig(
        exp_args, config["lookback"], config["horizon"], hp))
    return model, loader, dataset, batch_size


def evaluate_model(model, loader, horizon: int, device) -> dict:
    """MSE/MAE on an already-built loader.

    The body is a verbatim copy of the metric loop in
    ``scripts/search_phaseformer.py::evaluate`` (lines 478-506) with the split
    argument dropped, so the recomputed number is the same estimator the run
    recorded as ``val_mse``/``val_mae``.
    """
    import torch

    model.to(device).eval()
    abs_sum = 0.0
    sq_sum = 0.0
    count = 0
    with torch.inference_mode():
        for batch in loader:
            batch_x, batch_y, batch_x_mark, batch_y_mark = [
                value.to(device).float() for value in batch
            ]
            dec = model._build_decoder_input(batch_y)
            out, _, _ = model(batch_x, batch_x_mark, dec, batch_y_mark)
            pred = out[:, -horizon:, :]
            true = batch_y[:, -horizon:, :]
            if model.target_var_index != -1:
                index = model.target_var_index
                true = true[:, :, index:index + 1]
            err = pred - true
            abs_sum += torch.abs(err).sum().item()
            sq_sum += torch.square(err).sum().item()
            count += err.numel()
    if not count:
        raise RuntimeError("evaluation produced zero elements")
    return {"mse": sq_sum / count, "mae": abs_sum / count, "count": count}


def load_state_dict(checkpoint: Path) -> dict:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return payload["state_dict"]


def relative_gap(value, reference) -> float:
    try:
        reference = float(reference)
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if reference == 0:
        return float("nan")
    return 100.0 * (value - reference) / reference


def evaluate_setting(record: dict, args, device) -> list:
    """All rank rows of one (setting, seed) pair."""
    import torch

    run_dir = ROOT / record["full_rank_run_dir"]
    config = read_config(run_dir)
    full_state = load_state_dict(ROOT / record["full_rank_checkpoint"])
    full_weight = full_state["weak_period_residual.linear.weight"]

    model, loader, dataset, batch_size = build_head_and_loader(
        run_dir, config, args.evaluation_split, args.num_workers,
        args.max_eval_samples)
    model.load_state_dict(full_state, strict=True)
    full_rank_metrics = evaluate_model(model, loader, record["horizon"], device)
    if not math.isfinite(full_rank_metrics["mse"]):
        raise RuntimeError(f"non-finite full-rank metric for {record}")
    # Free the full-rank model before the truncated copies.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    comparison, comparison_arm = comparison_rank(
        record["low_rank"], min(record["ranks"]))
    rows = []
    for rank in record["ranks"]:
        model, loader, _dataset, _bs = build_head_and_loader(
            run_dir, config, args.evaluation_split, args.num_workers,
            args.max_eval_samples)
        model.load_state_dict(full_state, strict=True)
        if not hasattr(model.weak_period_residual, "linear"):
            raise RuntimeError(
                f"full-rank cell {record['dataset']}-{record['horizon']} has a "
                f"{type(model.weak_period_residual).__name__} head without a "
                "single H x 720 weight; refusing to truncate"
            )
        with torch.no_grad():
            model.weak_period_residual.linear.weight.copy_(
                svd_truncate(full_weight, rank))
        truncated = evaluate_model(model, loader, record["horizon"], device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        row = {
            "setting": f"{record['dataset']}-{record['horizon']}",
            "dataset": record["dataset"],
            "horizon": record["horizon"],
            "seed": record["seed"],
            "split": args.evaluation_split,
            "rank": rank,
            "svd_truncated_mse": truncated["mse"],
            "svd_truncated_mae": truncated["mae"],
            "trained_lowrank_arm": (comparison["arm"] if comparison else ""),
            "trained_lowrank_rank": (comparison["rank"] if comparison else ""),
            "trained_lowrank_mse": (comparison["val_mse"] if comparison else ""),
            "trained_lowrank_mae": (comparison["val_mae"] if comparison else ""),
            "full_rank_mse": full_rank_metrics["mse"],
            "full_rank_mae": full_rank_metrics["mae"],
            "gap_truncated_vs_trained_mse_pct": relative_gap(
                truncated["mse"], comparison["val_mse"] if comparison else None),
            "gap_truncated_vs_trained_mae_pct": relative_gap(
                truncated["mae"], comparison["val_mae"] if comparison else None),
            "gap_truncated_vs_full_mse_pct": relative_gap(
                truncated["mse"], full_rank_metrics["mse"]),
            "gap_trained_vs_full_mse_pct": relative_gap(
                comparison["val_mse"] if comparison else None,
                full_rank_metrics["mse"]),
            "eval_samples": truncated["count"],
            "batch_size": batch_size,
            "full_rank_run_dir": record["full_rank_run_dir"],
            "full_rank_cell_status": record["full_rank_cell_status"],
            "full_rank_source": record["full_rank_source"],
            "full_rank_gate_init": record["full_rank_gate_init"],
            "full_rank_learning_rate": record["full_rank_learning_rate"],
            "trained_lowrank_run_dir": (comparison["run_dir"] if comparison else ""),
            "trained_lowrank_cell_status": (comparison["cell_status"] if comparison else ""),
            "trained_lowrank_gate_init": (comparison["gate_init"] if comparison else ""),
            "trained_lowrank_learning_rate": (comparison["learning_rate"] if comparison else ""),
            "records_test": False,
        }
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

PER_RANK_FIELDS = [
    "setting", "dataset", "horizon", "seed", "split", "rank",
    "svd_truncated_mse", "svd_truncated_mae",
    "trained_lowrank_arm", "trained_lowrank_rank",
    "trained_lowrank_mse", "trained_lowrank_mae",
    "full_rank_mse", "full_rank_mae",
    "gap_truncated_vs_trained_mse_pct", "gap_truncated_vs_trained_mae_pct",
    "gap_truncated_vs_full_mse_pct", "gap_trained_vs_full_mse_pct",
    "eval_samples", "batch_size",
    "full_rank_run_dir", "full_rank_cell_status", "full_rank_source",
    "full_rank_gate_init", "full_rank_learning_rate",
    "trained_lowrank_run_dir", "trained_lowrank_cell_status",
    "trained_lowrank_gate_init", "trained_lowrank_learning_rate",
    "records_test",
]

TABLE_FIELDS = [
    "setting", "dataset", "horizon",
    "primary_rank", "rank_anchor", "seed", "split",
    "truncated_mse", "truncated_mae",
    "trained_lowrank_mse", "trained_lowrank_mae",
    "gap_truncated_vs_trained_mse_pct", "gap_truncated_vs_trained_mae_pct",
    "full_rank_mse", "full_rank_mae",
    "gap_trained_vs_full_mse_pct",
    "trained_lowrank_arm", "trained_lowrank_rank",
    "native_rank_q1_4", "native_rank_q1_8",
    "records_test",
    "full_rank_run_dir", "full_rank_cell_status",
    "trained_lowrank_run_dir", "trained_lowrank_cell_status",
    "full_rank_gate_init", "full_rank_learning_rate",
    "seeds_evaluated",
]


def write_csv(path: Path, rows: list, fields: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_seed_rows(per_rank_rows: list, primary_rank: int) -> list:
    """One row per (setting, seed): all three numbers at the same seed.

    The per-rank CSV already holds one row per (setting, seed, rank) and each of
    those rows carries the *same-seed* trained-low-rank cell it is compared
    with, so restricting to the anchor rank yields the (setting, seed)
    comparison directly.
    """
    return [row for row in per_rank_rows
            if same_int(row["rank"], primary_rank)]


def build_table(seed_rows: list, plan: list, seeds: list,
                primary_rank: int) -> list:
    """28 rows: one per setting, averaged over the seeds actually evaluated."""
    native: dict = {}
    for record in plan:
        key = f"{record['dataset']}-{record['horizon']}"
        bucket = native.setdefault(key, {})
        for item in record["low_rank"]:
            if isinstance(item.get("rank"), int):
                bucket.setdefault(item["arm"], item["rank"])

    by_setting: dict = {}
    for row in seed_rows:
        by_setting.setdefault(row["setting"], []).append(row)

    table = []
    for setting in sorted(by_setting,
                          key=lambda name: (name.rsplit("-", 1)[0],
                                            int(name.rsplit("-", 1)[1]))):
        rows = sorted(by_setting[setting], key=lambda row: row["seed"])
        first = rows[0]
        table.append({
            "setting": setting,
            "dataset": first["dataset"],
            "horizon": first["horizon"],
            "primary_rank": primary_rank,
            "rank_anchor": f"E11 anchor r={primary_rank}",
            "seed": ",".join(str(row["seed"]) for row in rows),
            "split": first["split"],
            "truncated_mse": fmt(mean([row["svd_truncated_mse"] for row in rows])),
            "truncated_mae": fmt(mean([row["svd_truncated_mae"] for row in rows])),
            "trained_lowrank_mse": fmt(mean(
                [row["trained_lowrank_mse"] for row in rows])),
            "trained_lowrank_mae": fmt(mean(
                [row["trained_lowrank_mae"] for row in rows])),
            "gap_truncated_vs_trained_mse_pct": fmt(mean(
                [row["gap_truncated_vs_trained_mse_pct"] for row in rows])),
            "gap_truncated_vs_trained_mae_pct": fmt(mean(
                [row["gap_truncated_vs_trained_mae_pct"] for row in rows])),
            "full_rank_mse": fmt(mean([row["full_rank_mse"] for row in rows])),
            "full_rank_mae": fmt(mean([row["full_rank_mae"] for row in rows])),
            "gap_trained_vs_full_mse_pct": fmt(mean(
                [row["gap_trained_vs_full_mse_pct"] for row in rows])),
            "trained_lowrank_arm": first["trained_lowrank_arm"],
            "trained_lowrank_rank": first["trained_lowrank_rank"],
            "native_rank_q1_4": native.get(setting, {}).get("l_q1_4", ""),
            "native_rank_q1_8": native.get(setting, {}).get("l_q1_8", ""),
            "records_test": False,
            "full_rank_run_dir": first["full_rank_run_dir"],
            "full_rank_cell_status": first["full_rank_cell_status"],
            "trained_lowrank_run_dir": first["trained_lowrank_run_dir"],
            "trained_lowrank_cell_status": first["trained_lowrank_cell_status"],
            "full_rank_gate_init": first["full_rank_gate_init"],
            "full_rank_learning_rate": first["full_rank_learning_rate"],
            "seeds_evaluated": len(rows),
        })
    return table


def same_int(left, right) -> bool:
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return False


def mean(values: list) -> float:
    clean = [float(value) for value in values
             if value not in ("", None) and math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else float("nan")


def fmt(value) -> object:
    """Float for a real number, empty string for a missing/NaN one."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if math.isfinite(number) else ""


def plot_table(table: list, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = [row["setting"] for row in table]
    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    positions = range(len(settings))
    axes[0].bar([p - 0.2 for p in positions],
                [float(row["gap_truncated_vs_trained_mse_pct"] or 0.0)
                 for row in table],
                width=0.4, label="truncated vs trained low-rank")
    axes[0].bar([p + 0.2 for p in positions],
                [float(row["gap_trained_vs_full_mse_pct"] or 0.0)
                 for row in table],
                width=0.4, label="trained low-rank vs full-rank")
    axes[0].set_ylabel("relative MSE gap (%)")
    axes[0].axhline(0.0, color="grey", linewidth=0.8)
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"SVD truncation vs rank-constrained training at r="
                      f"{table[0]['primary_rank'] if table else ''} "
                      f"(split={table[0]['split'] if table else ''})",
                      fontsize=10)
    axes[1].plot(positions,
                 [float(row["truncated_mse"] or float("nan"))
                  for row in table], marker="o", label="truncated")
    axes[1].plot(positions,
                 [float(row["trained_lowrank_mse"] or float("nan"))
                  for row in table], marker="s", label="trained low-rank")
    axes[1].plot(positions,
                 [float(row["full_rank_mse"] or float("nan"))
                  for row in table], marker="^",
                 linestyle="--", label="full-rank")
    axes[1].set_xticks(list(positions))
    axes[1].set_xticklabels(settings, rotation=90, fontsize=7)
    axes[1].set_ylabel("MSE")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, linewidth=0.3, alpha=0.5)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(json.dumps({"event": "figure", "path": str(path)}))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.evaluation_split not in ALLOWED_SPLITS:
        # argparse already restricts this; kept as an explicit second gate.
        raise SystemExit("only the validation split may be evaluated here")

    plan, problems, manifest_path, seeds, ranks = build_plan(args)
    primary_rank = min(ranks)
    if primary_rank < 1:
        raise SystemExit(f"--ranks must all be >= 1, got {ranks}")
    expected_settings = {f"{d}-{h}" for d, h in SETTINGS
                         if d in parse_list(args.datasets)
                         and h in parse_list(args.horizons, int)}
    planned_settings = {f"{record['dataset']}-{record['horizon']}"
                        for record in plan}
    report = {
        "experiment": "E18 row 3 (minipaper section 4.6) SVD truncation on 28 settings",
        "schedule_reference": "docs/PhaseFormer_L_execution_schedule.md section 2.3",
        "e14_manifest": repo_relative(manifest_path),
        "evaluation_split": args.evaluation_split,
        "test_split_read": False,
        "records_test": False,
        "truncation_ranks": ranks,
        "primary_rank": primary_rank,
        "primary_rank_rationale": (
            "r=10 is the rank at which E11 measured the Electricity-336 "
            "counterexample (truncated +29% vs trained +0.7%); keeping it fixed "
            "across all 28 settings keeps that one quoted quantity comparable"
        ),
        "seeds": seeds,
        "expected_settings": sorted(expected_settings),
        "planned_settings": sorted(planned_settings),
        "missing_settings": sorted(expected_settings - planned_settings),
        "problems": problems,
        "cells": plan,
        "e11_comparability": {
            "e11_script": "scripts/analyze_weak_residual_svd_truncation_eval.py",
            "e11_source": "docs/PhaseFormer_lowrank_mechanism_analysis.md section 3.2",
            "e11_settings": 7,
            "e11_split": "test",
            "e11_seed": 2021,
            "this_script_split": args.evaluation_split,
            "difference": (
                "E11 evaluated the real test set; this analysis is "
                "validation-only, so the two sets of numbers are not "
                "interchangeable and the section 4.6 table note must disclose it"
            ),
            "shared_definition": (
                "same truncated-vs-trained-vs-full-rank comparison and the same "
                "SVD truncation algebra; the checkpoint source differs "
                "(E14 l_main cells under the section 4.0 protocol instead of "
                "research_runs/rank_sweep_2_stage1)"
            ),
            "absolute_rank_note": ABSOLUTE_RANK_OVERRIDE_NOTE,
        },
        "protocol_mirrored_from_e14": {
            "lookback": LOOKBACK, "period": PERIOD, "loss": LOSS,
            "max_epochs": MAX_EPOCHS, "percent": PERCENT,
            "full_rank_arm": FULL_RANK_ARM, "low_rank_arms": list(LOW_RANK_ARMS),
            "arm_table_source": "scripts/phaseformer_L/e14_main_matrix.py::ARMS",
        },
    }
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "e18_svd_truncation_plan.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "planned",
                      "settings": len(planned_settings),
                      "seeds": len(seeds),
                      "ranks": ranks,
                      "problems": len(problems)}, ensure_ascii=False))

    if args.verify and problems:
        for problem in problems[:50]:
            print(json.dumps({"event": "problem", **problem}), flush=True)
        raise SystemExit(
            f"verify failed: {len(problems)} unresolved cells; refusing to evaluate")
    if args.verify and not problems:
        print(json.dumps({"event": "verify_ok",
                          "settings": len(planned_settings),
                          "seeds": len(seeds),
                          "cells": len(plan)}), flush=True)
    if args.dry_run:
        for record in plan:
            print(json.dumps({
                "setting": f"{record['dataset']}-{record['horizon']}",
                "seed": record["seed"],
                "full_rank": record["full_rank_run_dir"],
                "full_rank_status": record["full_rank_cell_status"],
                "low_rank": [(item["arm"], item["rank"]) for item in record["low_rank"]],
                "problems": record["problems"],
            }, sort_keys=True), flush=True)
        print(json.dumps({"event": "plan_only", "cells": len(plan)}))
        return
    if problems:
        print(json.dumps({"event": "warning", "unresolved": len(problems),
                          "note": "unresolved cells are skipped"}), flush=True)

    device = pick_device(args.device)
    per_rank_rows: list = []
    evaluated = []
    for record in plan:
        if record["problems"]:
            continue
        per_rank_rows.extend(evaluate_setting(record, args, device))
        evaluated.append(f"{record['dataset']}-{record['horizon']}-s{record['seed']}")
        print(json.dumps({"event": "evaluated", "setting":
                          f"{record['dataset']}-{record['horizon']}",
                          "seed": record["seed"]}), flush=True)

    per_rank_path = output_root / "svd_truncation_per_rank.csv"
    write_csv(per_rank_path, per_rank_rows, PER_RANK_FIELDS)
    seed_rows = build_seed_rows(per_rank_rows, primary_rank)
    table = build_table(seed_rows, plan, seeds, primary_rank)
    table_path = output_root / "svd_truncation_table_28.csv"
    write_csv(table_path, table, TABLE_FIELDS)

    summary = {
        **{key: report[key] for key in
           ("experiment", "e14_manifest", "evaluation_split", "test_split_read",
            "records_test", "truncation_ranks", "primary_rank",
            "primary_rank_rationale", "seeds", "e11_comparability",
            "protocol_mirrored_from_e14")},
        "settings_planned": len(planned_settings),
        "settings_evaluated": len({row["setting"] for row in table}),
        "cells_evaluated": len(evaluated),
        "unresolved": problems,
        "table_rows": len(table),
        "per_rank_rows": len(per_rank_rows),
        "outputs": {
            "plan": repo_relative(output_root / "e18_svd_truncation_plan.json"),
            "per_rank_csv": repo_relative(per_rank_path),
            "table_csv": repo_relative(table_path),
        },
        "worst_truncation_gaps": sorted(
            ({"setting": row["setting"],
              "gap_truncated_vs_trained_mse_pct":
                  row["gap_truncated_vs_trained_mse_pct"]}
             for row in table),
            key=lambda item: -(float(item["gap_truncated_vs_trained_mse_pct"])
                               if item["gap_truncated_vs_trained_mse_pct"] != ""
                               else float("-inf")),
        )[:5],
    }
    if not args.skip_figures and table:
        figure_path = output_root / "figures" / "svd_truncation_28.svg"
        plot_table(table, figure_path)
        summary["outputs"]["figure"] = repo_relative(figure_path)
    (output_root / "e18_svd_truncation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "finished",
                      "settings": summary["settings_evaluated"],
                      "table_rows": summary["table_rows"],
                      "table_csv": summary["outputs"]["table_csv"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
