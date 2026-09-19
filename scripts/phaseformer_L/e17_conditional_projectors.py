#!/usr/bin/env python3
"""E17 / minipaper section 4.5: build the two frozen direction-1 projectors.

``docs/PhaseFormer_L/e17_conditional/01_plan.md`` sections 4-5.  For each of the
seven test-selected settings of ``docs/PhaseFormer_L_execution_schedule.md``
section 2.3 this script produces, **on the train split only**,

* ``<dataset>_<horizon>_Q1.npy``     -- frozen **independent**-RRR direction 1
  (target ``D_ind = y - x_last``), the object the E8 arm already froze; and
* ``<dataset>_<horizon>_Q1COND.npy`` -- frozen **conditional**-RRR direction 1
  (target ``D_cond = y - y_phi``), the genuinely new object of E17.

File convention (identical to the E8 ``Q1.npy`` artifacts, verified against
``research_runs/top2_direction_retention_v1/projectors/ETTh2_96_Q1.npy``):
float64, shape ``(720, 1)``, C-order, orthonormal, consumed by
``scripts/run_top2_direction_retention.py --basis``.

Two routes are implemented and deliberately kept apart:

``standardized``
    The E8 recipe, restated on the repository's own train-split standardization
    (train mean / population std, ddof=0): ``Z = x_window - x_last``,
    ``D = y - x_last``, ``S = Szy^T (Szz + ridge I)^-1 Szy`` (``eigh``
    descending), ``b_i = u_i^T Szy^T (Szz + ridge I)^-1``, Gram-Schmidt to
    ``Q``.  This route has a **correctness gate**: the six settings whose
    projector E8 already shipped are recomputed here and compared with the
    stored files by ``|cos|`` (``--min-reproduction-cos``, default 0.999).
    Electricity-336 has no stored projector, so its ``Q1`` is new but comes out
    of the same code path that just reproduced the other six.

``revin``
    The branch's real private input ``z = (x - x_last) / sigma`` read off a
    trained E14 ``l_main`` checkpoint on the **train loader**, from which both
    targets are available in the RevIN-normalized space that
    ``src/models/PhaseFormer.py`` itself works in::

        mu, sigma = revin.normalize(x)          # stats of the forward pass
        D_cond    = target_norm - phase_norm    # == (y - y_phi) / sigma
        D_ind     = target_norm - anchor_norm   # == (y - x_last) / sigma
        z         = centered normalized history

    The conditional projector is built here.  Its closed-form sibling
    ``scripts/compute_phase_conditional_rrr.py`` targets the *gated* residual
    ``target_norm - (1-g) phase_norm - g anchor_norm`` with weights ``g^2``,
    which is a different estimator; E17 follows the section 4.5 definition
    ``D_cond = y - y_phi`` exactly and records the difference in its audit.

Why train-only: the forward pass iterates the **train loader** only, the
standardized route reads the CSV through ``e15_dimension.load_split`` (which
parses ``nrows = validation border``, so test rows are never loaded), no
gradient is taken, and no checkpoint is modified.  ``--dry-run`` resolves and
prints the whole plan without importing numpy or torch.

Usage::

    # stage 0 (no imports of numpy/torch): show the plan, write nothing
    python scripts/phaseformer_L/e17_conditional_projectors.py --dry-run

    # the real build (CPU is enough; the train forward pass is the slow part)
    python scripts/phaseformer_L/e17_conditional_projectors.py \
        --output-dir research_runs/phaseformer_L_e17_conditional_v1/projectors \
        --e14-root   research_runs/phaseformer_L_e14_main_v1 \
        --reference-dir research_runs/top2_direction_retention_v1/projectors
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Protocol constants (frozen; see docs/PhaseFormer_L/e17_conditional/01_plan.md)
# ---------------------------------------------------------------------------
LOOKBACK = 720
PERIOD = 24
SEEDS = (2021, 2022, 2023)
#: Ridge of the E8 Stage-0 fit (scripts/compute_top2_direction_projectors.py).
RIDGE = 1e-6
#: RevIN epsilon of src/models/phase_adapters.py::RevIN (mirrored by E16).
REVIN_EPS = 1e-5
#: Default ``|cos|`` a recomputed E8 projector must reach.  The two code paths
#: compute the same eigendecomposition, so the expected agreement is ~1 - 1e-12;
#: the threshold is loose enough to absorb BLAS/LAPACK differences only.
MIN_REPRODUCTION_COS = 0.999
#: Row block for the Gram accumulation of the standardized route.  The full
#: design of Electricity-336 is ~5.6e6 x 720 float64; blocking keeps the
#: transients small while leaving the (associative) sums unchanged.
ACCUMULATION_BLOCK = 65_536

SETTINGS = (
    ("ETTh2", 96),
    ("ETTh2", 720),
    ("ETTm2", 96),
    ("ETTm2", 192),
    ("Weather", 96),
    ("Weather", 192),
    ("Electricity", 336),
)
#: The six settings whose E8 projector exists and therefore has a reproduction gate.
E8_SETTINGS = tuple(s for s in SETTINGS if s != ("Electricity", 336))

DATASET_KIND = {
    "ETTh2": "ett_hour",
    "ETTm2": "ett_minute",
    "Weather": "custom",
    "Electricity": "custom",
}

# DATA_TYPE of src/dataset/data_info.py -> split family of e15_dimension.py
SPLIT_KIND_BY_DATA = {
    "ett_h": "ett_hour",
    "ett_m": "ett_minute",
    "custom": "custom",
}

ARM_INDEPENDENT = "e17_frozen_independent_direction_1"
ARM_CONDITIONAL = "e17_frozen_conditional_direction_1"


def parse_list(raw, cast=str) -> list:
    return [cast(item) for item in str(raw).split(",") if str(item).strip()]


def parse_settings(raw: str) -> list[tuple[str, int]]:
    """``"ETTh2-96,ETTm2:192"`` -> ``[("ETTh2", 96), ("ETTm2", 192)]``.

    ``rsplit`` is used so a dataset name containing a dash cannot be split in the
    wrong place.
    """
    out = []
    for chunk in str(raw).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        separator = ":" if ":" in chunk else "-"
        dataset, horizon = chunk.rsplit(separator, 1)
        out.append((dataset, int(horizon)))
    return out


def setting_name(dataset: str, horizon: int) -> str:
    return f"{dataset}-{int(horizon)}"


def display_path(path: Path) -> str:
    """Repository-relative path when possible, absolute otherwise."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# CLI (built without importing numpy/torch so --help and --dry-run work here)
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="e17_conditional_projectors.py",
        description=(
            "E17 stage 0: build the frozen independent-RRR and conditional-RRR "
            "direction-1 projectors for the seven test-selected settings, on the "
            "train split only.  Never reads the validation or test split statistics "
            "of the standardized route and never reads the test split at all."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="research_runs/phaseformer_L_e17_conditional_v1/projectors",
        help="where the *_Q1.npy and *_Q1COND.npy artifacts and the audit are written",
    )
    parser.add_argument(
        "--reference-dir",
        default="research_runs/top2_direction_retention_v1/projectors",
        help="E8 projector directory holding the shipped <ds>_<H>_Q1.npy used as "
        "the reproduction gate; a missing file is reported, not fabricated",
    )
    parser.add_argument(
        "--settings",
        default=",".join(setting_name(d, h) for d, h in SETTINGS),
        help="comma list of <dataset>-<horizon>; default is the seven "
        "test-selected settings of the schedule",
    )
    parser.add_argument("--seq-len", type=int, default=LOOKBACK)
    parser.add_argument("--ridge", type=float, default=RIDGE)
    parser.add_argument("--chunk", type=int, default=4096,
                        help="window block of the standardized route")
    parser.add_argument("--accumulation-block", type=int, default=ACCUMULATION_BLOCK)
    parser.add_argument(
        "--min-reproduction-cos", type=float, default=MIN_REPRODUCTION_COS,
        help="|cos| a recomputed E8 Q1 must reach to PASS the reproduction gate",
    )
    parser.add_argument(
        "--data-root", default="",
        help="override for the leading 'all_datasets/' part of the registered "
        "dataset root; empty uses DATASET_INFO (the server layout)",
    )
    # conditional (revin) route
    parser.add_argument(
        "--routes", default="standardized,revin",
        help="comma list of 'standardized' (E8 recipe; the independent arm's "
        "projector) and 'revin' (model route; the conditional arm's projector)",
    )
    parser.add_argument(
        "--e14-root", default="research_runs/phaseformer_L_e14_main_v1",
        help="E14 stage-A root; its runs/ supply the l_main checkpoint whose phase "
        "backbone defines D_cond",
    )
    parser.add_argument(
        "--phase-seed", type=int, default=2021,
        help="seed of the E14 l_main checkpoint used for the phase forecast; the "
        "conditional projector is one frozen object per setting, so the choice is "
        "fixed and recorded (see 01_plan.md section 4.2)",
    )
    parser.add_argument("--batch-size", type=int, default=0,
                        help="0 = the checkpoint's own recorded batch size")
    parser.add_argument("--gpus", default="", help="e.g. '0'; empty = CPU")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="cap the train batches (smoke knob; 0 = all)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve the plan, print it, import neither numpy nor "
                        "torch, write nothing")
    return parser


# ---------------------------------------------------------------------------
# Dataset registry / split borders (repository conventions, not re-derived)
# ---------------------------------------------------------------------------
def load_dataset_info() -> dict:
    """Read ``DATASET_INFO`` from ``src/dataset/data_info.py`` without torch.

    Same device as ``scripts/phaseformer_L/e15_dimension.py::load_dataset_info``:
    importing ``src.dataset`` pulls torch in through its ``__init__``, and this
    script must stay import-light for ``--dry-run``.
    """
    path = ROOT / "src" / "dataset" / "data_info.py"
    if not path.is_file():
        raise SystemExit(f"cannot find the dataset registry: {path}")
    spec = importlib.util.spec_from_file_location("_e17_data_info", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load the dataset registry: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    info = getattr(module, "DATASET_INFO", None)
    if not isinstance(info, dict) or not info:
        raise SystemExit(f"DATASET_INFO missing or empty in {path}")
    return info


def build_registry(dataset_info: dict) -> dict:
    registry = {}
    for name, info in dataset_info.items():
        kind = SPLIT_KIND_BY_DATA.get(str(info.get("data", "")))
        if kind is None:
            continue
        registry[name] = {
            "kind": kind,
            "root_path": str(info.get("root_path", "")),
            "data_path": str(info.get("data_path", "")),
            "channels": info.get("num_variants"),
        }
    return registry


def resolve_csv_path(dataset: str, registry: dict, data_root: str) -> Path:
    entry = registry[dataset]
    root_path = entry["root_path"].replace("\\", "/")
    if data_root:
        marker = "all_datasets/"
        suffix = (
            root_path.split(marker, 1)[1] if marker in root_path
            else Path(root_path).name
        )
        return (Path(data_root) / suffix / entry["data_path"]).resolve()
    return Path(root_path) / entry["data_path"]


# ---------------------------------------------------------------------------
# Shared linear algebra (E8 numerics restated; no new estimator)
# ---------------------------------------------------------------------------
def rrr_directions(np, szz, szy, ridge):
    """Top RRR eigenvalues plus the input-side directions ``b_i``.

    Literal restatement of ``scripts/compute_top2_direction_projectors.py``
    (the script that produced the shipped E8 ``Q1``/``Q12`` files): eigendecompose
    the symmetrized ``S = Szy^T (Szz + ridge I)^-1 Szy``, sort descending, clip
    negatives to zero, and map the leading eigenvectors back to the input side
    with ``b_i = u_i^T Szy^T (Szz + ridge I)^-1``.
    """
    szz_r = szz + ridge * np.eye(szz.shape[0])
    szz_inv_szy = np.linalg.solve(szz_r, szy)          # L x H
    ols = szz_inv_szy.T                                # H x L
    s_mat = 0.5 * (szy.T @ szz_inv_szy + (szy.T @ szz_inv_szy).T)
    values, vectors = np.linalg.eigh(s_mat)
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 0.0, None)
    vectors = vectors[:, order]
    directions = vectors.T @ ols                       # H x L
    return values, directions


def orthonormalize(np, vectors):
    """Gram-Schmidt orthonormal basis of ``(k, L)`` rows, with pivots."""
    basis = []
    pivots = []
    for vector in vectors:
        residual = np.array(vector, dtype=np.float64, copy=True)
        for existing in basis:
            residual -= (residual @ existing) * existing
        norm = float(np.linalg.norm(residual))
        pivots.append(norm)
        if norm <= 1e-10:
            raise SystemExit(
                "RRR directions are numerically degenerate; refusing to emit a "
                "non-orthonormal projector"
            )
        basis.append(residual / norm)
    return np.stack(basis, axis=1), np.asarray(pivots)


def orthonormality_error(np, basis) -> float:
    k = basis.shape[1]
    return float(np.abs(basis.T @ basis - np.eye(k)).max())


def projector_idempotence_error(np, basis) -> float:
    projector = basis @ basis.T
    return float(np.abs(projector @ projector - projector).max())


def cosine_abs(np, a, b) -> float:
    """``|cos|`` between two direction vectors (orientation is arbitrary)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(abs(a @ b / denom))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Route 1: standardized (the E8 recipe)
# ---------------------------------------------------------------------------
def accumulate_standardized(np, sliding_window_view, train_seg, seq_len,
                            horizon, chunk):
    """``Szz``/``Szy`` of the E8 route, pooled over windows and channels."""
    total = len(train_seg) - seq_len - horizon + 1
    if total <= 0:
        raise SystemExit(
            f"the train split has {len(train_seg)} rows but a window needs "
            f"{seq_len + horizon}"
        )
    x_all = sliding_window_view(train_seg, seq_len, axis=0)[:total]
    y_all = sliding_window_view(train_seg[seq_len:], horizon, axis=0)[:total]
    s_zz = np.zeros((seq_len, seq_len))
    s_zy = np.zeros((seq_len, horizon))
    s_yy = np.zeros((horizon, horizon))
    count = 0
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        x = np.asarray(x_all[start:stop], dtype=np.float64)
        y = np.asarray(y_all[start:stop], dtype=np.float64)
        last = x[:, :, -1:]
        z = (x - last).reshape(-1, seq_len)
        d = (y - last).reshape(-1, horizon)
        s_zz += z.T @ z
        s_zy += z.T @ d
        s_yy += d.T @ d
        count += int(z.shape[0])
    n = max(count, 1)
    return {
        "szz": s_zz / n,
        "szy": s_zy / n,
        "syy": s_yy / n,
        "n_pairs": count,
        "windows_per_channel": int(total),
    }


def build_standardized(np, registry, args, dataset, horizon):
    """The independent-RRR direction 1 of the E8 recipe (train split only)."""
    from scripts.phaseformer_L.e15_dimension import load_split

    from numpy.lib.stride_tricks import sliding_window_view

    csv_path = resolve_csv_path(dataset, registry, args.data_root)
    train_seg, _val_seg, meta = load_split(
        csv_path, DATASET_KIND[dataset], int(args.seq_len)
    )
    moments = accumulate_standardized(
        np, sliding_window_view, train_seg, int(args.seq_len), horizon,
        int(args.chunk),
    )
    values, directions = rrr_directions(
        np, moments["szz"], moments["szy"], float(args.ridge)
    )
    total_lambda = float(values.sum()) + 1e-12
    q1, pivots = orthonormalize(np, directions[:1])
    return {
        "basis": q1,
        "pivots": [float(v) for v in pivots],
        "lambda_shares": [float(v) / total_lambda for v in values[:4]],
        "lambda2_lambda3_gap": (
            float((values[1] - values[2]) / values[1]) if values[1] > 0 else 0.0
        ),
        "n_pairs": int(moments["n_pairs"]),
        "windows_per_channel": int(moments["windows_per_channel"]),
        "csv_path": str(csv_path),
        "csv_rows_parsed": int(meta["rows_read"]),
        "test_border_row": int(meta["test_border_row"]),
        "channels": int(train_seg.shape[1]),
    }


# ---------------------------------------------------------------------------
# Route 2: revin (the model route; the conditional projector)
# ---------------------------------------------------------------------------
def resolve_e14_run_dir(e14_root: Path, dataset: str, horizon: int, seed: int):
    """Locate the E14 ``l_main`` run directory of one cell.

    ``e14_main_matrix._arm_match`` is imported rather than reimplemented, so
    "which run implements ``l_main``" stays defined in exactly one place.  A run
    carrying ``weak_residual_projection`` is a frozen-subspace arm and is never
    an E14 ``l_main`` cell (E14 already excludes those), which is what keeps the
    E17 arms trained later from being mistaken for the phase backbone's source.

    **Reused cells do not live under ``e14_root``.**  E14 deliberately does not
    copy the audited E3-lineage artifacts: it *reuses* them in place, so
    ``stage_a_manifest.json`` records ``status="reused"`` with
    ``source.run_dir`` pointing into the E3 root.  All seven settings E17 needs
    are reused (they are the test-selected seven), so the manifest must be
    consulted before globbing, otherwise every cell reports "no run" even though
    the checkpoint the conditional target needs is present and audited.
    """
    from scripts.phaseformer_L import e14_main_matrix as e14

    candidates: list[tuple[Path, str]] = []
    manifest_path = e14_root / "stage_a_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = {}
        for cell in manifest.get("cells", []):
            if (
                cell.get("arm") != "l_main"
                or str(cell.get("dataset")) != dataset
                or int(cell.get("horizon", -1)) != int(horizon)
                or int(cell.get("seed", -1)) != int(seed)
            ):
                continue
            source = cell.get("source") or {}
            run_dir = str(source.get("run_dir", "")).strip()
            if run_dir:
                candidates.append((ROOT / run_dir, "manifest_reuse"))

    matches = []
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
                and e14._arm_match(config, "l_main")
            ):
                candidates.append((config_path.parent, "e14_run"))

    # Keep only directories that actually exist and implement l_main.
    matches = []
    for run_dir, origin in candidates:
        config_path = run_dir / "config.json"
        if not config_path.is_file() or not run_dir.is_dir():
            continue
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            continue
        if e14._arm_match(config, "l_main"):
            matches.append((origin, run_dir))

    def sort_key(item):
        origin, run_dir = item
        has_metrics = (run_dir / "metrics.csv").is_file()
        # A reused manifest cell is the audited artifact E14 actually reports,
        # so it wins over an incidental duplicate found by globbing.
        return (
            0 if origin == "manifest_reuse" else 1,
            0 if has_metrics else 1,
            str(run_dir),
        )

    matches.sort(key=sort_key)
    return (
        matches[0][1] if matches else None,
        [str(path.relative_to(ROOT)) for _, path in matches],
    )


def checkpoint_in_run_dir(run_dir: Path):
    """Best-validation checkpoint of a run (same rule as E16)."""
    import csv as csv_module

    metrics_path = run_dir / "metrics.csv"
    recorded = ""
    if metrics_path.is_file():
        with metrics_path.open(newline="") as handle:
            first = next(csv_module.DictReader(handle), None) or {}
        recorded = str(first.get("checkpoint", "")).strip()
    if recorded:
        raw = Path(recorded)
        candidates = [ROOT / raw, run_dir / raw]
        parts = raw.parts
        for index, part in enumerate(parts):
            if part == "runs" and index + 1 < len(parts):
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


def build_revin(np, torch, registry, args, dataset, horizon):
    """``Q_cond`` (used) and ``Q_ind`` (diagnostic) from the trained phase backbone.

    One ``inference_mode`` pass over the **train loader** of the E14 ``l_main``
    checkpoint.  The head is patched with E16's ``dense_head_forward`` and the
    model forward with ``lowrank_checkpoint_model._capture_forward`` so the
    recorded quantities (RevIN stats, phase-only forecast, gate, centered input)
    are exactly the ones the registered evaluator caches.

    Both targets live in the same RevIN-normalized space as the branch's own
    input, which is why ``D_cond = y - y_phi`` can be fitted here without any
    extra weighting.
    """
    from scripts.lowrank_checkpoint_model import (
        _capture_forward,
        build_loaders,
        build_model,
        set_seed,
    )
    from scripts.phaseformer_L.e16_dissection import (
        dense_head_forward,
        instrument_branch,
    )
    from src.models.phase_adapters import PooledLowRankWeakPeriodResidualHead

    if torch is None:
        raise SystemExit(
            "the revin route needs torch; it is imported lazily so --dry-run can "
            "run without it"
        )

    run_dir, alternatives = resolve_e14_run_dir(
        ROOT / args.e14_root, dataset, horizon, int(args.phase_seed)
    )
    if run_dir is None:
        raise SystemExit(
            f"no E14 l_main run directory for {dataset}-{horizon} seed "
            f"{args.phase_seed} under {args.e14_root}; the conditional target "
            "needs a trained phase backbone (schedule section 4.1: E17 depends on "
            "E14)"
        )
    config = json.loads((run_dir / "config.json").read_text())
    hyperparams = dict(config["hyperparams"])
    batch_size = int(
        args.batch_size
        or config.get("batch_size")
        or hyperparams.get("batch_size")
        or 256
    )
    exp_args, handles = build_loaders(
        dataset, LOOKBACK, horizon, hyperparams, batch_size, ROOT,
        splits=("train",),
    )
    train_loader = handles["train"][1]

    set_seed(int(args.phase_seed))
    model = build_model(exp_args, LOOKBACK, horizon, hyperparams)
    checkpoint_path, rule = checkpoint_in_run_dir(run_dir)
    if checkpoint_path is None:
        raise SystemExit(f"no checkpoint found inside {run_dir}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=True)

    device = torch.device("cpu")
    if args.gpus:
        if not torch.cuda.is_available():
            raise SystemExit("--gpus was given but CUDA is not available")
        device = torch.device("cuda", int(parse_list(args.gpus, int)[0]))
        torch.cuda.set_device(device)
    model.to(device).eval()

    head_kind = (
        "pooled_lowrank"
        if isinstance(model.weak_period_residual, PooledLowRankWeakPeriodResidualHead)
        else "dense"
    )
    if head_kind != "dense":
        raise SystemExit(
            "the phase-backbone checkpoint for D_cond must be the dense "
            "('shared') l_main head; got "
            f"{type(model.weak_period_residual).__name__}"
        )
    if getattr(model.weak_period_residual, "projection_basis", None) is not None:
        raise SystemExit(
            "the phase-backbone checkpoint carries a frozen projection basis: "
            "that is a frozen-subspace arm, not the joint l_main backbone"
        )

    szz = np.zeros((LOOKBACK, LOOKBACK))
    szy_ind = np.zeros((LOOKBACK, horizon))
    szy_cond = np.zeros((LOOKBACK, horizon))
    syy_cond = np.zeros((horizon, horizon))
    n_pairs = 0
    gate_sum = 0.0
    gate_count = 0
    gate_min = None
    gate_max = None
    batches = 0

    with torch.inference_mode():
        for batch_index, batch in enumerate(train_loader):
            if args.max_batches and batch_index >= int(args.max_batches):
                break
            batch = [item.to(device) if torch.is_tensor(item) else item for item in batch]
            x, y, x_mark, y_mark = batch
            dec = model._build_decoder_input(y.float())
            with instrument_branch(model, head_kind) as instrumented:
                instrumented(x.float(), x_mark.float(), dec, y_mark.float())
                records = instrumented.last_lowrank_records
            if records.get("phase_norm") is None or records.get("z") is None:
                raise SystemExit(
                    "the instrumented forward did not record the normalized phase "
                    "forecast / centered input; refusing to emit a conditional "
                    "projector from an unverified quantity"
                )
            mu, sigma = records["stats"]
            mu = mu.reshape(mu.shape[0], 1, -1)
            sigma = sigma.reshape(sigma.shape[0], 1, -1)
            centered = records["z"].double()                     # (B, C, L)
            phase = records["phase_norm"].double()[:, -horizon:, :]   # (B, H, C)
            anchor = centered[:, :, -1:].permute(0, 2, 1)        # (B, H, C) via broadcast
            anchor = anchor.expand(-1, horizon, -1)
            target = (y.double()[:, -horizon:, :] - mu) / sigma  # (B, H, C)
            cond = target - phase
            ind = target - anchor
            gate = records.get("gate")
            if gate is None:
                raise SystemExit("the instrumented forward did not record the gate")

            z_flat = centered.permute(0, 2, 1).reshape(-1, LOOKBACK)  # (B*C, L)
            ind_flat = ind.reshape(-1, horizon)
            cond_flat = cond.reshape(-1, horizon)
            rows = z_flat.shape[0]
            block = max(1, int(args.accumulation_block))
            for start in range(0, rows, block):
                stop = min(start + block, rows)
                z_block = np.ascontiguousarray(z_flat[start:stop])
                ind_block = np.ascontiguousarray(ind_flat[start:stop])
                cond_block = np.ascontiguousarray(cond_flat[start:stop])
                szz += z_block.T @ z_block
                szy_ind += z_block.T @ ind_block
                szy_cond += z_block.T @ cond_block
                syy_cond += cond_block.T @ cond_block
            n_pairs += int(rows)
            batches += 1
            gate_flat = gate.double().reshape(-1)
            gate_sum += float(gate_flat.sum())
            gate_count += int(gate_flat.numel())
            gate_min = float(gate_flat.min()) if gate_min is None else min(gate_min, float(gate_flat.min()))
            gate_max = float(gate_flat.max()) if gate_max is None else max(gate_max, float(gate_flat.max()))
            del x, y, x_mark, y_mark, dec, records, centered, phase, cond, ind
    if n_pairs == 0:
        raise SystemExit(
            f"{dataset}-{horizon}: the train pass produced no rows; refusing to "
            "emit a projector"
        )
    if not np.isfinite(szz).all() or not np.isfinite(szy_cond).all():
        raise SystemExit("non-finite second moments in the revin route")

    _vals_ind, directions_ind = rrr_directions(np, szz, szy_ind, float(args.ridge))
    _vals_cond, directions_cond = rrr_directions(np, szz, szy_cond, float(args.ridge))
    q_ind, pivots_ind = orthonormalize(np, directions_ind[:1])
    q_cond, pivots_cond = orthonormalize(np, directions_cond[:1])
    return {
        "conditional_basis": q_cond,
        "independent_basis": q_ind,
        "pivots_conditional": [float(v) for v in pivots_cond],
        "pivots_independent": [float(v) for v in pivots_ind],
        "n_pairs": int(n_pairs),
        "batches": int(batches),
        "gate_mean": gate_sum / max(gate_count, 1),
        "gate_min": gate_min,
        "gate_max": gate_max,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "alternative_run_dirs": alternatives[:5],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_rule": rule,
        "config_hash": config.get("config_hash"),
        "batch_size": batch_size,
        "residual_head": type(model.weak_period_residual).__name__,
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def write_projector(np, path: Path, basis, note: str) -> dict:
    np.save(path, np.ascontiguousarray(basis, dtype=np.float64))
    stored = np.load(path)
    try:
        displayed = str(path.relative_to(ROOT))
    except ValueError:
        displayed = str(path)
    return {
        "file": path.name,
        "path": displayed,
        "shape": [int(v) for v in stored.shape],
        "dtype": str(stored.dtype),
        "fortran_order": False,
        "sha256": sha256_of(path),
        "orthonormality_error": orthonormality_error(np, stored),
        "projector_idempotence_error": projector_idempotence_error(np, stored),
        "units": "orthonormal input-side direction(s) of Q Q^T",
        "consumer": "scripts/run_top2_direction_retention.py --basis <path>",
        "note": note,
    }


def main() -> None:
    args = build_parser().parse_args()
    requested = parse_settings(args.settings)
    routes = [item.strip() for item in str(args.routes).split(",") if item.strip()]
    unknown = [name for name in routes if name not in ("standardized", "revin")]
    if unknown:
        raise SystemExit(f"unknown --routes entries: {unknown}")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    reference_dir = Path(args.reference_dir)
    if not reference_dir.is_absolute():
        reference_dir = ROOT / reference_dir

    plan = {
        "protocol": "phaseformer-L-e17-conditional-projectors-v1",
        "lookback": int(args.seq_len),
        "period": PERIOD,
        "ridge": float(args.ridge),
        "routes": routes,
        "conditional_target": "D_cond = y - y_phi (RevIN-normalized space: "
                              "target_norm - phase_norm)",
        "independent_target": "D_ind = y - x_last",
        "conditional_target_route": "train_split_forward_pass",
        "conditional_phase_seed": int(args.phase_seed),
        "steps": [
            {
                "step": "standardized",
                "dataset": dataset,
                "horizon": horizon,
                "reproduction_reference": display_path(
                    reference_dir / f"{dataset}_{horizon}_Q1.npy"
                ),
                "reference_exists": (
                    reference_dir / f"{dataset}_{horizon}_Q1.npy"
                ).is_file(),
            }
            for dataset, horizon in requested
        ] if "standardized" in routes else [],
        "revin_steps": [
            {
                "step": "revin",
                "dataset": dataset,
                "horizon": horizon,
                "e14_root": args.e14_root,
                "phase_seed": int(args.phase_seed),
            }
            for dataset, horizon in requested
        ] if "revin" in routes else [],
        "writes": [
            "projectors.json",
            "projector_audit.json",
            "reproduction_gate.json",
        ],
        "imports_numpy_or_torch": False,
        "reads_test_split": False,
    }

    if args.dry_run:
        plan["dry_run"] = True
        print(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True))
        print(json.dumps({
            "event": "dry_run",
            "settings": len(requested),
            "artifact_files": 2 * len(requested),
            "note": "numpy/torch were not imported",
        }))
        return

    import numpy as np  # noqa: E402  (lazy: --dry-run must stay import-light)

    dataset_info = load_dataset_info()
    registry = build_registry(dataset_info)
    for dataset, _horizon in requested:
        if dataset not in registry:
            raise SystemExit(
                f"dataset {dataset!r} is not in the repository registry "
                f"(known: {sorted(registry)})"
            )
    torch = None
    if "revin" in routes:
        import torch  # noqa: E402  (lazy, and only when the model route is asked for)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit: list[dict] = []
    for dataset, horizon in requested:
        setting = setting_name(dataset, horizon)
        record: dict = {
            "setting": setting,
            "dataset": dataset,
            "horizon": int(horizon),
            "seq_len": int(args.seq_len),
            "ridge": float(args.ridge),
            "splits_read": ["train"],
            "notes": [
                "Standardized route: only the train split is standardized/read "
                "(e15_dimension.load_split parses nrows=validation border).",
                "Revin route: one inference_mode pass over the train loader of the "
                "E14 l_main checkpoint; the test split is never loaded.",
            ],
        }
        print(f"[e17-projectors] {setting}", flush=True)

        if "standardized" in routes:
            std = build_standardized(np, registry, args, dataset, horizon)
            basis = std.pop("basis")
            file_info = write_projector(
                np, output_dir / f"{dataset}_{horizon}_Q1.npy", basis,
                "frozen independent-RRR direction 1 (E8 recipe, train split)",
            )
            reference_path = reference_dir / f"{dataset}_{horizon}_Q1.npy"
            gate: dict = {
                "reference_path": str(reference_path),
                "reference_exists": reference_path.is_file(),
                "min_reproduction_cos": float(args.min_reproduction_cos),
            }
            if reference_path.is_file():
                stored = np.load(reference_path)
                gate["reference_shape"] = [int(v) for v in stored.shape]
                gate["reference_dtype"] = str(stored.dtype)
                gate["reference_sha256"] = sha256_of(reference_path)
                gate["abs_cos"] = cosine_abs(np, basis[:, 0], stored[:, 0])
                gate["reproduced"] = bool(
                    gate["abs_cos"] >= float(args.min_reproduction_cos)
                )
            else:
                gate["abs_cos"] = None
                gate["reproduced"] = None
                gate["reason"] = "no E8 projector shipped for this setting"
            record["standardized"] = {**std, "projector": file_info,
                                      "reproduction_gate": gate}
            print(
                json.dumps({
                    "event": "standardized",
                    "setting": setting,
                    "abs_cos": gate["abs_cos"],
                    "reproduced": gate["reproduced"],
                    "file": file_info["file"],
                }),
                flush=True,
            )

        if "revin" in routes:
            revin = build_revin(np, torch, registry, args, dataset, horizon)
            cond_basis = revin.pop("conditional_basis")
            ind_basis = revin.pop("independent_basis")
            cond_info = write_projector(
                np, output_dir / f"{dataset}_{horizon}_Q1COND.npy", cond_basis,
                "frozen conditional-RRR direction 1 (target D_cond = y - y_phi, "
                "train split forward pass)",
            )
            ind_info = write_projector(
                np, output_dir / f"{dataset}_{horizon}_Q1INDREVIN.npy", ind_basis,
                "diagnostic only: independent-RRR direction 1 in the same RevIN "
                "space as D_cond; the trained frozen-independent arm uses the "
                "standardized Q1 file, not this one",
            )
            record["revin"] = {
                **revin,
                "conditional_projector": cond_info,
                "revin_space_independent_projector": ind_info,
                "abs_cos_conditional_vs_independent": cosine_abs(
                    np, cond_basis[:, 0], ind_basis[:, 0]
                ),
            }
            print(
                json.dumps({
                    "event": "revin",
                    "setting": setting,
                    "pairs": revin["n_pairs"],
                    "gate_mean": revin["gate_mean"],
                    "abs_cos_cond_vs_indep": record["revin"][
                        "abs_cos_conditional_vs_independent"
                    ],
                    "file": cond_info["file"],
                }),
                flush=True,
            )
        audit.append(record)

    failed_gate = [
        record["setting"] for record in audit
        if (record.get("standardized", {}).get("reproduction_gate", {})
            .get("reproduced") is False)
    ]
    gate_report = {
        "protocol": "phaseformer-L-e17-conditional-projectors-v1",
        "min_reproduction_cos": float(args.min_reproduction_cos),
        "checked": [
            {
                "setting": record["setting"],
                **record.get("standardized", {}).get("reproduction_gate", {}),
            }
            for record in audit
            if "standardized" in record
        ],
        "failed": failed_gate,
        "note": (
            "The standardized route restates scripts/compute_top2_direction_"
            "projectors.py, so a recomputed Q1 must match the shipped E8 Q1 up to "
            "sign and floating-point noise.  Electricity-336 has no shipped "
            "projector: its Q1 is new but produced by the same code path."
        ),
    }
    index = {
        "protocol": "phaseformer-L-e17-conditional-projectors-v1",
        "lookback": int(args.seq_len),
        "period": PERIOD,
        "ridge": float(args.ridge),
        "only_train_split_read": True,
        "conditional_target": plan["conditional_target"],
        "conditional_target_route": "train_split_forward_pass",
        "conditional_phase_seed": int(args.phase_seed),
        "e14_root": args.e14_root,
        "projectors": {
            record["setting"]: {
                "dataset": record["dataset"],
                "horizon": record["horizon"],
                "q1_file": record.get("standardized", {})
                .get("projector", {}).get("file"),
                "q1_sha256": record.get("standardized", {})
                .get("projector", {}).get("sha256"),
                "q1cond_file": record.get("revin", {})
                .get("conditional_projector", {}).get("file"),
                "q1cond_sha256": record.get("revin", {})
                .get("conditional_projector", {}).get("sha256"),
                "reproduction_abs_cos": record.get("standardized", {})
                .get("reproduction_gate", {}).get("abs_cos"),
                "reproduction_reproduced": record.get("standardized", {})
                .get("reproduction_gate", {}).get("reproduced"),
                "abs_cos_conditional_vs_independent": record.get("revin", {})
                .get("abs_cos_conditional_vs_independent"),
                "phase_run_dir": record.get("revin", {}).get("run_dir"),
                "phase_checkpoint": record.get("revin", {}).get("checkpoint_path"),
            }
            for record in audit
        },
    }
    (output_dir / "projector_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "projectors.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "reproduction_gate.json").write_text(
        json.dumps(gate_report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "event": "projectors_written",
        "settings": len(audit),
        "output_dir": str(output_dir),
        "reproduction_failures": failed_gate,
        "reads_test_split": False,
    }, ensure_ascii=False))
    if failed_gate:
        raise SystemExit(
            "the E8 reproduction gate failed for: " + ", ".join(failed_gate)
            + " -- refusing to declare the independent projectors valid"
        )


if __name__ == "__main__":
    main()
