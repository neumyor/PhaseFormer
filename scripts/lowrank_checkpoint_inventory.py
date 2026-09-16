#!/usr/bin/env python3
"""Checkpoint inventory for the low-rank checkpoint information analysis.

Walks the three remote artifact roots that hold the conditioned rank-sweep
checkpoints and produces one auditable row per formal analysis cell
(dataset x rank cell x seed).  A cell can contain several duplicate training
artifacts because the runner executed the same configuration on several workers;
the module selects one deterministically and records every candidate so the
selection is auditable rather than implicit.

Reads only ``config.json`` / ``metrics.csv`` / ``status.json`` plus the
checkpoint file itself.  It never loads a dataset and never touches the test
split.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

# Formal analysis units of plan section 2.1.
FORMAL_SETTINGS: tuple[tuple[str, int], ...] = (
    ("ETTh2", 96),
    ("ETTh2", 720),
    ("ETTm2", 96),
    ("ETTm2", 192),
    ("Weather", 96),
    ("Weather", 192),
    ("Electricity", 336),
)
FORMAL_SEEDS: tuple[int, ...] = (2021, 2022, 2023)
LOW_RANK_LABELS: tuple[str, ...] = ("q=1/4", "q=1/8", "q=1/16", "q=1/32")
# Planned relative-rank ladder.  ``q=1`` (the factorized full-rank cell) exists
# for seed 2021 only and is used as a parameterization diagnostic.
RELATIVE_RANKS: tuple[float, ...] = (0.25, 0.125, 0.0625, 0.03125)
DIAGNOSTIC_RELATIVE_RANK = 1.0
LABEL_OF_RELATIVE_RANK = {
    0.25: "q=1/4",
    0.125: "q=1/8",
    0.0625: "q=1/16",
    0.03125: "q=1/32",
    1.0: "q=1",
}

# Source roots, in the order used to break a metric tie between duplicates.
SOURCE_ROOTS: tuple[str, ...] = (
    "research_runs/rank_sweep_2_stage1",
    "research_runs/rank_sweep_2_multiseed_stage1_20260914_v4",
    "research_runs/rank_sweep_2_multiseed_stage1_20260914_repair_v1",
)
SEED_ROOT_HINT = {2021: 0, 2022: 1, 2023: 1}


@dataclass
class Candidate:
    run_dir: Path
    source: str
    seed: int
    dataset: str
    horizon: int
    cell: str
    rank: int | None
    pool_factor: int
    relative_rank: float | None
    status: str
    val_mse: float | None
    val_mae: float | None
    checkpoint: Path | None
    metrics_complete: bool
    checkpoint_bytes: int = 0

    def sort_key(self, source_priority: int):
        """Prefer completed runs, then lower validation MSE, then the root order."""
        return (
            0 if self.status == "completed" else 1,
            0 if self.checkpoint is not None else 1,
            0 if self.metrics_complete else 1,
            self.val_mse if self.val_mse is not None else float("inf"),
            source_priority,
            str(self.run_dir),
        )


@dataclass
class Cell:
    setting: str
    dataset: str
    horizon: int
    seed: int
    cell: str
    rank: int | None
    pool_factor: int
    relative_rank: float | None
    requested_relative_rank: float | None = None
    is_diagnostic: bool = False
    selected: Candidate | None = None
    candidates: list[Candidate] = field(default_factory=list)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_cell(mechanism: str, hyperparams: dict) -> tuple[str, int | None, int]:
    head = hyperparams.get("weak_period_residual_head_type", "shared")
    if mechanism == "no_residual":
        return "phase_only", None, 1
    if head == "shared":
        return "direct_nlinear", None, 1
    if head == "pooled_lowrank":
        pool = int(hyperparams.get("weak_period_residual_pool_factor", 1))
        rank = int(hyperparams.get("weak_period_residual_rank", 0))
        return "low_rank", rank, pool
    return f"other:{head}", None, 1


def relative_rank_of(rank: int, horizon: int, pool_factor: int) -> float:
    """Recover the planned relative rank q from the realized integer rank."""
    max_rank = min(-(-720 // pool_factor), horizon)
    return rank / float(max_rank)


def planned_rank(pool_factor: int, relative_rank: float, horizon: int) -> int:
    """The plan's exact-rank rule: ``round(q * min(ceil(720/pool), H))``.

    The rank-sweep runner uses this rule for the multi-seed rounds and a
    multiple-of-4 variant for the seed-2021 round.  Both land on the same
    integer for every planned ``q``, so the mapping below is single-valued and
    is verified against the realized checkpoints by :func:`rank_lookup`.
    """
    max_rank = min(-(-720 // pool_factor), horizon)
    return max(1, min(max_rank, int(round(relative_rank * max_rank))))


def rank_lookup(pool_factor: int, horizon: int) -> dict[int, str]:
    """``{realized rank: planned cell label}`` for one (pool, horizon) pair."""
    lookup: dict[int, str] = {}
    for relative_rank in RELATIVE_RANKS:
        lookup.setdefault(
            planned_rank(pool_factor, relative_rank, horizon),
            LABEL_OF_RELATIVE_RANK[relative_rank],
        )
    return lookup


def requested_relative_ranks(
    repo_root: Path,
) -> dict[tuple[str, int, int, int], set[float]]:
    """``{(dataset, horizon, seed, rank): {requested q, ...}}``.

    The runner writes one ``*_results.csv`` per (dataset, horizon, seed) listing
    ``config_id``, ``relative_rank`` and ``rank``.  Those summaries are the only
    place where the *requested* relative rank survives, so they are read back
    instead of being re-derived from the realized integer rank.  Several
    requested values can share one realized rank, which is exactly why this is a
    set rather than a single value.
    """
    mapping: dict[tuple[str, int, int, int], set[float]] = {}
    for source in SOURCE_ROOTS:
        for path in sorted((repo_root / source).glob("*_results.csv")):
            with path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    if not row.get("relative_rank") or row.get("rank") in ("", None):
                        continue
                    key = (
                        str(row["dataset"]),
                        int(row["horizon"]),
                        int(row["seed"]),
                        int(row["rank"]),
                    )
                    mapping.setdefault(key, set()).add(float(row["relative_rank"]))
    return mapping


def scan_candidates(repo_root: Path) -> list[Candidate]:
    rows: list[Candidate] = []
    for source in SOURCE_ROOTS:
        runs_dir = repo_root / source / "runs"
        if not runs_dir.is_dir():
            continue
        for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
            config_path = run_dir / "config.json"
            if not config_path.is_file():
                continue
            config = json.loads(config_path.read_text())
            hyperparams = config.get("hyperparams", {})
            cell, rank, pool = classify_cell(
                config.get("mechanism", ""), hyperparams
            )
            if cell != "low_rank":
                continue
            relative = relative_rank_of(rank, int(config["horizon"]), pool)
            metrics_path = run_dir / "metrics.csv"
            row = {
                "val_mse": None,
                "val_mae": None,
                "test_mse": None,
                "test_mae": None,
            }
            metrics_complete = False
            if metrics_path.is_file():
                with metrics_path.open(newline="") as handle:
                    metrics = next(csv.DictReader(handle), None)
                if metrics:
                    for key in row:
                        raw = metrics.get(key, "")
                        row[key] = float(raw) if raw not in ("", None) else None
                    metrics_complete = all(
                        row[key] is not None
                        for key in ("val_mse", "val_mae", "test_mse", "test_mae")
                    )
            status = ""
            status_path = run_dir / "status.json"
            if status_path.is_file():
                status = json.loads(status_path.read_text()).get("status", "")
            checkpoints = sorted(
                run_dir.glob("attempts/*/checkpoints/best.ckpt"),
                key=lambda path: path.parent.parent.name,
            )
            checkpoint = checkpoints[-1] if checkpoints else None
            rows.append(
                Candidate(
                    run_dir=run_dir,
                    source=source,
                    seed=int(config.get("seed", -1)),
                    dataset=str(config.get("dataset", "")),
                    horizon=int(config.get("horizon", -1)),
                    cell=cell,
                    rank=rank,
                    pool_factor=pool,
                    relative_rank=relative,
                    status=status,
                    val_mse=row["val_mse"],
                    val_mae=row["val_mae"],
                    checkpoint=checkpoint,
                    metrics_complete=metrics_complete,
                    checkpoint_bytes=checkpoint.stat().st_size if checkpoint else 0,
                )
            )
    return rows


def build_cells(
    repo_root: Path,
    include_diagnostic: bool = True,
) -> tuple[dict[tuple[str, int, int, str], Cell], list[Candidate]]:
    """Assign every low-rank checkpoint to exactly one planned analysis cell.

    A cell is ``(setting, seed, relative-rank cell)``.  Assignment uses the
    *requested* relative rank recorded in the runner summaries, because equal
    relative ranks can share one realized integer rank: with ``H = 96`` the
    planned ``q=1`` and ``q=1/4`` cells both realize rank 24 for ETTh2-96, and
    with ``H = 720`` the planned ``q=1/16`` and ``q=1/32`` cells both realize
    rank 22.  Without this lookup the two cells would be silently merged or
    mislabelled, so a checkpoint whose realized rank carries no recorded
    requested value is reported instead of being renamed.
    """
    candidates = scan_candidates(repo_root)
    requested = requested_relative_ranks(repo_root)
    cells: dict[tuple[str, int, int, str], Cell] = {}
    for (dataset, horizon) in FORMAL_SETTINGS:
        for seed in FORMAL_SEEDS:
            for relative_rank in RELATIVE_RANKS:
                label = LABEL_OF_RELATIVE_RANK[relative_rank]
                cells[(dataset, horizon, seed, label)] = Cell(
                    setting=f"{dataset}-{horizon}",
                    dataset=dataset,
                    horizon=horizon,
                    seed=seed,
                    cell=label,
                    rank=planned_rank(1, relative_rank, horizon),
                    pool_factor=1,
                    relative_rank=relative_rank,
                )
    if include_diagnostic:
        # ``q=1`` is the factorized full-rank parameterization.  It exists for
        # seed 2021 only and never enters a cross-seed conclusion, but it has to
        # own its artifacts so that no checkpoint is orphaned.
        for (dataset, horizon) in FORMAL_SETTINGS:
            label = LABEL_OF_RELATIVE_RANK[DIAGNOSTIC_RELATIVE_RANK]
            cells[(dataset, horizon, 2021, label)] = Cell(
                setting=f"{dataset}-{horizon}",
                dataset=dataset,
                horizon=horizon,
                seed=2021,
                cell=label,
                rank=planned_rank(1, DIAGNOSTIC_RELATIVE_RANK, horizon),
                pool_factor=1,
                relative_rank=DIAGNOSTIC_RELATIVE_RANK,
                is_diagnostic=True,
            )
    unassigned: list[tuple[str, int, int, int]] = []
    for candidate in candidates:
        key = (
            candidate.dataset,
            candidate.horizon,
            candidate.seed,
            int(candidate.rank or -1),
        )
        ladder = requested.get(key)
        if candidate.pool_factor != 1 or not ladder:
            unassigned.append(key)
            continue
        matched = False
        for relative_rank in sorted(ladder):
            label = LABEL_OF_RELATIVE_RANK.get(relative_rank)
            if label is None:
                continue
            cell = cells.get((candidate.dataset, candidate.horizon, candidate.seed, label))
            if cell is None:
                continue
            cell.candidates.append(candidate)
            cell.requested_relative_rank = relative_rank
            matched = True
        if not matched:
            unassigned.append(key)
    for key, cell in cells.items():
        if not cell.candidates:
            continue
        cell.candidates.sort(
            key=lambda item: item.sort_key(SOURCE_ROOTS.index(item.source))
        )
        cell.selected = cell.candidates[0]
        realized = {candidate.rank for candidate in cell.candidates}
        if len(realized) != 1:
            raise RuntimeError(
                f"ambiguous realized ranks for {cell.setting} seed={cell.seed} "
                f"{cell.cell}: {sorted(realized)}"
            )
    if unassigned:
        raise RuntimeError(
            "low-rank checkpoints whose realized rank carries no recorded "
            f"requested relative rank: {sorted(set(unassigned))}"
        )
    return cells, candidates


def inventory_rows(repo_root: Path, hash_checkpoints: bool = True) -> list[dict]:
    cells, _ = build_cells(repo_root)
    rows = []
    for key in sorted(cells, key=lambda item: (item[0], item[1], item[2], item[3])):
        cell = cells[key]
        selected = cell.selected
        record = {
            "setting": cell.setting,
            "is_diagnostic_only": bool(cell.is_diagnostic),
            "dataset": cell.dataset,
            "horizon": cell.horizon,
            "seed": cell.seed,
            "cell": cell.cell,
            "pool_factor": cell.pool_factor,
            "rank": cell.rank if cell.rank is not None else "",
            "relative_rank_q": (
                f"{cell.relative_rank:.6f}" if cell.relative_rank is not None else ""
            ),
            "n_duplicate_candidates": len(cell.candidates),
            "selected_source": selected.source if selected else "",
            "selected_run_dir": str(selected.run_dir.relative_to(repo_root)) if selected else "",
            "selected_status": selected.status if selected else "",
            "selected_val_mse": selected.val_mse if selected else "",
            "selected_val_mae": selected.val_mae if selected else "",
            "checkpoint_path": (
                str(selected.checkpoint.relative_to(repo_root))
                if selected and selected.checkpoint
                else ""
            ),
            "checkpoint_bytes": selected.checkpoint_bytes if selected else 0,
            "checkpoint_sha256": (
                sha256_of(selected.checkpoint)
                if selected and selected.checkpoint and hash_checkpoints
                else ""
            ),
        }
        if record["relative_rank_q"] == "":
            record["relative_rank_q"] = ""
        if selected and selected.checkpoint and record["checkpoint_sha256"]:
            record["checkpoint_sha256_short"] = record["checkpoint_sha256"][:16]
        else:
            record["checkpoint_sha256_short"] = ""
        rows.append(record)
    return rows


def write_inventory(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["setting"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="research_runs/lowrank_checkpoint_information_v1/checkpoint_inventory.csv")
    parser.add_argument("--no-hash", action="store_true")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    rows = inventory_rows(repo_root, hash_checkpoints=not args.no_hash)
    write_inventory(rows, repo_root / args.output)
    missing = [row for row in rows if not row["checkpoint_path"]]
    print(f"inventory rows: {len(rows)}")
    print(f"cells without a checkpoint: {len(missing)}")
    for row in missing:
        print("  MISSING", row["setting"], row["seed"], row["cell"])
    duplicates = sum(1 for row in rows if int(row["n_duplicate_candidates"]) > 1)
    print(f"cells with duplicate training artifacts: {duplicates}")


if __name__ == "__main__":
    main()
