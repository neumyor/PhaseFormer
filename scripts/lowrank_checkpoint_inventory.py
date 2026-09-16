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
    status: str
    val_mse: float | None
    val_mae: float | None
    checkpoint: Path | None
    metrics_complete: bool
    checkpoint_bytes: int = 0
    config_q: float | None = None
    requested_relative_rank: float | None = None
    assignment: str = ""

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


def nearest_cell(requested_q: float, candidates: list[float]) -> float:
    """Relative-rank cell whose planned q is closest to a requested q.

    ``config_id`` in the runner summaries spells the requested fraction
    explicitly (for example ``pool1_q0.0305556_r22``), which is the only place
    that number survives: the checkpoint stores just the realized integer rank.
    The seed-2021 sweep requested slightly different fractions for its smallest
    cell (``22/720`` and ``10/336``), so the label is chosen by nearest planned
    ``q`` rather than by exact equality.
    """
    if not candidates:
        raise ValueError("no candidate relative ranks")
    return min(candidates, key=lambda value: (abs(value - requested_q), -value))


def max_rank(pool_factor: int, horizon: int) -> int:
    return min(-(-720 // pool_factor), horizon)


def planned_rank(pool_factor: int, relative_rank: float, horizon: int) -> int:
    """The plan's exact-rank rule: ``round(q * min(ceil(720/pool), H))``.

    The multi-seed rank-sweep runner uses this rule; the seed-2021 round uses a
    multiple-of-4 variant of the same formula.  The two agree on every planned
    ``q`` except the smallest one at horizon 720, where they differ by one
    integer step -- which is why :func:`rank_ladder` accepts both.
    """
    ceiling = max_rank(pool_factor, horizon)
    return max(1, min(ceiling, int(round(relative_rank * ceiling))))


def rounding_rank(pool_factor: int, relative_rank: float, horizon: int) -> int:
    """The seed-2021 variant: round ``q * ceiling`` to a multiple of four."""
    ceiling = max_rank(pool_factor, horizon)
    value = relative_rank * ceiling
    return max(4, min(ceiling, int(4 * round(value / 4))))


def rank_ladder(pool_factor: int, horizon: int, ranks: list[float]) -> dict[float, int]:
    """``{requested q: realized rank}`` for one (pool, horizon, q-set)."""
    ladder = {}
    for relative_rank in ranks:
        if relative_rank < 1.0:
            ladder[relative_rank] = planned_rank(pool_factor, relative_rank, horizon)
        else:
            ladder[relative_rank] = max_rank(pool_factor, horizon)
    return ladder


def ladder_q_for_rank(
    pool_factor: int,
    horizon: int,
    ranks: list[float],
    realized: int,
    strategy: str = "exact",
) -> float | None:
    """Which requested ``q`` realizes ``realized`` under one rank rule."""
    if strategy == "exact":
        ladder = rank_ladder(pool_factor, horizon, ranks)
    elif strategy == "rounding":
        ladder = {
            q: rounding_rank(pool_factor, q, horizon)
            for q in ranks
            if q < 1.0
        }
        if 1.0 in ranks:
            ladder[1.0] = max_rank(pool_factor, horizon)
    else:
        raise ValueError(f"unknown rank strategy {strategy!r}")
    hits = [q for q, rank in ladder.items() if rank == realized]
    if not hits:
        return None
    return max(hits)


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


def _flags_from_command(command: list) -> tuple[str | None, int | None, int | None]:
    """Read ``--dataset/--horizon/--seed`` out of a bare command list."""

    def value(flag: str):
        if flag in command:
            index = command.index(flag) + 1
            if index < len(command):
                return command[index]
        return None

    dataset = value("--dataset")
    horizon = value("--horizon")
    seed = value("--seed")
    return (
        str(dataset) if dataset is not None else None,
        int(horizon) if horizon is not None else None,
        int(seed) if seed is not None else None,
    )


def runner_ladders(
    repo_root: Path,
) -> dict[tuple[str, int, int], list[float]]:
    """``{(dataset, horizon, seed): [requested q, ...]}`` from job manifests.

    The multi-seed runner keeps the requested relative-rank ladder only in its
    job manifests, so the ladder is recovered from the job entries there and
    unioned across every job that covers the same (dataset, horizon, seed).
    Three shapes occur in this repository:

    * ``--relative-ranks`` inside a job ``command`` (the v4 sweep);
    * a ``config`` label such as ``q=0.03125`` plus an explicit ``overrides``
      rank (the repair driver); and
    * a top-level ``relative_ranks`` list covering every job (the seed-2021
      sweep).
    """
    ladders: dict[tuple[str, int, int], set[float]] = {}
    for source in SOURCE_ROOTS:
        for path in sorted((repo_root / source).glob("*manifest.json")):
            try:
                payload = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            for job in payload.get("jobs") or []:
                if isinstance(job, dict):
                    command = job.get("command") or []
                    dataset = job.get("dataset", payload.get("dataset"))
                    horizon = job.get("horizon", payload.get("horizon"))
                    seed = job.get("seed", payload.get("seed"))
                    if dataset is None and command:
                        dataset, horizon, seed = _flags_from_command(command)
                    ranks: set[float] = set()
                    if "--relative-ranks" in command:
                        raw = command[command.index("--relative-ranks") + 1]
                        ranks.update(float(item) for item in raw.split(",") if item)
                    config_label = str(job.get("config", ""))
                    if config_label.startswith("q="):
                        ranks.add(float(config_label[2:]))
                    if not ranks and payload.get("relative_ranks"):
                        ranks.update(float(item) for item in payload["relative_ranks"])
                else:
                    command = list(job)
                    dataset, horizon, seed = _flags_from_command(command)
                    ranks = set()
                    if "--relative-ranks" in command:
                        raw = command[command.index("--relative-ranks") + 1]
                        ranks.update(float(item) for item in raw.split(",") if item)
                if dataset is None or horizon is None or not ranks:
                    continue
                # A manifest that does not name a seed covers every seed of its
                # setting; the caller unions those ladders in anyway.
                seeds = (seed,) if seed is not None else FORMAL_SEEDS
                for item in seeds:
                    ladders.setdefault((str(dataset), int(horizon), int(item)), set()).update(ranks)
    return {key: sorted(value) for key, value in ladders.items()}


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
) -> tuple[dict[tuple[str, int, int, str], Cell], list[Candidate], list[dict], list[dict]]:
    """Assign every low-rank checkpoint to exactly one planned analysis cell.

    A cell is ``(setting, seed, relative-rank cell)``.  Assignment uses the
    *requested* relative rank, which is recovered in two steps because equal
    relative ranks can share one realized integer rank:

    * per-run, from the runner's ``*_results.csv`` when it exists (seed 2021 and
      a few multi-seed cells);
    * otherwise from the runner job manifests, which record the requested
      ``--relative-ranks`` ladder for the whole (dataset, horizon, seed) job.

    The realized rank is then mapped back through the *exact-rank* rule of the
    plan first and through the seed-2021 rounding variant second, because the
    multi-seed rounds and the seed-2021 round used those two rules.  Both rules
    agree on every planned ``q`` except ``q=1/32`` at horizon 720 (22 versus 23),
    so the fallback is recorded as an inferred label rather than silently
    merged; a realized rank that matches no ladder at all is reported.
    """
    candidates = scan_candidates(repo_root)
    requested = requested_relative_ranks(repo_root)
    ladders = runner_ladders(repo_root)
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
                rank=max_rank(1, horizon),
                pool_factor=1,
                relative_rank=DIAGNOSTIC_RELATIVE_RANK,
                is_diagnostic=True,
            )

    unassigned: list[tuple[str, int, int, int]] = []
    inferred: list[dict] = []
    cell_of_rank: dict[tuple[str, int, int, int], set[str]] = {}
    planned_cells = sorted(RELATIVE_RANKS)
    for candidate in candidates:
        key = (
            candidate.dataset,
            candidate.horizon,
            candidate.seed,
            int(candidate.rank or -1),
        )
        if candidate.pool_factor != 1:
            unassigned.append(key)
            continue
        summary_q = requested.get(key, set())
        if summary_q:
            relative_rank = nearest_cell(max(summary_q), planned_cells)
            how = "runner_summary"
            candidate.config_q = max(summary_q)
        else:
            ladder = sorted(
                ladders.get((candidate.dataset, candidate.horizon, candidate.seed), ())
            )
            relative_rank = ladder_q_for_rank(
                candidate.pool_factor, candidate.horizon, ladder, int(candidate.rank), "exact"
            )
            how = "manifest_ladder_exact_rank"
            if relative_rank is None:
                relative_rank = ladder_q_for_rank(
                    candidate.pool_factor, candidate.horizon, ladder, int(candidate.rank), "rounding"
                )
                how = "manifest_ladder_rounding_rank"
        label = LABEL_OF_RELATIVE_RANK.get(relative_rank)
        cell = cells.get((candidate.dataset, candidate.horizon, candidate.seed, label))
        if relative_rank is None or cell is None:
            unassigned.append(key)
            continue
        candidate.assignment = how
        candidate.requested_relative_rank = relative_rank
        cell.candidates.append(candidate)
        cell.requested_relative_rank = relative_rank
        cell_of_rank.setdefault((candidate.dataset, candidate.horizon, candidate.seed, int(candidate.rank or -1)), set()).add(cell.cell)
        if how != "runner_summary":
            inferred.append(
                {
                    "setting": cell.setting,
                    "seed": cell.seed,
                    "cell": cell.cell,
                    "rank": candidate.rank,
                    "how": how,
                    "run_dir": str(candidate.run_dir.relative_to(repo_root)),
                }
            )
    # A realized rank that two different planned cells both claim is a genuine
    # label collision.  It happens at horizon 720, where the plan's ``q=1/16``
    # and ``q=1/32`` both realize rank 22; the requested fraction recorded in
    # the runner summary resolves it above, and the collision is recorded so the
    # report can disclose which artifact carries which label.
    collisions = [
        {"dataset": key[0], "horizon": key[1], "seed": key[2], "rank": key[3],
         "cells": sorted(value)}
        for key, value in cell_of_rank.items()
        if len(value) > 1
    ]
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
    return cells, candidates, inferred, collisions


def inventory_rows(
    repo_root: Path,
    hash_checkpoints: bool = True,
) -> tuple[list[dict], list[dict], list[dict]]:
    cells, candidates, inferred, collisions = build_cells(repo_root)
    del candidates
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
            "rank_assignment": (
                cell.selected.assignment if cell.selected else ""
            ),
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
        if selected and selected.checkpoint and record["checkpoint_sha256"]:
            record["checkpoint_sha256_short"] = record["checkpoint_sha256"][:16]
        else:
            record["checkpoint_sha256_short"] = ""
        rows.append(record)
    return rows, inferred, collisions


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
    rows, inferred, collisions = inventory_rows(
        repo_root, hash_checkpoints=not args.no_hash
    )
    write_inventory(rows, repo_root / args.output)
    missing = [row for row in rows if not row["checkpoint_path"]]
    print(f"inventory rows: {len(rows)}")
    print(f"cells without a checkpoint: {len(missing)}")
    for row in missing:
        print("  MISSING", row["setting"], row["seed"], row["cell"])
    duplicates = sum(1 for row in rows if int(row["n_duplicate_candidates"]) > 1)
    print(f"cells with duplicate training artifacts: {duplicates}")
    formal = [row for row in rows if not row["is_diagnostic_only"]]
    print(f"formal rows: {len(formal)}, diagnostic-only rows: {len(rows) - len(formal)}")
    inferred = [row for row in rows if row["rank_assignment"] not in ("runner_summary", "")]
    print(f"rows whose q was inferred from a manifest ladder: {len(inferred)}")
    for row in inferred:
        print("  INFERRED", row["setting"], row["seed"], row["cell"], row["rank"], row["rank_assignment"])
    print("q values in the inventory:", sorted({row["relative_rank_q"] for row in rows}))
    print(f"rank labels resolved from a manifest ladder: {len(inferred)}")
    for entry in inferred:
        print(
            "  LADDER", entry["setting"], entry["seed"], entry["cell"],
            f"rank={entry['rank']}", entry["how"],
        )
    print(f"realized ranks claimed by two cells: {len(collisions)}")
    for entry in collisions:
        print("  COLLISION", entry)


if __name__ == "__main__":
    main()
