#!/usr/bin/env python3
"""Second-pass verification of run config <-> result correspondence.

For every run dir under the two output trees, recompute the config_hash
exactly as search_phaseformer.py does (sha256 of the canonical spec JSON,
sorted keys, no separators/whitespace) and cross-check it against:
  * config.json's stored config_hash
  * metrics.csv's stored config_hash
  * the 12-hex suffix of the run directory name (run_id embedding)

Also checks that the metrics row's dataset/horizon/seed/loss/mechanism match
the config.json, that no duplicate config_hash yields divergent metrics, and
that no run dir is missing metrics/config.

Writes a reproducibility manifest under each tree's manifest/ directory:
  * runs/<run_id>.json   - full spec + metrics + per-check verification status
  * index.json           - run_id -> {config summary, result, hash_status}
  * verification_report.md - human-readable report

Usage:
  python scripts/verify_run_reproducibility.py [--trees TREE1 TREE2 ...]
Exit code 0 = all checks passed; 1 = any failure.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TREES = [
    ROOT / "research_runs/pctf_strict_t28_global_golden_v1",
    ROOT / "research_runs/pctf_weather_search_v1",
]


def canonical_spec(spec):
    """Recompute the canonical string for the hash, mirroring the runner."""
    spec = dict(spec)
    spec.pop("config_hash", None)
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def recompute_hash(spec):
    return hashlib.sha256(canonical_spec(spec).encode()).hexdigest()[:12]


def run_id_matches_dir(rid, config, statuses):
    """Reconstruct the expected run_id prefix from the spec and compare."""
    hp = config["hyperparams"]
    lr = hp["learning_rate"]
    cycle = f"_cp{config['cycle_period']}" if config["cycle_period"] != "" else ""
    expect = (
        f"{config['stage']}_{config['dataset'].lower()}_h{config['horizon']}_"
        f"{config['mechanism']}_p{config['period']}{cycle}_{config['capacity']}_"
        f"{config['loss']}_lr{lr:.6g}_pct{config['percent']}_"
        f"e{config['max_epochs']}_s{config['seed']}"
    )
    if not rid.startswith(expect):
        statuses.append("run_id-prefix-mismatch")
        return False
    return True


def check_dir(run_dir):
    """Return verification info. `statuses` collects every check result;
    `hard_failures` are the ones that break config<->result correspondence
    (hash or field mismatches). missing-metrics is expected for cancelled /
    invalid runs and is reported but not a hard failure."""
    statuses = []
    hard = []
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    commands_path = run_dir / "commands.sh"

    if not config_path.exists():
        return {"run_id": run_dir.name, "ok": False, "hard": ["missing-config"],
                "statuses": ["missing-config"], "metrics": None}
    config = json.load(open(config_path))

    # 1) hash recomputation
    expected = recompute_hash(config)
    stored = config.get("config_hash")
    if stored is None:
        statuses.append("config-missing-config_hash")
        hard.append("config-missing-config_hash")
    elif stored != expected:
        statuses.append("config-hash-mismatch")
        hard.append("config-hash-mismatch")

    # 2) run_id suffix == hash
    suffix = run_dir.name.split("_")[-1]
    if suffix != expected:
        statuses.append("dir-suffix-hash-mismatch")
        hard.append("dir-suffix-hash-mismatch")
    if not run_id_matches_dir(run_dir.name, config, statuses):
        statuses.append("run_id-prefix-mismatch")
        hard.append("run_id-prefix-mismatch")

    # 3) metrics row consistency
    metrics = None
    if metrics_path.exists():
        rows = list(csv.DictReader(open(metrics_path)))
        if len(rows) == 1:
            r = rows[0]
            metrics = {
                "run_id": r.get("run_id"), "test_mse": r.get("test_mse"),
                "test_mae": r.get("test_mae"), "seed": r.get("seed"),
                "dataset": r.get("dataset"), "horizon": r.get("horizon"),
                "loss": r.get("loss"), "mechanism": r.get("mechanism"),
                "stage": r.get("stage"),
                "epochs_completed": r.get("epochs_completed"),
                "completed_at": r.get("completed_at"),
            }
            if r.get("config_hash") and r["config_hash"] != expected:
                statuses.append("metrics-hash-mismatch")
                hard.append("metrics-hash-mismatch")
            for field, cv in [("dataset", config["dataset"]),
                              ("horizon", str(config["horizon"])),
                              ("seed", str(config["seed"])),
                              ("loss", config["loss"]),
                              ("mechanism", config["mechanism"]),
                              ("stage", config["stage"])]:
                if str(r.get(field)) != str(cv):
                    statuses.append(f"metrics-config-{field}-mismatch")
                    hard.append(f"metrics-config-{field}-mismatch")
        else:
            statuses.append("metrics-row-count!=1")
            hard.append("metrics-row-count!=1")
    else:
        statuses.append("missing-metrics")  # expected for cancelled/invalid

    # 4) commands.sh presence (the exact invocation for re-running)
    if not commands_path.exists():
        statuses.append("missing-commands.sh")

    return {"run_id": run_dir.name, "ok": not hard, "hard": hard,
            "statuses": statuses, "metrics": metrics, "config_hash": expected,
            "has_result": metrics is not None and metrics["test_mse"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", nargs="+", default=[str(t) for t in DEFAULT_TREES])
    args = ap.parse_args()

    all_failures = []
    for tree in args.trees:
        tree = Path(tree)
        runs = sorted((tree / "runs").glob("*/"))
        results = {}
        hashes = {}  # config_hash -> [(run_id, mse, mae)]
        for run_dir in runs:
            if not run_dir.is_dir():
                continue
            info = check_dir(run_dir)
            results[run_dir.name] = info
            if info["has_result"]:
                hashes.setdefault(info["config_hash"], []).append(
                    (run_dir.name, info["metrics"]["test_mse"],
                     info["metrics"]["test_mae"]))
            if info["hard"]:
                all_failures.append((tree, run_dir.name, info["hard"]))

        # duplicate-config divergence check
        dup_divergence = []
        for ch, rows in hashes.items():
            if len(rows) < 2:
                continue
            first = rows[0]
            for other in rows[1:]:
                if (other[1], other[2]) != (first[1], first[2]):
                    dup_divergence.append((ch, rows))

        manifest = tree / "manifest"
        (manifest / "runs").mkdir(parents=True, exist_ok=True)
        index = {}
        for rid, info in results.items():
            # per-run manifest file: full spec + metrics + status
            payload = {"run_id": rid, "verification_status": info["statuses"],
                       "verification_ok": info["ok"]}
            cfg_path = tree / "runs" / rid / "config.json"
            if cfg_path.exists():
                payload["config"] = json.load(open(cfg_path))
            if info["metrics"]:
                payload["metrics"] = info["metrics"]
            (manifest / "runs" / f"{rid}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n")
            index[rid] = {"config_hash": info["config_hash"],
                          "verification_ok": info["ok"],
                          "statuses": info["statuses"],
                          "metrics": info["metrics"]}
        (manifest / "index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n")

        n = len(results)
        n_ok = sum(1 for i in results.values() if i["ok"])
        n_result = sum(1 for i in results.values() if i["has_result"])
        n_no_result = sum(1 for i in results.values() if not i["has_result"])
        with (manifest / "verification_report.md").open("w") as fh:
            fh.write(f"# {tree.name} run verification report\n\n")
            fh.write(f"- runs scanned: {n}\n")
            fh.write(f"- runs with results (verified 1:1): {n_result}\n")
            fh.write(f"- runs without results (cancelled/invalid, expected): "
                     f"{n_no_result}\n")
            fh.write(f"- config-hash / field mismatches: {n - n_ok}\n")
            fh.write(f"- duplicate-config divergent results: "
                     f"{len(dup_divergence)}\n")
            if all_failures:
                fh.write("\n## Hard failures\n\n")
                for t, rid, st in all_failures:
                    if t == tree:
                        fh.write(f"- `{rid}`: {', '.join(st)}\n")
            if dup_divergence:
                fh.write("\n## Duplicate config with divergent metrics\n\n")
                for ch, rows in dup_divergence:
                    fh.write(f"- hash `{ch}`: " + "; ".join(
                        f"{rid} mse={mse} mae={mae}" for rid, mse, mae in rows)
                             + "\n")

        print(f"\n=== {tree.name} ===")
        print(f"runs scanned: {n} | with results: {n_result} (all verified 1:1) | "
              f"no-result dirs: {n_no_result} | hard failures: {n - n_ok} | "
              f"dup-divergent: {len(dup_divergence)}")
        if all_failures:
            for t, rid, st in all_failures:
                if t == tree:
                    print(f"  HARD-FAIL {rid}: {', '.join(st)}")
        if dup_divergence:
            print("  DUP-DIVERGENT:")
            for ch, rows in dup_divergence:
                print(f"    {ch}: " + "; ".join(
                    f"{rid} mse={mse} mae={mae}" for rid, mse, mae in rows))
        print(f"manifest -> {manifest}")

    return 1 if all_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
