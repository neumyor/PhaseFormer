import os
import sys
import csv
import time
import glob
import subprocess
from datetime import datetime
from typing import List, Dict, Optional


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(REPO_ROOT, "log", "training_results", "PhaseFormer")


RUN_SCRIPTS_ORDER = [
    "run_etth1.py",
    "run_etth2.py",
    "run_ettm1.py",
    "run_ettm2.py",
    "run_electricity.py",
    "run_traffic.py",
    "run_weather.py",
]


def list_existing_summaries() -> List[str]:
    os.makedirs(LOG_DIR, exist_ok=True)
    return sorted(glob.glob(os.path.join(LOG_DIR, "summary_*.csv")))


def run_script(path: str) -> int:
    env = os.environ.copy()
    cmd = [sys.executable, path]
    print(f"\n=== Running: {os.path.basename(path)} ===")
    print("Command:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    return proc.returncode


def find_new_summary(before: List[str]) -> Optional[str]:
    after = set(list_existing_summaries())
    before_set = set(before)
    new_files = sorted(after - before_set)
    if new_files:
        return new_files[-1]
    # fallback: pick the most recent file
    all_files = list(after)
    if not all_files:
        return None
    return max(all_files, key=os.path.getmtime)


def read_summary_csv(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: (v if v is not None else "") for k, v in row.items()})
    except Exception as e:
        print(f"Warning: failed to read {csv_path}: {e}")
    return rows


def normalize_row(row: Dict[str, str], run_file: str) -> Dict[str, str]:
    dataset = row.get("dataset") or row.get("data") or ""
    horizon = row.get("horizon") or row.get("pred_len") or row.get("pred_len_hrs") or ""
    lookback = row.get("lookback") or row.get("seq_len") or ""
    test_mae = row.get("test_mae") or row.get("mae") or ""
    test_mse = row.get("test_mse") or row.get("mse") or ""
    learning_rate = row.get("learning_rate") or row.get("lr") or ""
    log_dir = row.get("log_dir") or row.get("logdir") or ""
    model_layers = row.get("layers") or row.get("phase_layers") or ""
    latent_dim = row.get("latent_dim") or row.get("lat_dim") or ""
    routers = row.get("routers") or row.get("phase_num_routers") or ""

    return {
        "dataset": str(dataset),
        "pred_len": str(horizon),
        "seq_len": str(lookback),
        "run_file": run_file,
        "layers": str(model_layers),
        "latent_dim": str(latent_dim),
        "routers": str(routers),
        "learning_rate": str(learning_rate),
        "test_mae": str(test_mae),
        "test_mse": str(test_mse),
        "log_dir": str(log_dir),
    }


def write_merged_csv(rows: List[Dict[str, str]], out_path: str) -> None:
    if not rows:
        print("No rows to write; skipping merged CSV.")
        return
    fieldnames = [
        "dataset",
        "pred_len",
        "seq_len",
        "run_file",
        "layers",
        "latent_dim",
        "routers",
        "learning_rate",
        "test_mae",
        "test_mse",
        "log_dir",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\n✅ Merged results saved to: {out_path}")


def main():
    # Allow selecting a subset by CLI arg if needed
    # Example: python run_all_experiments.py run_etth1.py run_weather.py
    run_list = RUN_SCRIPTS_ORDER
    if len(sys.argv) > 1:
        run_list = [x for x in sys.argv[1:] if x.endswith(".py")]
        if not run_list:
            print("No valid run_*.py provided; falling back to default order.")
            run_list = RUN_SCRIPTS_ORDER

    print("Planned runs:")
    for s in run_list:
        print(" -", s)

    os.makedirs(LOG_DIR, exist_ok=True)
    merged_rows: List[Dict[str, str]] = []

    for script in run_list:
        script_path = os.path.join(REPO_ROOT, script)
        if not os.path.exists(script_path):
            print(f"Skip missing script: {script}")
            continue

        before = list_existing_summaries()
        code = run_script(script_path)
        if code != 0:
            print(f"❌ Script failed ({code}): {script}")
            continue

        # Wait briefly for file system flush
        time.sleep(1.0)
        new_csv = find_new_summary(before)
        if not new_csv:
            print(f"⚠️ No new summary CSV detected for {script}.")
            continue
        print(f"Found summary: {new_csv}")

        rows = read_summary_csv(new_csv)
        for row in rows:
            merged_rows.append(normalize_row(row, run_file=os.path.basename(script)))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(LOG_DIR, f"all_experiments_summary_{ts}.csv")
    write_merged_csv(merged_rows, out_csv)
    print("\nHow to run this orchestrator next time:")
    print(f"  python {os.path.basename(__file__)}")
    print("Or to run a subset:")
    print(f"  python {os.path.basename(__file__)} run_etth1.py run_weather.py")


if __name__ == "__main__":
    main()


