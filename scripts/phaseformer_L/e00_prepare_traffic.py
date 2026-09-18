#!/usr/bin/env python3
"""Prepare the Traffic dataset for this repository's data loader.

Source: ``laiguokun/multivariate-time-series-data`` ``traffic/traffic.txt.gz``
(862 sensors, 17,544 hourly steps, comma separated, **no header and no date
column**).  ``src/dataset/data_loader.py`` requires a ``date`` column because
``Dataset_Custom.__read_data__`` reads ``df_raw[["date"]]`` for time features,
and every other dataset in ``resources/all_datasets/`` carries one
(``electricity.csv``: ``date,0,1,...``).

So this script only re-frames the published data: it prepends the canonical
hourly timestamps and a ``date,0..861`` header.  It does not resample, scale,
reorder, drop, or otherwise alter any value.

The timestamps are the ones the upstream release implies: the series is hourly
and covers exactly 17,544 steps = 731 days = 2015-01-01 00:00:00 through
2016-12-31 23:00:00 inclusive, which is the 2015-2016 window used for the
PEMS/Traffic benchmark splits.

Usage::

    /home/yyk/yyk03/miniconda3/envs/time/bin/python \
        scripts/phaseformer_L/e00_prepare_traffic.py \
        --source ~/niuyiming/traffic_dl/traffic.txt \
        --output resources/all_datasets/traffic/traffic.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Canonical start of the released Traffic series (hourly, 2015-01-01 00:00).
START_TIMESTAMP = "2015-01-01 00:00:00"
EXPECTED_STEPS = 17544
EXPECTED_CHANNELS = 862


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", default="", help="optional JSON audit manifest")
    parser.add_argument("--force", action="store_true", help="overwrite existing output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite existing {args.output}; pass --force to replace"
        )

    # The upstream file has no header, so read every column as data.
    frame = pd.read_csv(args.source, header=None, dtype=np.float64)
    n_steps, n_channels = frame.shape
    print(f"source  : {args.source}")
    print(f"shape   : {n_steps} steps x {n_channels} channels")

    if n_steps != EXPECTED_STEPS:
        raise SystemExit(f"unexpected number of steps {n_steps} != {EXPECTED_STEPS}")
    if n_channels != EXPECTED_CHANNELS:
        raise SystemExit(
            f"unexpected channel count {n_channels} != {EXPECTED_CHANNELS}"
        )

    values = frame.to_numpy()
    n_nan = int(np.isnan(values).sum())
    if n_nan:
        raise SystemExit(f"source contains {n_nan} NaN values; refusing to proceed")

    # A missing value would have been read as an empty field and is already
    # rejected by read_csv; assert finiteness explicitly for the audit trail.
    if not np.isfinite(values).all():
        raise SystemExit("source contains non-finite values")

    timestamps = pd.date_range(
        start=START_TIMESTAMP, periods=n_steps, freq="h"
    )

    # This repository's multi-variate CSVs name their LAST data column ``OT``:
    # ``Dataset_Custom_Multi.__read_data__`` unconditionally runs
    # ``cols.remove(self.target)`` with ``dataset_args.target == "OT"``
    # (config/base_config.py), then appends the target back last. The sibling
    # datasets follow the same convention -- resources/all_datasets/electricity/
    # electricity.csv ends in ``...,319,OT`` and weather/weather.csv ends in
    # ``...,Tlog (degC),OT`` -- so the released unnamed columns 0..861 are
    # relabelled with 861 -> OT. With ``features="M"`` every column is still a
    # model input, so this rename changes no value and no channel count.
    columns = [str(i) for i in range(n_channels)]
    columns[-1] = "OT"
    out = pd.DataFrame(values, columns=columns)
    out.insert(0, "date", timestamps.strftime("%Y-%m-%d %H:%M:%S"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    # Re-read what we just wrote and assert the loader contract, so a silent
    # header mistake fails here instead of inside a GPU smoke run.
    check = pd.read_csv(args.output, nrows=5)
    expected_header = ["date"] + [str(i) for i in range(n_channels - 1)] + ["OT"]
    if list(check.columns) != expected_header:
        raise SystemExit(
            "written CSV header does not match the required contract: "
            f"got {list(check.columns)[:4]}...{list(check.columns)[-2:]}"
        )
    if "OT" not in check.columns:
        raise SystemExit("target column 'OT' missing from written CSV")
    if check.shape[1] != n_channels + 1:
        raise SystemExit(
            f"written CSV has {check.shape[1]} columns, expected {n_channels + 1}"
        )
    print(f"verify  : header OK ({n_channels} data columns, target='OT' last)")

    manifest = {
        "role": "dataset preparation (not an experiment)",
        "source": str(args.source),
        "source_sha256": sha256_of(args.source),
        "output": str(args.output),
        "output_sha256": sha256_of(args.output),
        "rows": int(n_steps),
        "channels": int(n_channels),
        "nan_values": n_nan,
        "start_timestamp": START_TIMESTAMP,
        "end_timestamp": str(timestamps[-1]),
        "value_min": float(values.min()),
        "value_max": float(values.max()),
        "header": f"date,0..{n_channels - 2},OT",
        "transformations": [
            "prepend hourly date column (2015-01-01 00:00, 17544 steps)",
            f"prepend header date,0..{n_channels - 2},OT "
            "(last released column renamed 861 -> OT to match the repository "
            "convention; required by Dataset_Custom_Multi `cols.remove(target)`)",
        ],
        "target_column": "OT (last column; renamed from 861, no values changed)",
        "unmodified_fields": "all 862 value columns, row order, and row count",
    }
    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"manifest: {args.manifest}")

    print(f"written : {args.output}")
    print(f"end     : {manifest['end_timestamp']}")
    print(f"range   : [{manifest['value_min']:.4f}, {manifest['value_max']:.4f}]")
    print(f"sha256  : {manifest['output_sha256']}")


if __name__ == "__main__":
    main()
