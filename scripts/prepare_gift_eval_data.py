"""Convert GIFT-Eval hourly arrow datasets into the main-table csv layout.

Output: one csv per dataset with columns ``date, <item_id>..., OT`` (time x channel,
the last channel is renamed ``OT`` because the loaders use it as the target column).
The 7:1:2 train/val/test split is applied by the loaders, not here.

Usage:
    python -I scripts/prepare_gift_eval_data.py <gift_eval_data_dir> <all_datasets_dir>
Example:
    python -I scripts/prepare_gift_eval_data.py ~/Code/gift_eval_data resources/all_datasets
"""

import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa

# GIFT-Eval folder -> (output subdir, output csv name)
DATASETS = {
    "LOOP_SEATTLE": ("loop_seattle", "loop_seattle.csv"),
    "M_DENSE": ("m_dense", "m_dense.csv"),
}


def read_arrow(path):
    with pa.memory_map(path) as source:
        return pa.ipc.open_stream(source).read_all()


def convert(src_root, out_root, name):
    subdir, csv_name = DATASETS[name]
    table = read_arrow(os.path.join(src_root, name, "H", "data-00000-of-00001.arrow"))
    rows = table.to_pylist()
    lengths = {len(r["target"]) for r in rows}
    starts = {str(r["start"]) for r in rows}
    assert len(lengths) == 1 and len(starts) == 1, "series must be aligned"
    data = np.array([r["target"] for r in rows], dtype=np.float64).T  # (time, channel)
    assert not np.isnan(data).any()

    columns = [str(r["item_id"]) for r in rows]
    columns[-1] = "OT"
    df = pd.DataFrame(data, columns=columns)
    df.insert(0, "date", pd.date_range(rows[0]["start"], periods=len(df), freq="h"))

    out_dir = os.path.join(out_root, subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, csv_name)
    df.to_csv(out_path, index=False)
    train = data[: int(len(df) * 0.7)]
    print(
        f"{name}: {len(df)} steps x {data.shape[1]} channels -> {out_path}; "
        f"{df['date'].iloc[0]} .. {df['date'].iloc[-1]}; "
        f"zero-variance channels in train: {int((train.std(0) == 0).sum())}"
    )


if __name__ == "__main__":
    src_root, out_root = (os.path.expanduser(p) for p in sys.argv[1:3])
    for dataset in DATASETS:
        convert(src_root, out_root, dataset)
