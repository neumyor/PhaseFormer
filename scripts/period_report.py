#!/usr/bin/env python3
"""Summarise best test MSE per (horizon, period) across stages."""
import glob
import json
import os
import re
import sys

BASE = sys.argv[1]
GOLDEN = {96: 0.293, 192: 0.323, 336: 0.358, 720: 0.412}

best = {}
for stage in ("stage1", "ettm1_period", "ettm1_stage3"):
    for d in glob.glob(os.path.join(BASE, stage, "runs", "*")):
        f = os.path.join(d, "result.json")
        if not os.path.exists(f):
            continue
        name = os.path.basename(d)
        try:
            r = json.load(open(f))
        except Exception:
            continue
        m = r.get("best") or {}
        if "test_mse" not in m:
            continue
        h = r.get("horizon")
        pm = re.search(r"__p(\d+)$", name)
        period = pm.group(1) if pm else "24"
        key = (h, period)
        if key not in best or m["test_mse"] < best[key][0]:
            best[key] = (m["test_mse"], name)

print("%4s %10s %10s %10s %10s %8s %8s %10s" %
      ("h", "p24", "p48", "p96", "best", "golden", "vsG", "gap2%"))
for h in (96, 192, 336, 720):
    row = [best.get((h, p), (float("nan"), ""))[0] for p in ("24", "48", "96")]
    vals = [x for x in row if x == x]
    bmin = min(vals) if vals else float("nan")
    g = GOLDEN[h]
    print("%4d %10.6f %10.6f %10.6f %10.6f %8.3f %+7.2f%% %10.6f" %
          (h, row[0], row[1], row[2], bmin, g, (bmin / g - 1) * 100, bmin - g * 0.98))

print()
for h in (96, 192, 336, 720):
    for p in ("24", "48", "96"):
        if (h, p) in best:
            print("h%d p%s: %.6f  %s" % (h, p, best[(h, p)][0], best[(h, p)][1]))
