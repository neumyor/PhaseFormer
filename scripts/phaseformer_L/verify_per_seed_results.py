"""Verify the minipaper's per-seed sections against E14 results.csv, cell by cell.

Checks
  * §4.2.4 (i) and 附录 A.1 count tables: every count recomputed from the raw rows;
  * §4.2.4 (ii) and each 附录 A.2 table: all 3 seeds' MSE/MAE equal the recorded
    values, and the trailing "双指标胜 (n/3)" equals the recomputed count.

Bold markers (the "< Golden" flags) are stripped before the numeric comparison;
the flags themselves are re-derived and compared.
"""
import csv, re, collections, sys

E14 = "research_runs/phaseformer_L_e14_main_v1/results.csv"
DOC = "docs/PhaseFormer_L_minipaper.md"
G = {("ETTh1",96):(0.359,0.382),("ETTh1",192):(0.397,0.404),("ETTh1",336):(0.425,0.424),("ETTh1",720):(0.431,0.450),
     ("ETTh2",96):(0.275,0.338),("ETTh2",192):(0.341,0.376),("ETTh2",336):(0.369,0.405),("ETTh2",720):(0.402,0.436),
     ("ETTm1",96):(0.293,0.344),("ETTm1",192):(0.323,0.361),("ETTm1",336):(0.358,0.381),("ETTm1",720):(0.412,0.410),
     ("ETTm2",96):(0.163,0.256),("ETTm2",192):(0.219,0.293),("ETTm2",336):(0.269,0.326),("ETTm2",720):(0.351,0.379),
     ("Weather",96):(0.148,0.195),("Weather",192):(0.193,0.237),("Weather",336):(0.242,0.278),("Weather",720):(0.309,0.332),
     ("Electricity",96):(0.129,0.221),("Electricity",192):(0.148,0.238),("Electricity",336):(0.165,0.257),("Electricity",720):(0.201,0.285),
     ("Traffic",96):(0.361,0.238),("Traffic",192):(0.373,0.243),("Traffic",336):(0.385,0.248),("Traffic",720):(0.428,0.270)}
ARMS = ["phase_only","l_main","l_q1_4","l_q1_8","l_rcrf","a1"]
LAB = {"`phase_only`":"phase_only","PhaseFormer-L (`l_main`)":"l_main","`l_q1_4`":"l_q1_4",
       "`l_q1_8`":"l_q1_8","`l_rcrf`":"l_rcrf","`a1`":"a1"}
SEEDS = (2021,2022,2023)

R = collections.defaultdict(dict)
for r in csv.DictReader(open(E14)):
    if r["test_mse"].strip():
        R[(r["arm"], r["dataset"], int(r["horizon"]))][int(r["seed"])] = (float(r["test_mse"]), float(r["test_mae"]))

problems, checked = [], 0
doc = open(DOC, encoding="utf-8").read()

def cell_check(setting, arm, seed, raw, gm, ga):
    """raw like '**0.365555** / 0.396061'"""
    global checked
    rec = R[(arm, setting[0], setting[1])].get(seed)
    if rec is None:
        return "—" in raw
    parts = [p.strip() for p in raw.split("/")]
    for got, want, golden in zip(parts, rec, (gm, ga)):
        bold = got.startswith("**")
        num = got.strip("*")
        if abs(float(num) - want) > 5e-7:
            problems.append(f"{setting} {arm} s{seed}: doc={num} run={want:.6f}")
        if bold != (want < golden):
            problems.append(f"{setting} {arm} s{seed}: flag doc={bold} recomputed={want < golden}")
        checked += 1

# ---- 附录 A.2 tables (all arms) ----
for m in re.finditer(r"^#### (ETTh1|ETTh2|ETTm1|ETTm2|Weather|Electricity|Traffic)-(\d+)（Golden ([\d.]+)/([\d.]+)）$",
                     doc, re.M):
    ds, h, gm, ga = m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4))
    seg = doc[m.end():]
    seg = seg[:seg.index("\n#") if "\n#" in seg else len(seg)]
    for line in seg.split("\n"):
        if not line.startswith("| "): continue
        p = [x.strip() for x in line.strip("|").split("|")]
        if len(p) != 5 or p[0] not in LAB: continue
        arm = LAB[p[0]]
        for i, sd in enumerate(SEEDS):
            cell_check((ds, h), arm, sd, p[1 + i], gm, ga)
        nb = sum(1 for sd in SEEDS if sd in R[(arm, ds, h)]
                 and R[(arm, ds, h)][sd][0] < gm and R[(arm, ds, h)][sd][1] < ga)
        if int(p[4].split("/")[0]) != nb:
            problems.append(f"{ds}-{h} {arm}: 双指标胜 doc={p[4]} recomputed={nb}/3")

# ---- §4.2.4 (ii) l_main table ----
sec = doc[doc.index("#### 4.2.4"):doc.index("### 4.3 ")]
tbl = sec[sec.index("（ii）"):]
for line in tbl.split("\n"):
    if not line.startswith("| ") or "---" in line or line.startswith("| setting"): continue
    p = [x.strip() for x in line.strip("|").split("|")]
    if len(p) != 5: continue
    ds, h = p[0].rsplit("-", 1); h = int(h)
    gm, ga = G[(ds, h)]
    for i, sd in enumerate(SEEDS):
        cell_check((ds, h), "l_main", sd, p[1 + i], gm, ga)
    nb = sum(1 for sd in SEEDS if R[("l_main", ds, h)][sd][0] < gm and R[("l_main", ds, h)][sd][1] < ga)
    if int(p[4].split("/")[0]) != nb:
        problems.append(f"§4.2.4 {ds}-{h} l_main: 双指标胜 doc={p[4]} recomputed={nb}/3")

# ---- count tables (§4.2.4 (i) and A.1 must agree with each other and with the raw data) ----
def counted():
    out = {}
    for arm in ARMS:
        for crit, fn in (("both", lambda m, a, gm, ga: m < gm and a < ga),
                         ("either", lambda m, a, gm, ga: m < gm or a < ga)):
            for universe in ("main", "all"):
                n = 0
                for (d, h), gm, ga in ((k, *G[k]) for k in G):
                    if universe == "main" and d == "Traffic": continue
                    if arm not in [k[0] for k in R if (k[1], k[2]) == (d, h)]: continue
                    vals = R.get((arm, d, h))
                    if not vals: continue
                    if any(fn(m, a, gm, ga) for m, a in vals.values()): n += 1
                out[(arm, crit, universe)] = n
    return out
C = counted()
for seg_name, marker in (("§4.2.4 (i)", "#### 4.2.4"), ("A.1", APPENDIX_MARK := "### A.1 ")):
    seg = doc[doc.index(marker):]
    seg = seg[:seg.index("\n###") if "\n###" in seg[4:] else 4000]
    for line in seg.split("\n"):
        if not line.startswith("| "): continue
        p = [x.strip() for x in line.strip("|").split("|")]
        if len(p) != 6 or p[0] not in LAB: continue
        arm = LAB[p[0]]
        want = [f"{C[(arm,'both','main')]}/24", f"{C[(arm,'either','main')]}/24",
                f"{C[(arm,'both','all')]}/28", f"{C[(arm,'either','all')]}/28"]
        got = [p[1].strip("*"), p[2].strip("*"), p[3].strip("*"), p[4].strip("*")]
        if got != want:
            problems.append(f"{seg_name} {arm}: counts doc={got} recomputed={want}")

print(f"numeric cells checked: {checked}")
print(f"PROBLEMS: {len(problems)}")
for p in problems[:15]:
    print("  -", p)
sys.exit(1 if problems else 0)
