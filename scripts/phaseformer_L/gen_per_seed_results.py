"""Emit the per-seed (3-seed) detail for the main table, straight from E14 results.csv.

Only input: research_runs/phaseformer_L_e14_main_v1/results.csv (one row per
arm x dataset x horizon x seed, with the stage-B test read).

Outputs, both into docs/PhaseFormer_L_minipaper.md (the repo keeps a single
minipaper, so the full dump goes in as its appendix rather than a second file):
  * §4.2.4 -- the exact ">=1 seed satisfies" counts + the l_main per-seed table
  * the appendix section -- all 6 arms x 28 settings x 3 seeds

Golden is the repo-wide three-decimal reference table (same constants as
verify_main_table_repro.py); the seed-level flags are additionally checked
against a 1e-3-shifted Golden so rounding-sensitive cells are disclosed.
"""
import csv, json, collections, sys, re

E14 = "research_runs/phaseformer_L_e14_main_v1"
MINIPAPER = "docs/PhaseFormer_L_minipaper.md"
ANCHOR = "### 4.3 相位补空间的维数（7 数据集，train/validation）"
APPENDIX_HEAD = "## 7. 附录 A：主表逐 seed 全量明细（6 臂 × 28 setting × 3 seed）"

G = {("ETTh1",96):(0.359,0.382),("ETTh1",192):(0.397,0.404),("ETTh1",336):(0.425,0.424),("ETTh1",720):(0.431,0.450),
     ("ETTh2",96):(0.275,0.338),("ETTh2",192):(0.341,0.376),("ETTh2",336):(0.369,0.405),("ETTh2",720):(0.402,0.436),
     ("ETTm1",96):(0.293,0.344),("ETTm1",192):(0.323,0.361),("ETTm1",336):(0.358,0.381),("ETTm1",720):(0.412,0.410),
     ("ETTm2",96):(0.163,0.256),("ETTm2",192):(0.219,0.293),("ETTm2",336):(0.269,0.326),("ETTm2",720):(0.351,0.379),
     ("Weather",96):(0.148,0.195),("Weather",192):(0.193,0.237),("Weather",336):(0.242,0.278),("Weather",720):(0.309,0.332),
     ("Electricity",96):(0.129,0.221),("Electricity",192):(0.148,0.238),("Electricity",336):(0.165,0.257),("Electricity",720):(0.201,0.285),
     ("Traffic",96):(0.361,0.238),("Traffic",192):(0.373,0.243),("Traffic",336):(0.385,0.248),("Traffic",720):(0.428,0.270)}
DS = ["ETTh1","ETTh2","ETTm1","ETTm2","Weather","Electricity","Traffic"]
ARMS = ["phase_only","l_main","l_q1_4","l_q1_8","l_rcrf","a1"]
ARM_LABEL = {"phase_only":"`phase_only`","l_main":"PhaseFormer-L (`l_main`)","l_q1_4":"`l_q1_4`",
             "l_q1_8":"`l_q1_8`","l_rcrf":"`l_rcrf`","a1":"`a1`"}
SEEDS = (2021,2022,2023)
ALLS = [(d,h) for d in DS for h in (96,192,336,720)]
MAIN = [s for s in ALLS if s[0] != "Traffic"]

R = {}
for r in csv.DictReader(open(f"{E14}/results.csv")):
    if not r["test_mse"].strip():
        continue
    R[(r["arm"], r["dataset"], int(r["horizon"]), int(r["seed"]))] = (float(r["test_mse"]), float(r["test_mae"]))

def seeds(arm, s):
    return [(sd,) + R[(arm, s[0], s[1], sd)] for sd in SEEDS if (arm, s[0], s[1], sd) in R]

def n_both(arm, s, eps=0.0):
    gm, ga = G[s]
    return sum(1 for _, m, a in seeds(arm, s) if m < gm - eps and a < ga - eps)

def n_either(arm, s, eps=0.0):
    gm, ga = G[s]
    return sum(1 for _, m, a in seeds(arm, s) if m < gm - eps or a < ga - eps)

def union(fn, eps):
    """settings where SOME arm has >=1 seed satisfying the criterion"""
    return [s for s in ALLS if any(fn(a, s, eps) >= 1 for a in ARMS)]

def count(fn, universe):
    return sum(1 for s in universe if fn(s) >= 1)

# ---- robustness: is any seed's flag sensitive to the 3-decimal Golden? ----
close = []
for arm in ARMS:
    for s in ALLS:
        gm, ga = G[s]
        for _, m, a in seeds(arm, s):
            if abs(m - gm) < 1e-3 or abs(a - ga) < 1e-3:
                close.append((arm, s, m, a, gm, ga))
print(f"robustness: {len(close)} metric values within 1e-3 of the 3-decimal Golden")

# ================= appendix =================
APP = []
A = APP.append
A(APPENDIX_HEAD)
A("")
A("> **来源**：`research_runs/phaseformer_L_e14_main_v1/results.csv`（一行 = 一个 arm×setting×seed，"
  "492 行；test 值无缺失；唯一缺口是 `a1` 的 Traffic 4 格——该臂未跑 Traffic 附录）。")
A("> **基准**：仓库统一的三位小数 Golden 参照表（与 `verify_main_table_repro.py`、§4.2.3 相同）。")
A("> **口径**：**每格单列该 seed 自己的 test MSE/MAE**，不取均值也不取最优；加粗 = 该指标低于 Golden。")
A("> 本节由 `scripts/phaseformer_L/gen_per_seed_results.py` 从 `results.csv` 生成，可重跑复核。")
A("")
A("### A.1 精确计数（\"至少 1 个 seed 满足\"）")
A("")
A("| 臂 | 主表 24：双指标 | 主表 24：任一指标 | 全 28：双指标 | 全 28：任一指标 |")
A("|---|---:|---:|---:|---:|")
for a in ARMS:
    A(f"| {ARM_LABEL[a]} | {count(lambda s: n_both(a,s), MAIN)}/24 | {count(lambda s: n_either(a,s), MAIN)}/24 "
      f"| {count(lambda s: n_both(a,s), ALLS)}/28 | {count(lambda s: n_either(a,s), ALLS)}/28 |")
ub, ue = union(n_both, 0.0), union(n_either, 0.0)
ub_s, ue_s = union(n_both, 1e-3), union(n_either, 1e-3)
A(f"| **任一臂（并集）** | **{sum(1 for s in MAIN if s in ub)}/24** | **{sum(1 for s in MAIN if s in ue)}/24** "
  f"| **{len(ub)}/28** | **{len(ue)}/28** |")
A("")
A("- \"双指标\" = 该 seed 的 MSE 与 MAE **同时**低于 Golden；\"任一指标\" = 至少一项低于 Golden。")
A("- 双指标口径未满足（全 28）：" + "、".join(f"{d}-{h}" for d, h in ALLS if (d, h) not in ub))
A("- 任一指标口径未满足（全 28）：" + "、".join(f"{d}-{h}" for d, h in ALLS if (d, h) not in ue))
A("")
A("#### A.1.1 对 Golden 精度的稳健性")
A("")
A(f"Golden 参照表只有三位小数，而**有 {len(close)} 个 seed×指标值与它相差 <1e-3**"
  "（主要是 Traffic 的 MAE，量级 0.236–0.248 对 Golden 0.238/0.243/0.248/0.270）。")
A("因此把\"相差 <1e-3 视为并列\"再做一遍（严格判据）作为上下界：")
A("")
A("| 判据 | 宽松（原样比较） | 严格（要求低 1e-3 以上） | 受影响的 setting |")
A("|---|---:|---:|---|")
A(f"| 双指标并集 | {len(ub)}/28 | {len(ub_s)}/28 | "
  + ("、".join(f"{d}-{h}" for d, h in ALLS if ((d, h) in ub) != ((d, h) in ub_s)) or "无") + " |")
A(f"| 任一指标并集 | {len(ue)}/28 | {len(ue_s)}/28 | "
  + ("、".join(f"{d}-{h}" for d, h in ALLS if ((d, h) in ue) != ((d, h) in ue_s)) or "无") + " |")
A("")
A("⇒ 真实计数落在两列之间；引用时必须说明 Golden 为三位小数参照表。")
A("")
A("### A.2 逐 setting 明细（金标 = 该 setting 的 Golden）")
A("")
for d, h in ALLS:
    gm, ga = G[(d, h)]
    A(f"#### {d}-{h}（Golden {gm}/{ga}）")
    A("")
    A("| 臂 | seed 2021 | seed 2022 | seed 2023 | 双指标胜（n/3）|")
    A("|---|---|---|---|---:|")
    for a in ARMS:
        cells = []
        for sd, m, ac in seeds(a, (d, h)):
            fm = f"**{m:.6f}**" if m < gm else f"{m:.6f}"
            fa = f"**{ac:.6f}**" if ac < ga else f"{ac:.6f}"
            cells.append(f"{fm} / {fa}")
        while len(cells) < 3:
            cells.append("—")
        nb = n_both(a, (d, h))
        A(f"| {ARM_LABEL[a]} | " + " | ".join(cells) + f" | {nb}/3 |")
    A("")
# ================= minipaper §4.2.4 block =================
M = []
M.append("#### 4.2.4 逐 seed 明细：\"至少 1 个 seed 满足\"口径的精确计数（2026-09-21 回填）")
M.append("")
M.append("> **为什么单列一节**：§4.2 报 3 seed **均值**、§4.2.3 报**每臂最优 seed**，两者都不列逐 seed 值，")
M.append("> 因而无法精确回答\"有几个 setting 存在某个 seed 在 MSE 与 MAE 上同时（或任一）低于 Golden\"。")
M.append("> 本节直接读 E14 `results.csv`（**一行 = 一个 arm×setting×seed**，492 行，test 值无缺失；")
M.append("> 唯一缺口是 `a1` 的 Traffic 4 格——该臂未跑 Traffic 附录），逐 seed 判定，")
M.append("> 基准为仓库统一的三位小数 Golden 参照表（与 §4.2.3 相同）。")
M.append("> **全部逐格明细（6 臂 × 28 setting × 3 seed）见本稿文末「附录 A」。**")
M.append("")
M.append("**（i）精确计数（\"至少 1 个 seed 满足\"）**")
M.append("")
M.append("| 臂 | 主表 24：双指标 | 主表 24：任一指标 | 全 28：双指标 | 全 28：任一指标 |")
M.append("|---|---:|---:|---:|---:|")
for a in ARMS:
    M.append(f"| {ARM_LABEL[a]} | {count(lambda s: n_both(a,s), MAIN)}/24 | {count(lambda s: n_either(a,s), MAIN)}/24 "
             f"| {count(lambda s: n_both(a,s), ALLS)}/28 | {count(lambda s: n_either(a,s), ALLS)}/28 |")
M.append(f"| **任一臂（并集）** | **{sum(1 for s in MAIN if s in ub)}/24** | **{sum(1 for s in MAIN if s in ue)}/24** "
         f"| **{len(ub)}/28** | **{len(ue)}/28** |")
M.append("")
M.append(f"- 双指标口径未满足（全 28）：{('、'.join(f'{d}-{h}' for d,h in ALLS if (d,h) not in ub))}；")
M.append(f"  主表 24 内未满足：{('、'.join(f'{d}-{h}' for d,h in MAIN if (d,h) not in ub))}。")
M.append(f"- 任一指标口径未满足（全 28）：{('、'.join(f'{d}-{h}' for d,h in ALLS if (d,h) not in ue))}；")
M.append(f"  主表 24 内未满足：{('、'.join(f'{d}-{h}' for d,h in MAIN if (d,h) not in ue))}。")
M.append("- **对 Golden 精度的稳健性**：Golden 参照表只有三位小数，**有 "
         f"{len(close)} 个 seed×指标值与其相差 <1e-3**（主要是 Traffic 的 MAE）。"
         f"按\"相差 <1e-3 视为并列\"的严格判据复核：双指标并集 **{len(ub_s)}/28**、"
         f"任一指标并集 **{len(ue_s)}/28**（宽松判据为 {len(ub)}/28、{len(ue)}/28）"
         f"⇒ 真实计数落在两者之间；受影响的 setting："
         + ("、".join(f"{d}-{h}" for d, h in ALLS
                      if ((d, h) in ub) != ((d, h) in ub_s) or ((d, h) in ue) != ((d, h) in ue_s)) or "无") + "。")
M.append("")
M.append("**（ii）PhaseFormer-L（`l_main`）逐 seed 明细**（MSE / MAE；**加粗 = 该项低于 Golden**）")
M.append("")
M.append("| setting | seed 2021 | seed 2022 | seed 2023 | 双指标胜（n/3）|")
M.append("|---|---|---|---|---:|")
for d, h in ALLS:
    gm, ga = G[(d, h)]
    cells = []
    for sd, m, a in seeds("l_main", (d, h)):
        fm = f"**{m:.6f}**" if m < gm else f"{m:.6f}"
        fa = f"**{a:.6f}**" if a < ga else f"{a:.6f}"
        cells.append(f"{fm} / {fa}")
    while len(cells) < 3:
        cells.append("—")
    M.append(f"| {d}-{h} | " + " | ".join(cells) + f" | {n_both('l_main', (d,h))}/3 |")
M.append("")
M.append("> **两口径的分工**：§4.2 的 3-seed 均值口径仍为主张 A–D 的唯一判据；"
         "§4.2.3 的\"最优 seed\"与本节\"逐 seed\"都属 **test-set selection 的条件性读数**，")
M.append("> 只用于回答\"能不能赢\"，不得改写判定。")
M.append("> 八格定向调参 winner 的逐 seed 值见复现手册 §2.3（同为零缺失三 seed 记录）。")
block = "\n".join(M) + "\n"

mp = open(MINIPAPER, encoding="utf-8").read()
mp = re.sub(r"#### 4\.2\.4.*?(?=\n### 4\.3 )", "", mp, flags=re.S)   # idempotent replace
mp = re.sub(r"\n" + re.escape(APPENDIX_HEAD) + r".*$", "", mp, flags=re.S)  # drop old appendix
i = mp.index(ANCHOR)
mp = mp[:i] + block + "\n" + mp[i:]
mp = mp.rstrip() + "\n\n" + "\n".join(APP) + "\n"
open(MINIPAPER, "w", encoding="utf-8").write(mp)

print(f"wrote §4.2.4 ({len(M)} lines) and 附录 A ({len(APP)} lines) into {MINIPAPER}")
print("counts:", {a: (count(lambda s: n_both(a,s), ALLS), count(lambda s: n_either(a,s), ALLS)) for a in ARMS})
print("union:", (len(ub), len(ue)), "of 28 | strict:", (len(ub_s), len(ue_s)))
