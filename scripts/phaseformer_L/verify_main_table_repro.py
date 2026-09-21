"""Verify the main-table handbook: every row's parameters and metrics must match
the run that the row claims, read from the run dirs / stage-B record.

Checks per row:
  * the (arm, dataset, horizon, seed) triple resolves to a real run config;
  * the printed gate_init / lr / head / rank equal that config's values;
  * the printed test MSE/MAE equal that run's recorded test metrics;
  * the "best seed" is indeed the row's worst-of-two-gaps optimum over the 3 seeds.
"""
import re, json, csv, glob

DOC = "docs/PhaseFormer_L_main_table_repro.md"
GOLDEN = {("ETTh1",96):(0.359,0.382),("ETTh1",192):(0.397,0.404),("ETTh1",336):(0.425,0.424),("ETTh1",720):(0.431,0.450),
          ("ETTh2",96):(0.275,0.338),("ETTh2",192):(0.341,0.376),("ETTh2",336):(0.369,0.405),("ETTh2",720):(0.402,0.436),
          ("ETTm1",96):(0.293,0.344),("ETTm1",192):(0.323,0.361),("ETTm1",336):(0.358,0.381),("ETTm1",720):(0.412,0.410),
          ("ETTm2",96):(0.163,0.256),("ETTm2",192):(0.219,0.293),("ETTm2",336):(0.269,0.326),("ETTm2",720):(0.351,0.379),
          ("Weather",96):(0.148,0.195),("Weather",192):(0.193,0.237),("Weather",336):(0.242,0.278),("Weather",720):(0.309,0.332),
          ("Electricity",96):(0.129,0.221),("Electricity",192):(0.148,0.238),("Electricity",336):(0.165,0.257),("Electricity",720):(0.201,0.285),
          ("Traffic",96):(0.361,0.238),("Traffic",192):(0.373,0.243),("Traffic",336):(0.385,0.248),("Traffic",720):(0.428,0.270)}
ARMBY = {"`phase_only`":"phase_only","`l_main` (PhaseFormer-L)":"l_main","`l_q1_4`":"l_q1_4",
         "`l_q1_8`":"l_q1_8","`l_rcrf`":"l_rcrf","`a1`":"a1"}
E14 = "research_runs/phaseformer_L_e14_main_v1"
records = {(r["arm"], r["dataset"], int(r["horizon"]), int(r["seed"])): r
           for r in csv.DictReader(open(f"{E14}/results.csv"))}

def metrics_of(run_dir, reason):
    mf = glob.glob(run_dir + "/metrics.csv") or glob.glob(run_dir + "/runs/*/metrics.csv")
    if mf:
        m = next(csv.DictReader(open(mf[0])))
        if m.get("test_mse","").strip():
            return float(m["test_mse"]), float(m["test_mae"])
    return None

def config_of(run_dir):
    cf = glob.glob(run_dir + "/config.json") or glob.glob(run_dir + "/runs/*/config.json")
    return json.load(open(cf[0])) if cf else None

text = open(DOC, encoding="utf-8").read()
blocks = re.split(r"^### ", text, flags=re.M)[1:]
problems, checked = [], 0
for b in blocks:
    title = b.split("\n", 1)[0].strip()
    ds, h = title.rsplit("-", 1); h = int(h)
    gm, ga = GOLDEN[(ds, h)]
    for line in b.split("\n"):
        if not line.startswith("| `"): continue
        parts = [x.strip() for x in line.strip("|").split("|")]
        if len(parts) < 11: continue
        arm = ARMBY.get(parts[0])
        if not arm: continue
        seed = int(parts[1].replace("*",""))
        gate, lr = parts[2], float(parts[3]); head, rank = parts[4], parts[5]
        mse, mae = float(parts[6]), float(parts[7])
        # resolve the run
        r = records.get((arm, ds, h, seed))
        if not r:
            problems.append(f"{title} {arm} seed {seed}: no such cell in results.csv"); continue
        cfg = config_of(r["run_dir"])
        if not cfg:
            problems.append(f"{title} {arm} seed {seed}: no config.json at {r['run_dir']}"); continue
        hp = cfg["hyperparams"]; checked += 1
        # parameters
        if gate != "—" and abs(float(hp["weak_period_residual_gate_init"]) - float(gate)) > 1e-9:
            problems.append(f"{title} {arm} seed {seed}: gate doc={gate} cfg={hp['weak_period_residual_gate_init']}")
        if abs(float(hp["learning_rate"]) - lr) > 1e-12:
            problems.append(f"{title} {arm} seed {seed}: lr doc={lr} cfg={hp['learning_rate']}")
        want_head = hp.get("weak_period_residual_head_type")
        if not head.startswith("—"):
            got = "shared" if head.startswith("shared") else ("pooled_lowrank" if head.startswith("pooled") else head)
            if want_head != got:
                problems.append(f"{title} {arm} seed {seed}: head doc={head} cfg={want_head}")
        if rank != "—" and int(hp.get("weak_period_residual_rank", -1)) != int(rank):
            problems.append(f"{title} {arm} seed {seed}: rank doc={rank} cfg={hp.get('weak_period_residual_rank')}")
        # metrics: from the run itself, or from the stage-B record it was reused from
        m = metrics_of(r["run_dir"], r.get("source",""))
        if m is None:
            if r.get("test_mse","").strip():
                m = (float(r["test_mse"]), float(r["test_mae"]))
            else:
                problems.append(f"{title} {arm} seed {seed}: no test metrics found"); continue
        if abs(m[0]-mse) > 5e-7 or abs(m[1]-mae) > 5e-7:
            problems.append(f"{title} {arm} seed {seed}: metrics doc={mse:.6f}/{mae:.6f} run={m[0]:.6f}/{m[1]:.6f}")
        # the claimed seed must be the row's optimum
        cand=[]
        for s in (2021,2022,2023):
            rr=records.get((arm,ds,h,s))
            if not rr: continue
            mm=metrics_of(rr["run_dir"], rr.get("source","")) or ((float(rr["test_mse"]),float(rr["test_mae"])) if rr.get("test_mse","").strip() else None)
            if mm: cand.append((s,mm[0],mm[1]))
        if cand:
            bs=min(cand,key=lambda c:max(c[1]/gm-1,c[2]/ga-1))[0]
            if bs!=seed:
                problems.append(f"{title} {arm}: doc's best seed={seed} but optimum={bs}")
print(f"rows checked: {checked}")
print(f"PROBLEMS: {len(problems)}")
for p in problems[:12]: print("  -", p)
