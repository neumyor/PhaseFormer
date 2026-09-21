"""Verify the FINAL table (final_selection.json) cell by cell against the run dirs.

The authoritative record is final_selection.json, which carries best_seed,
delta and max_epochs.  stage1_winners.json's `seed` field is just the stage-1
search seed (always 2021) and must NOT be read as the winning seed.
"""
import json, csv, glob, os

GOLDEN = {("ETTh1",96):(0.359,0.382),("ETTh1",192):(0.397,0.404),("ETTh1",336):(0.425,0.424),
          ("ETTm1",96):(0.293,0.344),("ETTm1",192):(0.323,0.361),("ETTm1",336):(0.358,0.381),
          ("ETTm1",720):(0.412,0.410),("Electricity",96):(0.129,0.221)}
PO = {("ETTh1",96):(0.361402,0.386687),("ETTh1",192):(0.404664,0.410919),("ETTh1",336):(0.441922,0.434654),
      ("ETTm1",96):(0.302441,0.351168),("ETTm1",192):(0.330420,0.363285),("ETTm1",336):(0.359302,0.381157),
      ("ETTm1",720):(0.415068,0.412775),("Electricity",96):(0.130440,0.222771)}
ROOT = "research_runs/phaseformer_L_golden_search_v1"
final = json.load(open(f"{ROOT}/final_selection.json"))
# delta is null in final_selection.json (the reporting gap); take it from the
# corrected table rebuilt from the run dirs.
fixed = {r["dataset"]+r["horizon"]: r for r in csv.DictReader(open(f"{ROOT}/final_with_delta.csv"))}

def dirname(r, seed, delta):
    # the table's head label is 'shared' or 'pooled_r<k>'; the run dir uses
    # 'dense' or 'r<k>'
    h = r["head"]
    head = "dense" if h == "shared" else h.replace("pooled_", "")
    ltag = "" if r["loss"] == "huber" else "_" + r["loss"]
    etag = "" if int(r.get("max_epochs") or 30) == 30 else "_e" + str(int(r["max_epochs"]))
    dtag = "" if not delta or delta == "None" else "_d" + str(delta)
    return f'{r["dataset"]}-h{r["horizon"]}_s{seed}_g{r["gate"]}_lr{r["lr"]}_{head}{ltag}{etag}{dtag}'

problems, rows = [], []
for r in final["results"]:
    key = (r["dataset"], r["horizon"]); gm, ga = GOLDEN[key]; pm, pa = PO[key]
    fx = fixed.get(r["dataset"]+str(r["horizon"]), {})
    delta = fx.get("delta")
    per = {}
    for seed in (2021, 2022, 2023):
        d = os.path.join(ROOT, "runs", dirname(r, seed, delta))
        cf, mf = glob.glob(d+"/runs/*/config.json"), glob.glob(d+"/runs/*/metrics.csv")
        if not cf or not mf:
            problems.append(f"{key} seed {seed}: no artifacts under {os.path.basename(d)}"); continue
        cfg = json.load(open(cf[0])); hp = cfg["hyperparams"]; m = next(csv.DictReader(open(mf[0])))
        if not m.get("test_mse","").strip():
            problems.append(f"{key} seed {seed}: metrics lacks test values"); continue
        # config must match the claimed winner
        for nm, got, want in (("lr", float(hp["learning_rate"]), float(r["lr"])),
                              ("gate", float(hp["weak_period_residual_gate_init"]), float(r["gate"])),
                              ("seed", int(cfg["seed"]), seed),
                              ("epochs", int(cfg["max_epochs"]), int(r.get("max_epochs") or 30))):
            if got != want: problems.append(f"{key} seed {seed}: {nm} cfg={got} claim={want}")
        if delta and delta != "None":
            if float(hp.get("huber_delta", 1.0)) != float(delta):
                problems.append(f"{key} seed {seed}: delta cfg={hp.get('huber_delta')} claim={delta}")
        per[seed] = (float(m["test_mse"]), float(m["test_mae"]))
    if not per: continue
    bs = min(per, key=lambda s: max(per[s][0]/gm-1, per[s][1]/ga-1))
    mse, mae = per[bs]
    nwin = sum(1 for s in per if per[s][0] < gm and per[s][1] < ga)
    rows.append(dict(setting=f"{key[0]}-{key[1]}", gate=r["gate"], lr=r["lr"], loss=r["loss"],
                     head=r["head"], delta=delta, epochs=int(r.get("max_epochs") or 30),
                     seed=bs, mse=mse, mae=mae, dm=100*(mse/gm-1), da=100*(mae/ga-1),
                     dpo_m=100*(mse/pm-1), dpo_a=100*(mae/pa-1),
                     win=(mse < gm and mae < ga), nwin=nwin, nseed=len(per),
                     reported_seed=r["best_seed"], reported_mse=r["best_test_mse"],
                     reported_mae=r["best_test_mae"]))
    if bs != r["best_seed"]:
        problems.append(f"{key}: final says best_seed={r['best_seed']} but recomputed={bs}")
    if abs(mse-r["best_test_mse"])>5e-7 or abs(mae-r["best_test_mae"])>5e-7:
        problems.append(f"{key}: final says {r['best_test_mse']:.6f}/{r['best_test_mae']:.6f} "
                        f"but run dirs give {mse:.6f}/{mae:.6f}")
    if bool(r["beats_golden_both"]) != (mse < gm and mae < ga):
        problems.append(f"{key}: final's beats_golden_both={r['beats_golden_both']} disagrees")

print(f"{'setting':14s} {'config':34s} {'seed':>5s} {'MSE':>9s} {'MAE':>9s} {'vsG':>7s} {'vsPO':>7s} {'W':>2s} {'n/3':>4s}")
for x in sorted(rows, key=lambda z: -max(z["dm"], z["da"])):
    cfg = f'g{x["gate"]} lr{x["lr"]} {x["loss"]} {x["head"]} d={x["delta"]} e{x["epochs"]}'
    print(f'{x["setting"]:14s} {cfg[:34]:34s} {x["seed"]:5d} {x["mse"]:9.6f} {x["mae"]:9.6f} '
          f'{x["dm"]:+6.2f}% {x["dpo_m"]:+6.2f}% {"Y" if x["win"] else "n":>2s} {x["nwin"]}/3')
nwin = sum(1 for x in rows if x["win"])
stable = [x["setting"] for x in rows if x["nwin"] == 3]
partial = [x["setting"] for x in rows if 0 < x["nwin"] < 3]
print()
print(f"qualifying: {nwin}/8 | 3/3 seeds stable: {stable} | 1-2/3: {partial}")
print()
print(f"PROBLEMS: {len(problems)}")
for p in problems: print("  -", p)
