import pickle, time, json, numpy as np, pandas as pd, networkx as nx
from sklearn.metrics import roc_curve
from core import build_topology, cascade_fraction, blast_radius

S = pickle.load(open("store.pkl", "rb"))
store, y = S["store"], S["y"]
THETA, TAU_D, EPS = 0.5, 0.40, 0.10
N_TOPO = 20

# ---- precompute topology facts: tier, blast radius, hard-isolation cascade (theta = 0.5)
topo = []
for t in range(N_TOPO):
    G = build_topology(t)
    btw = nx.betweenness_centrality(G, normalized=True)
    nodes = list(G.nodes)
    topo.append({"tier": np.array([G.nodes[v]["tier"] for v in nodes]),
                 "R": np.array([blast_radius(G, v, btw) for v in nodes]),
                 "casc": np.array([cascade_fraction(G, v, THETA) for v in nodes])})
N_HOST = len(topo[0]["tier"])


def jsd(P, Q):
    M = 0.5 * (P + Q)
    kl = lambda A, B: np.sum(A * (np.log2(A + 1e-12) - np.log2(B + 1e-12)), 1)
    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def youden(yb, s):
    f, t, th = roc_curve(yb, s)
    j = np.argmax(t - f)
    return th[j], t[j] - f[j]


def pillars(A, split, cal):
    """Return per-sample pillar scores for the split ('va' or 'te')."""
    MC = A["fused"]["MC" + split]                      # T x N x C
    pm_t = 1 - MC[:, :, 0]
    mu, var = pm_t.mean(0), pm_t.var(0)
    E = np.zeros_like(mu)
    for m in ("static", "behaviour", "network"):
        pm = 1 - A[m]["P" + split][:, 0]
        E += cal["w"][m] * (pm > cal["t"][m])
    E /= sum(cal["w"].values())
    cons = 1 - jsd(A["static"]["P" + split], A["behaviour"]["P" + split])
    return {"mu": mu, "var": var, "E": E, "cons": cons}


def calibrate(A, yva):
    yb = (yva != 0).astype(int)
    cal = {"t": {}, "w": {}}
    for m in ("static", "behaviour", "network"):
        th, J = youden(yb, 1 - A[m]["Pva"][:, 0])
        cal["t"][m], cal["w"][m] = th, max(J, 0)
    p = pillars(A, "va", cal)
    ok = (p["mu"] > 0.5) & (yb == 1)                    # correct malicious calls on validation
    cal["tau_u"] = float(np.quantile(p["var"][ok], 0.95))
    cal["tau_c"] = float(np.quantile(p["cons"][ok], 0.05))
    cal["tau_e"] = 0.5
    return cal


def reliability(p, cal, use=(1, 1, 1)):
    pass_e = (p["E"] >= cal["tau_e"]) | (not use[0])
    pass_u = (p["var"] <= cal["tau_u"]) | (not use[1])
    pass_c = (p["cons"] >= cal["tau_c"]) | (not use[2])
    d = []
    if use[0]: d.append((p["E"] - cal["tau_e"]) / (1 - cal["tau_e"]))
    if use[1]: d.append((cal["tau_u"] - p["var"]) / cal["tau_u"])
    if use[2]: d.append((p["cons"] - cal["tau_c"]) / (1 - cal["tau_c"]))
    dmin = np.min(np.stack(d), 0) if d else np.ones_like(p["mu"])
    allp = pass_e & pass_u & pass_c
    return np.where(~allp, "LOW", np.where(dmin < EPS, "MEDIUM", "HIGH"))


def policy_actions(name, p, rel, R, cal):
    pred = p["mu"] > 0.5
    a = np.full(len(pred), "none", dtype=object)
    if name == "B1 naive detect-isolate":
        a[pred] = "T4"
    elif name == "B2 uncertainty triage":
        a[pred & (p["var"] <= cal["tau_u"])] = "T4"
        a[pred & (p["var"] > cal["tau_u"])] = "defer"
    elif name == "B3 graduated only":
        a[pred] = "G"
    else:
        grad = "no-graduated" not in name
        p4 = "w/o P4" not in name
        hi = pred & (rel == "HIGH")
        a[pred & (rel == "LOW")] = "defer"
        a[pred & (rel == "MEDIUM")] = "T1" if grad else "defer"
        safe = (R <= TAU_D) | (not p4)
        a[hi & safe] = "T4"
        a[hi & ~safe] = "G" if grad else "defer"
    return a


def outcomes(a, ytrue, tier, casc, eff):
    mal = ytrue != 0
    e = np.select([a == "T4", a == "G", a == "T1"], [eff["T4"], eff["G"], eff["T1"]], 0.0)
    act = np.isin(a, ["T4", "G", "T1"])
    hard = a == "T4"
    cfr = np.where(hard, casc, 0.0)                     # collateral: other hosts collapsed
    n_benign = N_HOST - mal                             # benign hosts in the incident
    lost = cfr * (N_HOST - 1) + (hard & ~mal)           # collapsed hosts + a benign target cut off
    lost = np.where(mal & hard, cfr * (N_HOST - 1), lost)
    return {"containment": e[mal].mean(), "deferred_mal": (a[mal] == "defer").mean(),
            "false_action": act[~mal].mean(), "FIR": ((act & ~mal).sum() / max(act.sum(), 1)),
            "hard_iso": hard.mean(), "CFR": cfr.mean(), "P_cascade": (cfr > 0).mean(),
            "CFR_per_iso": cfr[hard].mean() if hard.any() else 0.0,
            "SAR": 1 - (lost / n_benign).mean()}


POLICIES = ["B1 naive detect-isolate", "B2 uncertainty triage", "B3 graduated only", "SR-DMAF (full)",
            "SR-DMAF w/o P1", "SR-DMAF w/o P2", "SR-DMAF w/o P3", "SR-DMAF w/o P4", "SR-DMAF no-graduated"]
USE = {"SR-DMAF w/o P1": (0, 1, 1), "SR-DMAF w/o P2": (1, 0, 1), "SR-DMAF w/o P3": (1, 1, 0)}
rows, rel_rows, cal_rows = [], [], []
for seed, D in store.items():
    A, te, va = D["agents"], D["te"], D["va"]
    cal = calibrate(A, y[va])
    cal_rows.append({"seed": seed, **{k: v for k, v in cal.items() if k.startswith("tau")},
                     **{f"w_{m}": v for m, v in cal["w"].items()}})
    p = pillars(A, "te", cal)
    yt = y[te]
    rng = np.random.default_rng(100 + seed)
    rel_full = reliability(p, cal)
    pred = p["mu"] > 0.5
    for lvl in ("HIGH", "MEDIUM", "LOW"):
        m = pred & (rel_full == lvl)
        rel_rows.append({"seed": seed, "level": lvl, "share_of_mal_calls": m.sum() / pred.sum(),
                         "precision": (yt[m] != 0).mean() if m.any() else np.nan})
    for scen in ("uniform host", "Tier-1 host"):
        ti = rng.integers(0, N_TOPO, len(te))
        if scen == "uniform host":
            hi = rng.integers(0, N_HOST, len(te))
        else:
            t1 = np.where(topo[0]["tier"] == 1)[0]
            hi = rng.choice(t1, len(te))
        tier = np.array([topo[t]["tier"][h] for t, h in zip(ti, hi)])
        R = np.array([topo[t]["R"][h] for t, h in zip(ti, hi)])
        casc = np.array([topo[t]["casc"][h] for t, h in zip(ti, hi)])
        for eg in (0.80, 0.90, 0.95, 1.00):
            eff = {"T4": 1.0, "G": eg, "T1": 0.80}
            for pol in POLICIES:
                rel = reliability(p, cal, USE.get(pol, (1, 1, 1)))
                a = policy_actions(pol, p, rel, R, cal)
                rows.append({"seed": seed, "scenario": scen, "eff_G": eg, "policy": pol,
                             **outcomes(a, yt, tier, casc, eff)})

# ---- latency of one incident through the full pipeline (numpy, single CPU core)
D = store[0]; A = D["agents"]
import pandas as _pd
from core import load, static_view, behav_view, net_view
df, _ = load("/mnt/user-data/uploads/ransom.csv")
Xv = {"static": static_view(df).values, "behaviour": behav_view(df).values, "network": net_view(df).values}
Xv["fused"] = np.hstack([Xv["static"], Xv["behaviour"], Xv["network"]])
ids = D["te"][:300]
lat = {"agents": [], "mc50": [], "srvl": []}
for i in ids:
    t0 = time.perf_counter()
    P = {m: A[m]["model"].predict_proba(A[m]["scaler"].transform(Xv[m][i:i + 1])) for m in ("static", "behaviour", "network")}
    t1 = time.perf_counter()
    xf = A["fused"]["scaler"].transform(Xv["fused"][i:i + 1])
    mc = A["fused"]["model"].mc_proba(np.repeat(xf, 50, 0), 1)[0]   # 50 passes batched
    t2 = time.perf_counter()
    _ = 1 - jsd(P["static"], P["behaviour"]); _ = mc[:, 0].var(); _ = topo[0]["R"][5] <= TAU_D
    t3 = time.perf_counter()
    lat["agents"].append(t1 - t0); lat["mc50"].append(t2 - t1); lat["srvl"].append(t3 - t2)
latency = {k: float(np.median(v) * 1000) for k, v in lat.items()}
latency["total_ms"] = sum(latency.values())

R = pd.DataFrame(rows); R.to_csv("results_policies.csv", index=False)
pd.DataFrame(rel_rows).to_csv("results_reliability.csv", index=False)
pd.DataFrame(cal_rows).to_csv("results_calibration.csv", index=False)
json.dump(latency, open("results_latency.json", "w"), indent=1)
print(latency)
print(pd.DataFrame(cal_rows).mean().round(4))
print(pd.DataFrame(rel_rows).groupby("level")[["share_of_mal_calls", "precision"]].agg(["mean", "std"]).round(4))
pd.set_option("display.width", 250)
for scen in ("uniform host", "Tier-1 host"):
    sub = R[(R.scenario == scen) & (R.eff_G == 0.95)].groupby("policy", sort=False).mean(numeric_only=True)
    print("\n==", scen, "eff_G=0.95\n", sub.drop(columns=["seed", "eff_G"]).round(4))
