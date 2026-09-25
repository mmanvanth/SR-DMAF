"""Seed variability, paired bootstrap significance (SR-DMAF vs B1), and sensitivity to side effects of graduated actions."""
import numpy as np, pandas as pd, json
src = open("exp_sensitivity.py").read()
exec(src[:src.index("BASE = dict(")])          # functions, PREP (stored predictions), run()
BASE = dict(n_dc=4, n_db=8, n_gw=12, n_ws=36, replicas=1, w_lo=0.4, w_hi=0.95, db_dc=2, gw_db=2, ws_gw=(1, 2))
out = {}

# ---------- side effects: graduated actions weaken the target's outgoing edges by a factor lam ----------
def cascade_partial(G, theta, lam):
    nodes = list(G.nodes); idx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
    W = np.zeros((N, N))
    for u, v, d in G.edges(data=True): W[idx[u], idx[v]] = d["w"]
    gamma = theta * W.sum(0); has_prov = W.sum(0) > 0
    res = np.zeros(N)
    for k in range(N):
        Wk = W.copy(); Wk[k, :] *= (1 - lam)            # host k stays up; its services are degraded
        s = np.ones(N)
        for _ in range(60):
            new = np.where(has_prov, (Wk.T @ s - gamma >= -1e-12).astype(float), 1.0)
            new = np.minimum(new, s)
            if np.array_equal(new, s): break
            s = new
        res[k] = (np.delete(s, k) == 0).mean()
    return res

facts = topo_facts(BASE, 0.5)
graphs = [build(t, **BASE) for t in range(N_TOPO)]
side = {lam: [cascade_partial(G, 0.5, lam) for G in graphs] for lam in (0.25, 0.5, 0.75)}

def per_alert(seed, lam=None):
    p, rel, cal, yt = PREP[seed]; rng = np.random.default_rng(100 + seed)
    ti = rng.integers(0, N_TOPO, len(yt)); hi = rng.integers(0, facts[0]["n"], len(yt))
    R = np.array([facts[t]["R"][h] for t, h in zip(ti, hi)]); casc = np.array([facts[t]["casc"][h] for t, h in zip(ti, hi)])
    g = np.zeros(len(yt)) if lam is None else np.array([side[lam][t][h] for t, h in zip(ti, hi)])
    o = {}
    for pol in ("B1", "B3", "SR-DMAF"):
        a = actions(pol, p, rel, R, cal)
        cfr = np.where(a == "T4", casc, np.where(a == "G", g, 0.0))
        act = np.isin(a, ["T4", "G", "T1"])
        o[pol] = {"benign": act & (yt == 0), "cfr": cfr, "isben": yt == 0}
    return o

# ---------- paired bootstrap over alerts (pooled over 5 seeds) ----------
alerts = [per_alert(s) for s in PREP]
b1_b = np.concatenate([a["B1"]["benign"][a["B1"]["isben"]] for a in alerts]).astype(float)
sr_b = np.concatenate([a["SR-DMAF"]["benign"][a["SR-DMAF"]["isben"]] for a in alerts]).astype(float)
b1_c = np.concatenate([a["B1"]["cfr"] for a in alerts]); sr_c = np.concatenate([a["SR-DMAF"]["cfr"] for a in alerts])
rng = np.random.default_rng(0); B = 10000
def boot(x, y):
    n = len(x); d = np.empty(B); r = np.empty(B)
    for i in range(B):
        j = rng.integers(0, n, n); mx, my = x[j].mean(), y[j].mean(); d[i] = mx - my; r[i] = 1 - my / mx
    return {"diff": float(x.mean() - y.mean()), "diff_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "red": float(1 - y.mean() / x.mean()), "red_ci": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
            "p_one_sided": float((d <= 0).mean()), "n": int(n)}
out["boot_benign"] = boot(b1_b, sr_b); out["boot_cfr"] = boot(b1_c, sr_c)
out["per_seed_benign"] = {"B1": [float(a["B1"]["benign"][a["B1"]["isben"]].mean()) for a in alerts],
                          "SR": [float(a["SR-DMAF"]["benign"][a["SR-DMAF"]["isben"]].mean()) for a in alerts]}
out["per_seed_cfr"] = {"B1": [float(a["B1"]["cfr"].mean()) for a in alerts], "SR": [float(a["SR-DMAF"]["cfr"].mean()) for a in alerts]}

# ---------- side-effect sensitivity ----------
out["side"] = {}
for lam in (0.25, 0.5, 0.75):
    al = [per_alert(s, lam) for s in PREP]
    out["side"][lam] = {pol: {"CFR_mean": float(np.mean([a[pol]["cfr"].mean() for a in al])),
                              "CFR_std": float(np.std([a[pol]["cfr"].mean() for a in al], ddof=1))} for pol in ("B1", "B3", "SR-DMAF")}
    out["side"][lam]["share_G_hosts_causing_cascade"] = float(np.mean(np.concatenate(side[lam]) > 0))

# ---------- seed std for Table 5 (from stored per-seed results) ----------
P = pd.read_csv("results_policies.csv"); P = P[P.eff_G == 0.95]
cols = ["containment", "deferred_mal", "false_action", "CFR", "P_cascade", "SAR"]
sd = P.groupby(["scenario", "policy"])[cols].std(ddof=1) * 100
out["table5_max_std"] = float(sd.values.max()); out["table5_std_B1_SR"] = sd.loc["uniform host"].loc[["B1 naive detect-isolate", "SR-DMAF (full)"]].round(3).to_dict()

# ---------- seed std for Tables 7 and 8 ----------
E = []
for eG in (0.70, 0.80, 0.90, 0.95, 1.00):
    for eT1 in (0.60, 0.80, 0.90):
        df, _ = run(BASE, 0.5, eG, eT1, facts=facts)
        E.append(df[(df.scen == "uniform") & df.pol.isin(["B3", "SR-DMAF"])].groupby("pol")["cont"].std(ddof=1).max() * 100)
out["table7_max_std"] = float(max(E))
CONF = [({}, 0.5), ({}, 0.3), ({}, 0.7), (dict(n_dc=2, n_db=4, n_gw=6, n_ws=18), 0.5), (dict(n_dc=8, n_db=16, n_gw=24, n_ws=72), 0.5),
        (dict(n_dc=2, n_db=4), 0.5), (dict(n_dc=6, n_db=12), 0.5), (dict(replicas=0), 0.5), (dict(replicas=2), 0.5),
        (dict(gw_db=1, ws_gw=(1, 1), db_dc=1), 0.5), (dict(gw_db=3, ws_gw=(2, 3), db_dc=3), 0.5), (dict(w_lo=0.2), 0.5), (dict(w_lo=0.6), 0.5)]
mx = 0.0; srmax = 0.0
for over, th in CONF:
    df, _ = run({**BASE, **over}, th)
    s = df[df.pol.isin(["B1", "SR-DMAF"])].groupby(["scen", "pol"])["CFR"].std(ddof=1) * 100
    mx = max(mx, s.max()); srmax = max(srmax, s.xs("SR-DMAF", level="pol").max())
out["table8_max_std"] = float(mx); out["table8_max_std_SR"] = float(srmax)
json.dump(out, open("results_stats_side.json", "w"), indent=1)
print(json.dumps(out, indent=1))
