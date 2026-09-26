"""Sensitivity of SR-DMAF to the simulated network and to the containment-success assumptions.
Uses the stored real test-set predictions (store.pkl from exp_classify.py); agents are NOT retrained.
"""
import pickle, json, numpy as np, pandas as pd, networkx as nx
from sklearn.metrics import roc_curve, roc_auc_score

S = pickle.load(open("store.pkl", "rb"))
store, y = S["store"], S["y"]
TAU_D, EPS, N_TOPO = 0.40, 0.10, 20
TW = {1: 1.0, 2: 0.6, 3: 0.2}


# ---------------- parameterised topology (base values reproduce core.build_topology) ----------------
def build(seed, n_dc=4, n_db=8, n_gw=12, n_ws=36, replicas=1, w_lo=0.4, w_hi=0.95,
          db_dc=2, gw_db=2, ws_gw=(1, 2), dc_rep_p=0.7):
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    dcs = [f"DC{i}" for i in range(n_dc)]; dbs = [f"DB{i}" for i in range(n_db)]
    gws = [f"GW{i}" for i in range(n_gw)]; ws = [f"PC{i}" for i in range(n_ws)]
    for n in dcs + dbs: G.add_node(n, tier=1)
    for n in gws: G.add_node(n, tier=2)
    for n in ws: G.add_node(n, tier=3)
    w = lambda: float(np.round(rng.uniform(w_lo, w_hi), 2))
    for d in dcs:
        for e in dcs:
            if d != e and rng.random() < dc_rep_p: G.add_edge(d, e, w=w())
    for i, b in enumerate(dbs):
        for d in rng.choice(dcs, min(db_dc, n_dc), replace=False): G.add_edge(d, b, w=w())
        for r in range(1, replicas + 1):
            G.add_edge(dbs[(i + r) % n_db], b, w=w())
    for g in gws:
        for b in rng.choice(dbs, min(gw_db, n_db), replace=False): G.add_edge(b, g, w=w())
        G.add_edge(rng.choice(dcs), g, w=w())
    for p in ws:
        k = int(rng.integers(ws_gw[0], ws_gw[1] + 1))
        for g in rng.choice(gws, min(k, n_gw), replace=False): G.add_edge(g, p, w=w())
        G.add_edge(rng.choice(dcs), p, w=w())
    return G


def cascade_all(G, theta):
    nodes = list(G.nodes); idx = {n: i for i, n in enumerate(nodes)}; N = len(nodes)
    W = np.zeros((N, N))
    for u, v, d in G.edges(data=True): W[idx[u], idx[v]] = d["w"]
    gamma = theta * W.sum(0); has_prov = W.sum(0) > 0
    out = np.zeros(N)
    for k in range(N):
        a = np.zeros(N); a[k] = 1; s = np.ones(N); s[k] = 0
        for _ in range(60):
            new = np.where(has_prov, (W.T @ s - gamma >= -1e-12).astype(float), 1.0) * (1 - a)
            new = np.minimum(new, s)
            if np.array_equal(new, s): break
            s = new
        out[k] = (np.delete(s, k) == 0).mean()
    return out


def topo_facts(cfg, theta):
    facts = []
    for t in range(N_TOPO):
        G = build(t, **cfg)
        btw = nx.betweenness_centrality(G, normalized=True)
        nodes = list(G.nodes); n = len(nodes)
        R = np.array([0.45 * len(nx.descendants(G, v)) / (n - 1) + 0.35 * btw[v] + 0.20 * TW[G.nodes[v]["tier"]]
                      for v in nodes])
        facts.append({"tier": np.array([G.nodes[v]["tier"] for v in nodes]), "R": R,
                      "casc": cascade_all(G, theta), "n": n})
    return facts


# ---------------- SRVL (identical to exp_srvl.py) ----------------
def jsd(P, Q):
    M = 0.5 * (P + Q)
    kl = lambda A, B: np.sum(A * (np.log2(A + 1e-12) - np.log2(B + 1e-12)), 1)
    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


def youden(yb, s):
    f, t, th = roc_curve(yb, s); j = np.argmax(t - f); return th[j], t[j] - f[j]


def pillars(A, split, cal):
    MC = A["fused"]["MC" + split]; pm_t = 1 - MC[:, :, 0]
    mu, var = pm_t.mean(0), pm_t.var(0)
    E = np.zeros_like(mu)
    for m in ("static", "behaviour", "network"):
        E += cal["w"][m] * ((1 - A[m]["P" + split][:, 0]) > cal["t"][m])
    E /= sum(cal["w"].values())
    return {"mu": mu, "var": var, "E": E, "cons": 1 - jsd(A["static"]["P" + split], A["behaviour"]["P" + split])}


def calibrate(A, yva):
    yb = (yva != 0).astype(int); cal = {"t": {}, "w": {}}
    for m in ("static", "behaviour", "network"):
        th, J = youden(yb, 1 - A[m]["Pva"][:, 0]); cal["t"][m], cal["w"][m] = th, max(J, 0)
    p = pillars(A, "va", cal); ok = (p["mu"] > 0.5) & (yb == 1)
    cal["tau_u"] = float(np.quantile(p["var"][ok], 0.95)); cal["tau_c"] = float(np.quantile(p["cons"][ok], 0.05))
    cal["tau_e"] = 0.5
    return cal


def reliability(p, cal):
    pe, pu, pc = p["E"] >= cal["tau_e"], p["var"] <= cal["tau_u"], p["cons"] >= cal["tau_c"]
    d = np.min(np.stack([(p["E"] - cal["tau_e"]) / (1 - cal["tau_e"]), (cal["tau_u"] - p["var"]) / cal["tau_u"],
                         (p["cons"] - cal["tau_c"]) / (1 - cal["tau_c"])]), 0)
    return np.where(~(pe & pu & pc), "LOW", np.where(d < EPS, "MEDIUM", "HIGH"))


def actions(pol, p, rel, R, cal):
    pred = p["mu"] > 0.5; a = np.full(len(pred), "none", dtype=object)
    if pol == "B1": a[pred] = "T4"
    elif pol == "B2":
        a[pred & (p["var"] <= cal["tau_u"])] = "T4"; a[pred & (p["var"] > cal["tau_u"])] = "defer"
    elif pol == "B3": a[pred] = "G"
    else:
        hi = pred & (rel == "HIGH"); a[pred & (rel == "LOW")] = "defer"; a[pred & (rel == "MEDIUM")] = "T1"
        a[hi & (R <= TAU_D)] = "T4"; a[hi & (R > TAU_D)] = "G"
    return a


def outcomes(a, yt, casc, n, eG, eT1):
    mal = yt != 0
    e = np.select([a == "T4", a == "G", a == "T1"], [1.0, eG, eT1], 0.0)
    act = np.isin(a, ["T4", "G", "T1"]); hard = a == "T4"
    cfr = np.where(hard, casc, 0.0)
    lost = np.where(mal, cfr * (n - 1), cfr * (n - 1) + hard)
    return {"cont": e[mal].mean(), "defer": (a[mal] == "defer").mean(), "benign": act[~mal].mean(),
            "CFR": cfr.mean(), "SAR": 1 - (lost / (n - mal)).mean()}


# precompute pillars once per seed
PREP = {}
for seed, D in store.items():
    A = D["agents"]; cal = calibrate(A, y[D["va"]]); p = pillars(A, "te", cal)
    PREP[seed] = (p, reliability(p, cal), cal, y[D["te"]])


def run(cfg, theta=0.5, eG=0.95, eT1=0.80, facts=None):
    facts = facts or topo_facts(cfg, theta)
    rows = []
    for seed, (p, rel, cal, yt) in PREP.items():
        rng = np.random.default_rng(100 + seed)
        for scen in ("uniform", "tier1"):
            ti = rng.integers(0, N_TOPO, len(yt))
            if scen == "uniform":
                hi = rng.integers(0, facts[0]["n"], len(yt))
            else:
                hi = rng.choice(np.where(facts[0]["tier"] == 1)[0], len(yt))
            R = np.array([facts[t]["R"][h] for t, h in zip(ti, hi)])
            casc = np.array([facts[t]["casc"][h] for t, h in zip(ti, hi)])
            n = facts[0]["n"]
            for pol in ("B1", "B2", "B3", "SR-DMAF"):
                a = actions(pol, p, rel, R, cal)
                rows.append({"seed": seed, "scen": scen, "pol": pol, **outcomes(a, yt, casc, n, eG, eT1)})
    # estimator quality on this network family
    Rall = np.concatenate([f["R"] for f in facts]); Call = np.concatenate([f["casc"] for f in facts])
    harm = Call > 0.05
    auc = roc_auc_score(harm, Rall) if 0 < harm.sum() < len(harm) else np.nan
    return pd.DataFrame(rows), {"auc": auc, "harm_share": harm.mean(), "caught": (Rall[harm] > TAU_D).mean() if harm.any() else np.nan}


BASE = dict(n_dc=4, n_db=8, n_gw=12, n_ws=36, replicas=1, w_lo=0.4, w_hi=0.95, db_dc=2, gw_db=2, ws_gw=(1, 2))
CONFIGS = [
    ("Base (60 hosts, θ=0.5)", {}, 0.5),
    ("θ = 0.3", {}, 0.3),
    ("θ = 0.7", {}, 0.7),
    ("30 hosts", dict(n_dc=2, n_db=4, n_gw=6, n_ws=18), 0.5),
    ("120 hosts", dict(n_dc=8, n_db=16, n_gw=24, n_ws=72), 0.5),
    ("Tier 1: 2 DC + 4 DB", dict(n_dc=2, n_db=4), 0.5),
    ("Tier 1: 6 DC + 12 DB", dict(n_dc=6, n_db=12), 0.5),
    ("No DB replicas", dict(replicas=0), 0.5),
    ("2 DB replicas", dict(replicas=2), 0.5),
    ("Sparse dependencies", dict(gw_db=1, ws_gw=(1, 1), db_dc=1), 0.5),
    ("Dense dependencies", dict(gw_db=3, ws_gw=(2, 3), db_dc=3), 0.5),
    ("Weights U[0.2, 0.95]", dict(w_lo=0.2), 0.5),
    ("Weights U[0.6, 0.95]", dict(w_lo=0.6), 0.5),
]
res = []
base_facts = None
for name, over, theta in CONFIGS:
    cfg = {**BASE, **over}
    facts = topo_facts(cfg, theta)
    if name.startswith("Base"): base_facts = facts
    df, est = run(cfg, theta, facts=facts)
    m = df.groupby(["scen", "pol"])[["cont", "benign", "CFR", "SAR"]].mean()
    res.append({"config": name,
                "B1_CFR": m.loc[("uniform", "B1"), "CFR"], "SR_CFR": m.loc[("uniform", "SR-DMAF"), "CFR"],
                "B1_CFR_T1": m.loc[("tier1", "B1"), "CFR"], "SR_CFR_T1": m.loc[("tier1", "SR-DMAF"), "CFR"],
                "B1_SAR": m.loc[("uniform", "B1"), "SAR"], "SR_SAR": m.loc[("uniform", "SR-DMAF"), "SAR"],
                "SR_cont": m.loc[("uniform", "SR-DMAF"), "cont"],
                "B1_benign": m.loc[("uniform", "B1"), "benign"], "SR_benign": m.loc[("uniform", "SR-DMAF"), "benign"],
                "AUC": est["auc"], "harm_share": est["harm_share"], "caught": est["caught"]})
    print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in res[-1].items()}, flush=True)
pd.DataFrame(res).to_csv("results_sensitivity_topology.csv", index=False)

# ---------------- containment-success sweep on the base network ----------------
eff = []
for eG in (0.70, 0.80, 0.90, 0.95, 1.00):
    for eT1 in (0.60, 0.80, 0.90):
        df, _ = run(BASE, 0.5, eG, eT1, facts=base_facts)
        m = df[df.scen == "uniform"].groupby("pol")[["cont", "benign", "CFR", "SAR"]].mean()
        for pol in m.index:
            eff.append({"eG": eG, "eT1": eT1, "pol": pol, **m.loc[pol].to_dict()})
E = pd.DataFrame(eff); E.to_csv("results_sensitivity_efficacy.csv", index=False)
print(E[E.pol.isin(["B3", "SR-DMAF"])].pivot_table(index=["eG", "eT1"], columns="pol", values=["cont", "benign", "CFR"]).round(4))
print("benign/CFR ranges across all efficacy settings:")
print(E.groupby("pol")[["benign", "CFR"]].agg(["min", "max"]).round(5))
