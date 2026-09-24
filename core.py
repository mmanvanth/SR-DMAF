"""SR-DMAF reproducible core: feature views, numpy MC-dropout MLP, cascade model.
Dataset: Ransomware Dataset 2024 (Zenodo 10.5281/zenodo.13890887), file ransom.csv.
"""
import ast, numpy as np, pandas as pd, networkx as nx

LABELS = ["Class", "Category", "Family"]
HASHES = ["md5", "sha1"]
BEHAV = ["registry_read", "registry_write", "registry_delete", "registry_total",
         "processes_malicious", "processes_suspicious", "processes_monitored", "total_procsses",
         "files_malicious", "files_suspicious", "files_text", "files_unknown", "dlls_calls", "apis"]
NET = ["network_dns", "network_http", "network_connections"]   # network_threats is constant 0 -> dropped
LIST_COLS = ["DllCharacteristics", "text_Characteristics", "rdata_Characteristics"]
CAT_COLS = ["PEType", "MachineType", "Magic", "Subsystem"]
FLOAT_COLS = ["OperatingSystemVersion", "ImageVersion"]


def load(path):
    df = pd.read_csv(path)
    # remove hashes whose rows carry conflicting labels (label noise)
    g = df.groupby("md5").Family.nunique()
    bad = set(g[g > 1].index)
    info = {"rows_raw": len(df), "unique_md5": df.md5.nunique(), "conflict_md5": len(bad),
            "rows_conflict": int(df.md5.isin(bad).sum())}
    df = df[~df.md5.isin(bad)].reset_index(drop=True)
    info["rows_clean"] = len(df)
    info["unique_md5_clean"] = df.md5.nunique()
    return df, info


def _hex(s):
    s = str(s).split(" ")[0]
    try:
        return int(s, 16) if s.lower().startswith("0x") else float(s)
    except ValueError:
        return np.nan


def static_view(df):
    out = {}
    skip = set(LABELS + HASHES + BEHAV + NET + LIST_COLS + CAT_COLS + FLOAT_COLS +
               ["network_threats", "file_extension", "magic_number", "AddressOfEntryPoint"])
    for c in df.columns:
        if c in skip:
            continue
        v = df[c].map(_hex).astype(float)
        out[c] = np.log1p(np.abs(v.fillna(0)))
    ep = df["AddressOfEntryPoint"].astype(str)
    out["AddressOfEntryPoint"] = np.log1p(ep.map(_hex).fillna(0).abs())
    sec = ep.str.extract(r"Section: ([^)]*)")[0].fillna("none")
    out["entry_in_text"] = (sec == ".text").astype(float)
    out["entry_no_section"] = (sec == "none").astype(float)
    out["checksum_zero"] = (df["Checksum"].map(_hex) == 0).astype(float)
    for c in FLOAT_COLS:
        out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    X = pd.DataFrame(out)
    for c in CAT_COLS:
        X = X.join(pd.get_dummies(df[c].astype(str), prefix=c).astype(float))
    for c in LIST_COLS:
        flags = df[c].map(lambda s: ast.literal_eval(s) if isinstance(s, str) and s.startswith("[") else [])
        allf = sorted({f for L in flags for f in L})
        for f in allf:
            X[f"{c}:{f.replace('IMAGE_', '')}"] = flags.map(lambda L: float(f in L))
    return X


def behav_view(df):
    return np.log1p(df[BEHAV].astype(float))


def net_view(df):
    return np.log1p(df[NET].astype(float))


class MCDropoutMLP:
    """Two-hidden-layer MLP with inverted dropout, Adam, class weights; dropout kept on for MC inference."""

    def __init__(self, d_in, n_cls, hidden=(128, 64), p=0.2, lr=1e-3, wd=1e-4, seed=0):
        self.rng = np.random.default_rng(seed)
        dims = [d_in, *hidden, n_cls]
        self.W = [self.rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(dims[:-1], dims[1:])]
        self.b = [np.zeros(b) for b in dims[1:]]
        self.p, self.lr, self.wd = p, lr, wd
        self.m = [np.zeros_like(w) for w in self.W + self.b]
        self.v = [np.zeros_like(w) for w in self.W + self.b]
        self.t = 0

    def _forward(self, X, drop):
        acts, masks = [X], []
        h = X
        for i in range(len(self.W) - 1):
            h = np.maximum(h @ self.W[i] + self.b[i], 0)
            if drop:
                mk = (self.rng.random(h.shape) > self.p) / (1 - self.p)
                h = h * mk
            else:
                mk = None
            masks.append(mk)
            acts.append(h)
        z = h @ self.W[-1] + self.b[-1]
        z -= z.max(1, keepdims=True)
        P = np.exp(z)
        P /= P.sum(1, keepdims=True)
        return P, acts, masks

    def _step(self, X, Y1h, cw):
        P, acts, masks = self._forward(X, True)
        w = (Y1h * cw).sum(1, keepdims=True)
        g = (P - Y1h) * w / len(X)
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ g + self.wd * self.W[i]
            gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i].T) * (acts[i] > 0)
                if masks[i - 1] is not None:
                    g = g * masks[i - 1]
        self.t += 1
        params = self.W + self.b
        grads = gW + gb
        for k, (pp, gg) in enumerate(zip(params, grads)):
            self.m[k] = 0.9 * self.m[k] + 0.1 * gg
            self.v[k] = 0.999 * self.v[k] + 0.001 * gg * gg
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            pp -= self.lr * mh / (np.sqrt(vh) + 1e-8)

    def fit(self, X, y, Xv, yv, n_cls, epochs=80, bs=256, patience=10, metric=None):
        Y1h = np.eye(n_cls)[y]
        cnt = np.bincount(y, minlength=n_cls).astype(float)
        cw = np.where(cnt > 0, np.sqrt(cnt.sum() / (n_cls * np.maximum(cnt, 1))), 0.0)
        best, best_state, bad = -1, None, 0
        for ep in range(epochs):
            idx = self.rng.permutation(len(X))
            for s in range(0, len(X), bs):
                j = idx[s:s + bs]
                self._step(X[j], Y1h[j], cw)
            score = metric(yv, self.predict_proba(Xv).argmax(1))
            if score > best + 1e-4:
                best, bad = score, 0
                best_state = ([w.copy() for w in self.W], [b.copy() for b in self.b])
            else:
                bad += 1
                if bad >= patience:
                    break
        self.W, self.b = best_state
        return best

    def predict_proba(self, X):
        return self._forward(X, False)[0]

    def mc_proba(self, X, T=50):
        return np.stack([self._forward(X, True)[0] for _ in range(T)])  # T x N x C


# ---------------- cascade model (Eqs. 1-3, 9) ----------------
TIER_WEIGHT = {1: 1.0, 2: 0.6, 3: 0.2}


def build_topology(seed=7):
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    dcs = [f"DC{i}" for i in range(1, 5)]
    dbs = [f"DB{i}" for i in range(1, 9)]
    gws = [f"GW{i}" for i in range(1, 13)]
    ws = [f"PC-{i:02d}" for i in range(1, 37)]
    for n in dcs + dbs:
        G.add_node(n, tier=1)
    for n in gws:
        G.add_node(n, tier=2)
    for n in ws:
        G.add_node(n, tier=3)
    w = lambda: float(np.round(rng.uniform(0.4, 0.95), 2))
    for d in dcs:
        for e in dcs:
            if d != e and rng.random() < 0.7:
                G.add_edge(d, e, w=w())
    for i, b in enumerate(dbs):
        for d in rng.choice(dcs, 2, replace=False):
            G.add_edge(d, b, w=w())
        G.add_edge(dbs[(i + 1) % 8], b, w=w())
    for g in gws:
        for b in rng.choice(dbs, 2, replace=False):
            G.add_edge(b, g, w=w())
        G.add_edge(rng.choice(dcs), g, w=w())
    for p in ws:
        for g in rng.choice(gws, int(rng.integers(1, 3)), replace=False):
            G.add_edge(g, p, w=w())
        G.add_edge(rng.choice(dcs), p, w=w())
    return G


def cascade_fraction(G, k, theta=0.5, T=30):
    """Fraction of the other hosts that lose service when host k is hard-isolated (Eq. 1 to a fixed point)."""
    nodes = list(G.nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    W = np.zeros((N, N))
    for u, v, d in G.edges(data=True):
        W[idx[u], idx[v]] = d["w"]
    gamma = theta * W.sum(0)
    has_prov = W.sum(0) > 0
    a = np.zeros(N)
    a[idx[k]] = 1
    s = np.ones(N)
    s[idx[k]] = 0
    for _ in range(T):
        new = np.where(has_prov, (W.T @ s - gamma >= -1e-12).astype(float), 1.0) * (1 - a)
        new = np.minimum(new, s)
        if np.array_equal(new, s):
            break
        s = new
    others = [i for i in range(N) if i != idx[k]]
    return float((s[others] == 0).mean())


def blast_radius(G, v, btw):
    n = G.number_of_nodes()
    frac = len(nx.descendants(G, v)) / (n - 1)
    return 0.45 * frac + 0.35 * btw[v] + 0.20 * TIER_WEIGHT[G.nodes[v]["tier"]]
