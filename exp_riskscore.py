"""Risk-score component ablation, fitted weights and operating metrics at tau_d (base networks, theta=0.5)."""
import numpy as np, networkx as nx, json
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from core import build_topology, cascade_fraction
TW = {1: 1.0, 2: 0.6, 3: 0.2}
rows = []
for t in range(20):
    G = build_topology(t); btw = nx.betweenness_centrality(G, normalized=True); n = G.number_of_nodes()
    for v in G:
        rows.append((t, len(nx.descendants(G, v)) / (n - 1), btw[v], TW[G.nodes[v]["tier"]], cascade_fraction(G, v, 0.5)))
A = np.array(rows); topo, reach, bt, tier, casc = A.T; y = (casc > 0.05).astype(int)
X = np.c_[reach, bt, tier]; W = np.array([0.45, 0.35, 0.20])
def auc(s): return roc_auc_score(y, s), average_precision_score(y, s)
res = {}
res["Full score (hand-set weights)"] = auc(X @ W)
for i, name in enumerate(["reach", "betweenness", "tier weight"]):
    res[f"{name} only"] = auc(X[:, i])
    m = np.ones(3, bool); m[i] = False
    res[f"without {name}"] = auc(X[:, m] @ W[m])
# fitted weights: train on networks 0-9, test on 10-19, and the reverse; average
fit = []
for tr, te in ((topo < 10, topo >= 10), (topo >= 10, topo < 10)):
    lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr]); s = lr.decision_function(X[te])
    fit.append((roc_auc_score(y[te], s), average_precision_score(y[te], s), lr.coef_[0]))
res["Fitted weights (held-out networks)"] = (np.mean([f[0] for f in fit]), np.mean([f[1] for f in fit]))
# hand-set score evaluated on the same held-out halves, for a like-for-like comparison
hs = [ (roc_auc_score(y[te], (X@W)[te]), average_precision_score(y[te], (X@W)[te])) for te in (topo>=10, topo<10)]
res["Full score, same held-out halves"] = tuple(np.mean(hs, 0))
for k, v in res.items(): print(f"{k:40s} ROC-AUC {v[0]:.3f}  PR-AUC {v[1]:.3f}")
print("fitted coefs (reach, btw, tier):", [np.round(f[2], 2) for f in fit])
# operating metrics at tau_d = 0.40
s = X @ W; veto = s > 0.40
tp = (veto & (y == 1)).sum(); fp = (veto & (y == 0)).sum(); fn = (~veto & (y == 1)).sum(); tn = (~veto & (y == 0)).sum()
op = {"prevalence": y.mean(), "precision_PPV": tp / (tp + fp), "recall": tp / (tp + fn), "specificity": tn / (tn + fp),
      "false_veto_rate": fp / (fp + tn), "NPV": tn / (tn + fn), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
print({k: round(float(v), 4) for k, v in op.items()})
json.dump({"auc": {k: [float(a), float(b)] for k, (a, b) in res.items()}, "op": {k: float(v) for k, v in op.items()}},
          open("results_riskscore.json", "w"), indent=1)
