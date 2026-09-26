"""Calibration of the attribution agent A4: single deterministic pass vs Monte Carlo dropout mean (T=50).
Uses the stored test-set predictions (store.pkl); no retraining."""
import pickle, json, numpy as np, pandas as pd

S = pickle.load(open("store.pkl", "rb")); store, y = S["store"], S["y"]

def ece(p, yb, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(yb[m].mean() - p[m].mean())
    return e

def top1(P, yt):  # multi-class: confidence of predicted class vs correctness
    conf, pred = P.max(1), P.argmax(1)
    return conf, (pred == yt).astype(float)

rows, diag = [], {}
for seed, D in store.items():
    A = D["agents"]["fused"]; yt = y[D["te"]]; yb = (yt != 0).astype(float)
    P1 = A["Pte"]; Pmc = A["MCte"].mean(0)
    for name, P in (("single pass", P1), ("MC dropout", Pmc)):
        pm = 1 - P[:, 0]
        c, corr = top1(P, yt)
        rows.append({"seed": seed, "model": name,
                     "ECE_det": ece(pm, yb), "Brier_det": np.mean((pm - yb) ** 2),
                     "ECE_fam": ece(c, corr), "Brier_fam": np.mean(np.sum((P - np.eye(P.shape[1])[yt]) ** 2, 1))})
        if seed == 0:
            edges = np.linspace(0, 1, 11); idx = np.clip(np.digitize(c, edges[1:-1]), 0, 9)
            diag[name] = [(float(c[idx == b].mean()), float(corr[idx == b].mean()), int((idx == b).sum()))
                          for b in range(10) if (idx == b).any()]
R = pd.DataFrame(rows); R.to_csv("results_calibration_ece.csv", index=False)
g = R.groupby("model")[["ECE_det", "Brier_det", "ECE_fam", "Brier_fam"]]
print((g.mean() * 100).round(3)); print((g.std() * 100).round(3))
json.dump(diag, open("results_reliability_diagram.json", "w"), indent=1)
for k, v in diag.items(): print(k, [(round(a, 3), round(b, 3), n) for a, b, n in v])

P = pd.read_csv("results_policies.csv"); u = P[(P.scenario == "uniform host") & (P.eff_G == 0.95)]
m = u.groupby("policy")[["false_action", "CFR", "SAR"]].mean()
b1, sr = m.loc["B1 naive detect-isolate"], m.loc["SR-DMAF (full)"]
print("benign", b1.false_action, sr.false_action, "reduction %", 100 * (1 - sr.false_action / b1.false_action), "ratio", b1.false_action / sr.false_action)
print("CFR", b1.CFR, sr.CFR, "reduction %", 100 * (1 - sr.CFR / b1.CFR), "ratio", b1.CFR / sr.CFR)
