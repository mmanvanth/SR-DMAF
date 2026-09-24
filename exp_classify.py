import json, time, pickle, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from core import load, static_view, behav_view, net_view, MCDropoutMLP

DATA = "/mnt/user-data/uploads/ransom.csv"
df, info = load(DATA)
Xs, Xb, Xn = static_view(df), behav_view(df), net_view(df)
views = {"static": Xs.values, "behaviour": Xb.values, "network": Xn.values}
views["fused"] = np.hstack([views["static"], views["behaviour"], views["network"]])
fams = sorted(df.Family.unique(), key=lambda f: (f != "Benign", f))  # Benign = class 0
fidx = {f: i for i, f in enumerate(fams)}
y = df.Family.map(fidx).values
C = len(fams)
ransom_fams = sorted(df[df.Category == "Ransomware"].Family.unique())
info.update({"n_static": Xs.shape[1], "n_behaviour": Xb.shape[1], "n_network": Xn.shape[1], "n_classes": C,
             "families": fams, "ransomware_families": ransom_fams})
print(json.dumps({k: v for k, v in info.items() if k not in ("families",)}, indent=1))
macro = lambda a, b: f1_score(a, b, average="macro")


def split(kind, seed):
    idx = np.arange(len(df))
    if kind == "group":
        g = df.md5.values
        tr, rest = next(GroupShuffleSplit(1, test_size=0.30, random_state=seed).split(idx, y, g))
        va, te = next(GroupShuffleSplit(1, test_size=0.50, random_state=seed).split(rest, y[rest], g[rest]))
        return tr, rest[va], rest[te]
    tr, rest = next(StratifiedShuffleSplit(1, test_size=0.30, random_state=seed).split(idx, y))
    va, te = next(StratifiedShuffleSplit(1, test_size=0.50, random_state=seed).split(rest, y[rest]))
    return tr, rest[va], rest[te]


def metrics(yt, P):
    pred = P.argmax(1)
    bt, pm = (yt != 0).astype(int), 1 - P[:, 0]
    bp = (pm > 0.5).astype(int)
    rt = np.isin(yt, [fidx[f] for f in ransom_fams])
    return {"bin_acc": accuracy_score(bt, bp), "bin_prec": precision_score(bt, bp), "bin_rec": recall_score(bt, bp),
            "bin_f1": f1_score(bt, bp), "bin_mcc": matthews_corrcoef(bt, bp), "bin_auc": roc_auc_score(bt, pm),
            "fam_acc": accuracy_score(yt, pred), "fam_macroF1": macro(yt, pred),
            "ransom_fam_macroF1": f1_score(yt[rt], pred[rt], labels=[fidx[f] for f in ransom_fams], average="macro")}


results, store = [], {}
for kind in ["group", "leaky"]:
    for seed in range(5):
        tr, va, te = split(kind, seed)
        assert kind == "leaky" or not set(df.md5.values[tr]) & set(df.md5.values[te])
        out = {}
        for name, X in views.items():
            sc = StandardScaler().fit(X[tr])
            Xtr, Xva, Xte = sc.transform(X[tr]), sc.transform(X[va]), sc.transform(X[te])
            m = MCDropoutMLP(X.shape[1], C, seed=seed)
            t0 = time.time()
            m.fit(Xtr, y[tr], Xva, y[va], C, metric=macro)
            P = m.predict_proba(Xte)
            r = {"split": kind, "seed": seed, "model": f"MLP-{name}", **metrics(y[te], P), "train_s": time.time() - t0}
            results.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            if kind == "group":
                out[name] = {"model": m, "scaler": sc, "Pva": m.predict_proba(Xva), "Pte": P,
                             "MCva": m.mc_proba(Xva, 50) if name == "fused" else None,
                             "MCte": m.mc_proba(Xte, 50) if name == "fused" else None}
        hgb = HistGradientBoostingClassifier(max_iter=300, random_state=seed, class_weight="balanced")
        hgb.fit(views["fused"][tr], y[tr])
        r = {"split": kind, "seed": seed, "model": "HGB-fused", **metrics(y[te], hgb.predict_proba(views["fused"][te]))}
        results.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        if kind == "group":
            store[seed] = {"tr": tr, "va": va, "te": te, "agents": out}

pd.DataFrame(results).to_csv("results_classification.csv", index=False)
with open("store.pkl", "wb") as f:
    pickle.dump({"store": store, "y": y, "fams": fams, "ransom_fams": ransom_fams, "info": info,
                 "md5": df.md5.values, "category": df.Category.values}, f)
