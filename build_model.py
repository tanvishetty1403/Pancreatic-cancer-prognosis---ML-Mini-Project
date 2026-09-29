"""Fit the final combined Cox model on all patients and save everything the demo needs.
Run once after clean.py + baseline.ipynb:  python build_model.py
"""
import pickle
import numpy as np, pandas as pd
import survival_common as sc

BUNDLE = sc.PROC / "demo_bundle.pkl"


def build():
    df = sc.feature_sets()["combined"]
    feats = sc.feats_of(df)
    cph, mu, sd = sc.fit(df, sc.PEN_FINAL)
    Z = (df[feats] - mu) / sd
    lp = cph.predict_log_partial_hazard(Z)
    z = (cph.params_ / cph.standard_errors_).abs().sort_values(ascending=False)
    clin_names = [c for c in sc.load()[0].columns if c not in sc.TARGET and c in feats]
    b = dict(cph=cph, mu=mu, sd=sd, feats=feats, clin=clin_names,
             genes=[f for f in feats if f not in clin_names], raw=df[feats], cohort_lp=lp,
             median_lp=float(lp.median()), top_genes=[g for g in z.index if g not in clin_names][:6],
             duration=df.duration, event=df.event)
    pickle.dump(b, open(BUNDLE, "wb"))
    return b


def load():
    return pickle.load(open(BUNDLE, "rb")) if BUNDLE.exists() else build()


if __name__ == "__main__":
    b = build()
    print("saved", BUNDLE, "| features:", len(b["feats"]), "| top genes:", b["top_genes"])
