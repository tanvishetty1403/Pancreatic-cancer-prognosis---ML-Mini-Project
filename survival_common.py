"""Shared helpers for Part B (KM curves, forest plot, covariate curves, demo).

Mirrors the preprocessing and model settings of baseline.ipynb so results line up
with Part A: L2-penalised Cox, features standardised on the fit data, top-100
variance genes, penaliser 1.0.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

PROC = Path("data/processed")
FIG = Path("reports/figures")
N_GENES, PEN_FINAL, PEN_CLIN, SEED = 100, 1.0, 1.0, 42
TARGET = ["duration", "event"]


def load():
    clin = pd.read_csv(PROC / "clinical.csv", index_col=0)
    expr = pd.read_csv(PROC / "expr.csv", index_col=0)
    comb = pd.read_csv(PROC / "combined.csv", index_col=0)
    return clin, expr, comb


def prep(df, min_count=10):
    """Identical to prep() in baseline.ipynb."""
    y = df[TARGET]
    X = df.drop(columns=TARGET)
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median())
    binary = [c for c in X if X[c].nunique() == 2]
    X = X.drop(columns=[c for c in binary if X[c].sum() < min_count or (1 - X[c]).sum() < min_count])
    X = X.loc[:, X.std() > 0]
    return pd.concat([y, X], axis=1)


def feature_sets(n_genes=N_GENES):
    clin, expr, comb = load()
    clin_cols = [c for c in clin.columns if c not in TARGET]
    genes = list(expr.columns[:n_genes])
    return {
        "clinical": prep(clin.loc[comb.index]),
        "combined": prep(comb[TARGET + clin_cols + genes]),
    }


def feats_of(df):
    return [c for c in df if c not in TARGET]


def standardise(df, mu=None, sd=None):
    f = feats_of(df)
    mu = df[f].mean() if mu is None else mu
    sd = (df[f].std().replace(0, 1)) if sd is None else sd
    out = df.copy()
    out[f] = (df[f] - mu) / sd
    return out, mu, sd


def fit(df, pen):
    """Fit on standardised data; returns (cph, mu, sd)."""
    sc, mu, sd = standardise(df)
    cph = CoxPHFitter(penalizer=pen, l1_ratio=0.0).fit(sc, "duration", "event")
    return cph, mu, sd


def oof_risk(df, pen=PEN_FINAL, k=10, seed=SEED):
    """Out-of-fold log-partial-hazard, same recipe as Part A."""
    f = feats_of(df)
    oof = pd.Series(index=df.index, dtype=float)
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(df):
        d_tr, d_te = df.iloc[tr].copy(), df.iloc[te].copy()
        mu, sd = d_tr[f].mean(), d_tr[f].std().replace(0, 1)
        d_tr[f] = (d_tr[f] - mu) / sd
        d_te[f] = (d_te[f] - mu) / sd
        m = CoxPHFitter(penalizer=pen, l1_ratio=0.0).fit(d_tr, "duration", "event")
        oof.iloc[te] = m.predict_log_partial_hazard(d_te).values
    return oof


def hazard_table(cph, sd, raw_units=True):
    """HR table. If raw_units, HR is per 1 raw unit (per year, present-vs-absent...);
    otherwise per 1 SD. Rescaling is exact because the model is linear in the features."""
    s = cph.summary[["coef", "se(coef)", "coef lower 95%", "coef upper 95%", "p"]].copy()
    div = sd.reindex(s.index) if raw_units else 1.0
    t = pd.DataFrame({
        "HR": np.exp(s["coef"] / div),
        "lo": np.exp(s["coef lower 95%"] / div),
        "hi": np.exp(s["coef upper 95%"] / div),
        "z": s["coef"] / s["se(coef)"],
        "p": s["p"],
    })
    return t
