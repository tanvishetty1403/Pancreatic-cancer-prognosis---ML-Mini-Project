"""Part B analysis: KM curves (high vs low risk), log-rank, forest plots, covariate curves.

Run:  python survival_analysis.py      (writes reports/figures/*.png, reports/results.json)
or import the functions from survival_analysis.ipynb.
"""
import json
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index
import survival_common as sc

from matplotlib.ticker import FuncFormatter, NullFormatter
from matplotlib.colors import LinearSegmentedColormap
CMAP = LinearSegmentedColormap.from_list("risk", ["#2a9d8f", "#d99a1e", "#c0392b"])
CLR = {"low": "#2a9d8f", "high": "#c0392b", "ink": "#22303c"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 110})


def _save(fig, name):
    sc.FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(sc.FIG / name, dpi=200, bbox_inches="tight")


# ------------------------------------------------------------------ 1. KM + log-rank
def risk_groups():
    """Out-of-fold risk scores (Part A's file if present, else recomputed) + median split."""
    p = sc.PROC / "risk_scores_oof.csv"
    if p.exists():
        r = pd.read_csv(p, index_col=0)
    else:
        df = sc.feature_sets()["combined"]
        r = df[sc.TARGET].assign(risk_score=sc.oof_risk(df))
    r["group"] = np.where(r.risk_score > r.risk_score.median(), "high", "low")
    return r


def km_high_low(r=None, save=True):
    r = risk_groups() if r is None else r
    hi, lo = r[r.group == "high"], r[r.group == "low"]
    lr = logrank_test(hi.duration, lo.duration, hi.event, lo.event)
    cx = CoxPHFitter().fit(
        r.assign(high=(r.group == "high").astype(int))[["duration", "event", "high"]], "duration", "event")
    hr = cx.summary.loc["high"]
    res = {
        "n": int(len(r)), "events": int(r.event.sum()),
        "logrank_chi2": float(lr.test_statistic), "logrank_p": float(lr.p_value),
        "hr_high_vs_low": float(np.exp(hr["coef"])),
        "hr_lo": float(np.exp(hr["coef lower 95%"])), "hr_hi": float(np.exp(hr["coef upper 95%"])),
        "pooled_oof_cindex": float(concordance_index(r.duration, -r.risk_score, r.event)),
    }
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    fits = []
    for name, d in [("low", lo), ("high", hi)]:
        k = KaplanMeierFitter().fit(d.duration, d.event, label=f"{name.capitalize()} risk (n={len(d)})")
        k.plot_survival_function(ax=ax, ci_show=True, color=CLR[name], ci_alpha=0.15, show_censors=True,
                                 censor_styles={"ms": 4, "marker": "|"})
        res[f"median_os_{name}"] = None if np.isinf(k.median_survival_time_) else float(k.median_survival_time_)
        fits.append(k)
    p = res["logrank_p"]
    ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.98, 0.95, f"Log-rank {ptxt}\nHR high vs low = {res['hr_high_vs_low']:.2f} "
            f"({res['hr_lo']:.2f}–{res['hr_hi']:.2f})", transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="#cccccc"))
    ax.set(xlabel="Months", ylabel="Overall survival probability", ylim=(0, 1.02),
           title="Overall survival by out-of-fold risk group (median split)")
    add_at_risk_counts(*fits, ax=ax, rows_to_show=["At risk"])
    if save: _save(fig, "km_high_low.png")
    return fig, res


# ------------------------------------------------------------------ 2. Forest plots
def _forest(ax, t, title, xlabel):
    t = t.iloc[::-1]
    y = np.arange(len(t))
    sig = t.p < 0.05
    ax.hlines(y, t.lo, t.hi, color=np.where(sig, CLR["high"], "#8a96a0"), lw=2)
    ax.scatter(t.HR, y, color=np.where(sig, CLR["high"], "#8a96a0"), zorder=3, s=36)
    ax.axvline(1, color=CLR["ink"], lw=1, ls="--")
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(t.index)
    ax.set(xlabel=xlabel + "\n(red = p < 0.05)", title=title)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([v for v in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5) if t.lo.min() / 1.2 <= v <= t.hi.max() * 1.2] or [1])
    for yi, (hr, lo, hi) in zip(y, t[["HR", "lo", "hi"]].values):
        ax.annotate(f"{hr:.2f} ({lo:.2f}–{hi:.2f})", (1.0, yi), xycoords=("axes fraction", "data"),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5)


def forest_plots(sets=None, top_n=15, save=True):
    sets = sc.feature_sets() if sets is None else sets
    out = {}
    # clinical model, raw units
    cph_c, mu_c, sd_c = sc.fit(sets["clinical"], sc.PEN_CLIN)
    tc = sc.hazard_table(cph_c, sd_c, raw_units=True).sort_values("HR")
    fig1, ax = plt.subplots(figsize=(8.8, 0.42 * len(tc) + 1.6))
    _forest(ax, tc, "Clinical model: hazard ratios (95% CI)", "Hazard ratio, log scale (per year / per unit / present vs absent)")
    fig1.subplots_adjust(right=0.72)
    # combined model, per SD, top by |z|
    cph, mu, sd = sc.fit(sets["combined"], sc.PEN_FINAL)
    tb = sc.hazard_table(cph, sd, raw_units=False)
    tt = tb.reindex(tb.z.abs().sort_values(ascending=False).index).head(top_n).sort_values("HR")
    fig2, ax = plt.subplots(figsize=(8.8, 0.42 * len(tt) + 1.6))
    _forest(ax, tt, f"Combined model: top {top_n} features by |z| (95% CI)", "Hazard ratio per +1 SD, log scale")
    fig2.subplots_adjust(right=0.72)
    if save:
        _save(fig1, "forest_clinical.png"); _save(fig2, "forest_combined_top.png")
        tc.round(4).to_csv(sc.PROC / "forest_clinical.csv"); tb.round(4).to_csv(sc.PROC / "forest_combined_all.csv")
    return fig1, fig2, tc, tb


# ------------------------------------------------------------------ 3. Covariate curves
def _profile_curves(cph, mu, sd, base_raw, feat, raw_values):
    """Predicted survival curves varying one feature, all others at cohort median/mean."""
    rows = []
    for v in raw_values:
        r = base_raw.copy(); r[feat] = v
        rows.append(((r - mu) / sd))
    X = pd.DataFrame(rows, index=[str(v) for v in raw_values])
    return cph.predict_survival_function(X)


def covariate_curves(sets=None, save=True):
    sets = sc.feature_sets() if sets is None else sets
    cc, mc, sdc = sc.fit(sets["clinical"], sc.PEN_CLIN)
    base_c = sets["clinical"][sc.feats_of(sets["clinical"])].median()
    cb, mb, sdb = sc.fit(sets["combined"], sc.PEN_FINAL)
    genes = [g for g in cb.summary.index if g not in sets["clinical"].columns]
    top = cb.summary.loc[genes].assign(a=lambda d: (d["coef"] / d["se(coef)"]).abs()).sort_values("a", ascending=False).index[:2]
    base_b = sets["combined"][sc.feats_of(sets["combined"])].mean()
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharey=True)

    def draw(ax, S, labels, title):
        for i, c in enumerate(S.columns):
            ax.step(S.index, S[c], where="post", color=CMAP(i / max(1, len(S.columns) - 1)), lw=2, label=labels[i])
        ax.set(title=title, xlabel="Months", ylim=(0, 1.02)); ax.legend(frameon=False, fontsize=9)

    if "age" in base_c.index:
        q = [round(sets["clinical"].age.quantile(p)) for p in (0.1, 0.5, 0.9)]
        draw(axs[0, 0], _profile_curves(cc, mc, sdc, base_c, "age", q), [f"age {v}" for v in q], "Age (clinical model)")
    if "stage" in base_c.index:
        sv = sorted(set(sets["clinical"].stage.dropna().round().astype(int)))
        if sv:
            draw(axs[0, 1], _profile_curves(cc, mc, sdc, base_c, "stage", sv), [f"stage {v}" for v in sv], "AJCC stage (clinical model)")
        else:
            axs[0, 1].set(title="AJCC stage unavailable", xlabel="Months", ylim=(0, 1.02))
    for ax, g in zip([axs[1, 0], axs[1, 1]], top):
        m, s = base_b[g], sets["combined"][g].std()
        vals = [m - s, m, m + s]
        draw(ax, _profile_curves(cb, mb, sdb, base_b, g, vals), ["−1 SD", "mean", "+1 SD"], f"{g} expression (combined model)")
    axs[0, 0].set_ylabel("Predicted survival"); axs[1, 0].set_ylabel("Predicted survival")
    fig.suptitle("Covariate-adjusted survival: vary one feature, hold the rest at cohort median/mean", y=1.0, fontweight="bold")
    fig.tight_layout()
    if save: _save(fig, "covariate_curves.png")
    return fig, list(top)


def km_by_stage(clin=None, save=True):
    """Descriptive (unadjusted) KM by AJCC stage; stages III-IV pooled if sparse."""
    clin = sc.load()[0] if clin is None else clin
    d = clin.copy()
    d["stage_grp"] = d.stage.round().astype("Int64").map({1: "I", 2: "II", 3: "III–IV", 4: "III–IV"})
    counts = d.stage_grp.value_counts()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    fits = []
    for g, c in zip(["I", "II", "III–IV"], ["#2a9d8f", "#e9c46a", "#c0392b"]):
        s = d[d.stage_grp == g]
        if len(s) < 5: continue
        k = KaplanMeierFitter().fit(s.duration, s.event, label=f"Stage {g} (n={len(s)})")
        k.plot_survival_function(ax=ax, ci_show=False, color=c, show_censors=True, censor_styles={"ms": 4, "marker": "|"})
        fits.append(k)
    keep = d[d.stage_grp.isin([g for g in counts.index if counts[g] >= 5])]
    if keep.stage_grp.nunique() >= 2:
        p = multivariate_logrank_test(keep.duration, keep.stage_grp, keep.event).p_value
        annotation = f"Log-rank (all groups) p = {p:.3f}"
    else:
        p = None
        annotation = "AJCC stage unavailable"
    ax.text(0.98, 0.95, annotation, transform=ax.transAxes, ha="right", va="top")
    ax.set(xlabel="Months", ylabel="Overall survival probability", ylim=(0, 1.02), title="Overall survival by AJCC stage (unadjusted)")
    if fits:
        add_at_risk_counts(*fits, ax=ax, rows_to_show=["At risk"])
    if save: _save(fig, "km_by_stage.png")
    return fig, p


# ------------------------------------------------------------------ run everything
def main():
    matplotlib.use("Agg")
    fig, res = km_high_low()
    _, _, tc, tb = forest_plots()
    _, top = covariate_curves()
    _, p_stage = km_by_stage()
    res["stage_logrank_p"] = p_stage
    res["top_genes"] = top
    res["top_clinical"] = tc.reindex(tc.z.abs().sort_values(ascending=False).index).head(3).round(3).reset_index().to_dict("records")
    with open("reports/results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
