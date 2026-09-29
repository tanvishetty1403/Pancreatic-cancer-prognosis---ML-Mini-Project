"""Streamlit demo: enter patient features -> risk score + predicted survival curve.
Run:  streamlit run demo_app.py
Research prototype only (TCGA-PAAD, n=183). Not for clinical use.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import build_model

st.set_page_config(page_title="PAAD survival risk", page_icon="🧬", layout="wide")


@st.cache_resource
def get_bundle():
    return build_model.load()


B = get_bundle()
cph, mu, sd, feats, raw = B["cph"], B["mu"], B["sd"], B["feats"], B["raw"]

st.title("Pancreatic cancer survival risk")
st.caption("Cox proportional-hazards model on TCGA-PAAD (clinical + top-variance gene expression). "
           "Research prototype — not a clinical tool.")

# ---- starting profile -------------------------------------------------
start = st.sidebar.selectbox("Start from", ["Cohort median patient"] + list(raw.index))
base = raw.median() if start == "Cohort median patient" else raw.loc[start]
k = start  # widget keys change with the selection so sliders reset to the chosen patient

x = base.copy()
st.sidebar.header("Clinical features")
NICE = {"age": "Age (years)", "weight": "Weight (kg)", "stage": "AJCC stage (1–4)", "t_stage": "T stage (1–4)"}
for f in [f for f in B["clin"] if raw[f].nunique() > 2]:
    lo, hi = float(raw[f].min()), float(raw[f].max())
    step = 1.0
    x[f] = st.sidebar.slider(NICE.get(f, f), lo, hi, float(np.clip(base[f], lo, hi)), step, key=f"{f}_{k}")
BIN = {"male": "Male", "n_positive": "Lymph-node positive", "m_positive": "Distant metastasis",
       "neoadjuvant": "Neoadjuvant treatment", "radiation_therapy": "Radiation therapy", "prior_dx": "Prior diagnosis"}
main_bin = [f for f in B["clin"] if raw[f].nunique() == 2 and f in BIN]
other_bin = [f for f in B["clin"] if raw[f].nunique() == 2 and f not in BIN]
for f in main_bin:
    x[f] = float(st.sidebar.checkbox(BIN[f], bool(base[f]), key=f"{f}_{k}"))
if other_bin:
    with st.sidebar.expander("Site / histology codes"):
        for f in other_bin:
            x[f] = float(st.checkbox(f, bool(base[f]), key=f"{f}_{k}"))

st.sidebar.header("Gene expression")
st.sidebar.caption("Top genes by model weight, as SD from cohort mean. "
                   "Other genes stay at the starting profile.")
z_base = (base - mu) / sd
for g in B["top_genes"]:
    z = st.sidebar.slider(g, -3.0, 3.0, float(np.clip(z_base[g], -3, 3)), 0.1, key=f"{g}_{k}")
    x[g] = mu[g] + z * sd[g]

# ---- prediction --------------------------------------------------------
xz = ((x[feats] - mu) / sd).to_frame().T
lp = float(cph.predict_log_partial_hazard(xz).iloc[0])
pct = float((B["cohort_lp"] < lp).mean() * 100)
high = lp > B["median_lp"]
rel = float(np.exp(lp))
surv = cph.predict_survival_function(xz).iloc[:, 0]
avg = cph.predict_survival_function(pd.DataFrame(0.0, index=[0], columns=feats)).iloc[:, 0]


def s_at(s, t):
    return float(s[s.index <= t].iloc[-1])


c1, c2, c3, c4 = st.columns(4)
c1.metric("Relative hazard", f"{rel:.2f}×", help="Hazard vs the average cohort patient (1.00×).")
ordn = lambda n: f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
c2.metric("Risk percentile", ordn(int(round(pct))), help="Share of the 183 cohort patients with a lower model risk.")
c3.metric("Risk group", "High" if high else "Low", help="Above/below the cohort median risk.")
c4.metric("12-month survival", f"{s_at(surv, 12) * 100:.0f}%", delta=f"{(s_at(surv, 12) - s_at(avg, 12)) * 100:+.0f} pts vs average")

left, right = st.columns([3, 2])
with left:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.step(avg.index, avg.values, where="post", color="#8a96a0", lw=2, label="Average patient")
    ax.step(surv.index, surv.values, where="post", color="#c0392b" if high else "#2a9d8f", lw=2.5, label="This patient")
    ax.set(xlabel="Months", ylabel="Predicted survival", ylim=(0, 1.02), title="Predicted overall survival")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
    st.pyplot(fig)
with right:
    contrib = (cph.params_ * xz.iloc[0]).sort_values(key=abs, ascending=False).head(8).iloc[::-1]
    fig2, ax2 = plt.subplots(figsize=(4.6, 4))
    ax2.barh(contrib.index, contrib.values, color=np.where(contrib > 0, "#c0392b", "#2a9d8f"))
    ax2.axvline(0, color="#22303c", lw=1)
    ax2.set(title="What drives this score", xlabel="Contribution to log-hazard (+ = worse)")
    ax2.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig2)

st.caption(f"Median predicted survival: "
           f"{(surv[surv <= 0.5].index[0] if (surv <= 0.5).any() else float('nan')):.1f} months. "
           "Model fit on all 183 patients; cross-validated C-index ≈ 0.67 (see README). "
           "Small cohort — treat outputs as illustrative.")
