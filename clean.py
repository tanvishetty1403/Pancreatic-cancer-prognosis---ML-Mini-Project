"""Build tidy survival tables from the cBioPortal TCGA-PAAD PanCancer Atlas download.

Expects the extracted tarball in data/raw/ (paad_tcga_pan_can_atlas_2018/).
Outputs (data/processed/): clinical.csv, expr.csv, combined.csv
"""
import pandas as pd, numpy as np
from pathlib import Path

RAW = Path("data/raw/paad_tcga_pan_can_atlas_2018")
OUT = Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)
N_GENES = 500  # top-variance genes kept as the high-dimensional "radiomics stand-in"

def read(name):
    return pd.read_csv(RAW / name, sep="\t", comment="#", low_memory=False)

# ---- clinical ----
pat = read("data_clinical_patient.txt").set_index("PATIENT_ID")
pat["duration"] = pd.to_numeric(pat["OS_MONTHS"], errors="coerce")
pat["event"] = pat["OS_STATUS"].astype(str).str.startswith("1").astype(int)  # "1:DECEASED"

# ---- features ----
# ---- features ----
feat = pd.DataFrame(index=pat.index)
feat["age"] = pd.to_numeric(pat["AGE"], errors="coerce")
feat["male"] = (pat["SEX"] == "Male").astype(int)
feat["stage"] = pat["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).str.extract(r"Stage (IV|I{1,3})")[0].map(
    {"I": 1, "II": 2, "III": 3, "IV": 4})
feat["t_stage"] = pat["PATH_T_STAGE"].astype(str).str.extract(r"T(\d)")[0].astype(float)
feat["n_positive"] = pat["PATH_N_STAGE"].astype(str).str.upper().str.startswith("N1").astype(int)
feat["m_positive"] = pat["PATH_M_STAGE"].astype(str).str.upper().str.startswith("M1").astype(int)
feat["neoadjuvant"] = (pat["HISTORY_NEOADJUVANT_TRTYN"] == "Yes").astype(int)
feat["radiation_therapy"] = (pat["RADIATION_THERAPY"] == "Yes").astype(int)
feat["prior_dx"] = (pat["PRIOR_DX"] == "Yes").astype(int)
feat["weight"] = pd.to_numeric(pat["WEIGHT"], errors="coerce")
for col, pre in [("ICD_O_3_SITE", "site"), ("ICD_O_3_HISTOLOGY", "hist")]:
    d = pd.get_dummies(pat[col].astype(str), prefix=pre, dtype=int)
    feat = feat.join(d.loc[:, d.sum() >= 10])

clin = pd.concat([pat[["duration", "event"]], feat], axis=1)
clin = clin[clin["duration"] > 0].dropna(subset=["duration"])
# fill feature gaps with median (simple; note it in the write-up)
fcols = [c for c in clin.columns if c not in ("duration", "event")]
clin[fcols] = clin[fcols].fillna(clin[fcols].median())
clin.to_csv(OUT / "clinical.csv")

# ---- expression ----
ex = read("data_mrna_seq_v2_rsem.txt")
ex = ex.drop(columns=[c for c in ("Entrez_Gene_Id",) if c in ex]).dropna(subset=["Hugo_Symbol"])
ex = ex.drop_duplicates("Hugo_Symbol").set_index("Hugo_Symbol")
ex.columns = [c[:12] for c in ex.columns]           # sample id -> patient id
ex = ex.loc[:, ~ex.columns.duplicated()]
ex = np.log2(ex.astype(float) + 1)
ex = ex.loc[ex.var(axis=1).nlargest(N_GENES).index].T   # patients x genes
ex = ex.loc[ex.index.intersection(clin.index)]
ex.to_csv(OUT / "expr.csv")

comb = clin.loc[ex.index].join(ex)
comb.to_csv(OUT / "combined.csv")
print(f"patients: {len(clin)} clinical, {len(ex)} with expression | "
      f"events: {int(clin.event.sum())} | clinical feats: {len(fcols)} | genes: {ex.shape[1]}")

