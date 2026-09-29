# Pancreatic Cancer Prognosis with Cox Survival Models

UE24CS352A Machine Learning mini-project (Problem 6), based on Jamalian, *Pancreatic cancer prognosis using clinical and radiomic data* (CS229, 2020).

Cox proportional hazards models predict overall survival in pancreatic adenocarcinoma from clinical features, gene expression, and both combined.

## Dataset
The paper's Stanford data is under NDA, so we use the public **TCGA-PAAD PanCancer Atlas** study from cBioPortal: 183 patients, 99 deaths. Gene expression (top-variance genes) stands in for the paper's radiomic features, since both are high-dimensional.

## Setup
1. `pip install -r requirements.txt`
2. Download `paad_tcga_pan_can_atlas_2018.tar.gz` from cbioportal.org/datasets ("Pancreatic Adenocarcinoma (TCGA, PanCancer Atlas)") and extract it so the folder sits at the path set as `RAW` in `clean.py`.
3. `python clean.py` builds `data/processed/clinical.csv`, `expr.csv` and `combined.csv`.
4. Run `baseline.ipynb` top to bottom.
5. [B: add the command to launch the demo and the plotting notebook here.]

## Method
- Cox PH with an L2 penalty (lifelines), 10-fold cross-validation scored by concordance index.
- Features are standardised inside each training fold to avoid leakage.
- The penalty (0.1 to 50) and the gene count (25 to 500) are tuned by a sweep, and the results are saved to `sweep_results.csv`.

## Results (10-fold CV C-index)
| Model | Untuned (500 genes, penalty 0.1) | Tuned |
|---|---|---|
| Clinical (12 features) | 0.63 | 0.63 |
| Expression | 0.57 | 0.64 (100 genes, penalty 1) |
| Combined | 0.58 | 0.67 (100 genes, penalty 1) |

The tuned numbers are the best of a sweep chosen on the same CV, so they are slightly optimistic.

## Outputs (`data/processed/`)
- `cindex_table.csv`: untuned baseline table
- `sweep_results.csv`: penalty and gene-count sweep
- `risk_scores.csv`: in-sample risk scores (used by the demo)
- `risk_scores_oof.csv`: out-of-fold risk scores (used for KM curves and log-rank)

## Team
- [Your name]: data, modelling, cross-validation
- [B's name]: survival curves, plots, demo
