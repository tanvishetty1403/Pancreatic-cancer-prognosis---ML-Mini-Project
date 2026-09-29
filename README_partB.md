## Part B: survival curves, forest plots, demo, slides (Y. V. Bhavana)

Run after `clean.py` and `baseline.ipynb` (needs `data/processed/*.csv`):

```
pip install lifelines streamlit matplotlib pandas scikit-learn python-pptx pillow
python survival_analysis.py     # figures -> reports/figures/, numbers -> reports/results.json
python build_model.py           # fits final model -> data/processed/demo_bundle.pkl
streamlit run demo_app.py       # live risk calculator
python build_deck.py            # slides.pptx filled from the figures/results above
```

`survival_analysis.ipynb` runs the same analysis interactively.

Outputs: `km_high_low.png` (out-of-fold median split + log-rank), `forest_clinical.png`, `forest_combined_top.png`, `covariate_curves.png`, `km_by_stage.png`, `forest_*.csv`.
