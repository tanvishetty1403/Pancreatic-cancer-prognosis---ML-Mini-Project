# Demo script (about 5 minutes, two speakers)

**Before you present:** run `python build_model.py`, then `streamlit run demo_app.py`. Open it in the browser and keep `slides.pptx` on the other screen. Pick a starting patient in advance (see step 3) so you aren't hunting during the demo.

## Slide flow and who says what

| Time | Slide | Speaker | Say |
|---|---|---|---|
| 0:00 | 1 Title | Tanvi | One sentence: we predict how long pancreatic cancer patients survive, using Cox models on public TCGA data. |
| 0:30 | 2 Data | Tanvi | 183 patients, 99 deaths. The rest are *censored*: still alive at last follow-up, so the Cox model uses them without treating them as deaths. Gene expression stands in for radiomics because both are many-features, few-patients. |
| 1:00 | 3–4 Method, C-index | Tanvi | Ridge-penalised Cox, 10-fold CV, scaling inside each fold. Combined model reaches 0.67 vs 0.63 clinical only. Tuned scores are slightly optimistic. |
| 1:45 | 5 KM + log-rank | Bhavana | Each patient scored by a model that never saw them, split at the median. State the log-rank p and hazard ratio from the slide. Point at the at-risk table: the tail is thin. |
| 2:30 | 6 Forest | Bhavana | HR above 1 means shorter survival; red intervals exclude 1. Name the top two or three features from the figure. Most intervals are wide: 183 patients is small. |
| 3:00 | 7 Covariate curves | Bhavana | Same model, one feature moved at a time. This shows direction and size of effect. |
| 3:30 | 8 Demo | Bhavana | Switch to the browser. |

## Live demo steps (about 90 seconds)

1. **Cohort median patient.** Say: "This is the reference: relative hazard about 1.0, curve sits on the grey average line."
2. **Move one clinical control**, for example age up or lymph-node positive on. The relative hazard rises and the curve drops below the grey line. Say: "This matches the hazard ratio we saw in the forest plot."
3. **Load a real patient** from the "Start from" dropdown (use the ID you picked earlier). Read out the risk percentile and group.
4. **Move the top gene slider** to +2 SD (the sidebar lists the model's strongest genes). Point at the bar chart: "This is what drives the score."
5. Wrap up: "Small cohort, research prototype, not clinical."

## Likely questions

- **Why not just use stage?** The clinical model on its own has C-index 0.63; adding genes lifts it to 0.67, but with 183 patients that gain has uncertainty. (Check the slide 4 numbers before answering.)
- **Is 0.67 good?** Modest. 0.5 is chance. Prognostic models in pancreatic cancer usually land in this range.
- **Why out-of-fold scores for the KM curve?** In-sample scores would flatter the model because it already saw those patients' outcomes.
- **Why median split?** It is a simple, common choice. It throws away information, which is why the C-index is the main metric.
- **Are the hazard-ratio CIs exact?** No. They come from a penalised fit, so treat them as approximate.
- **Why does the demo only have a few gene sliders?** The model uses 100 genes. The sliders cover the six strongest; the rest stay at the starting patient's values.
- **Sweep on the same CV?** Yes, so tuned numbers are slightly optimistic. Nested CV or a held-out cohort would fix that; not done here.

## If something breaks

- Streamlit won't start: run `pip install streamlit`, then retry. As a fallback, show slide 8 and describe the steps.
- `demo_bundle.pkl` missing: `python build_model.py`.
- Figures missing from the deck: run `python survival_analysis.py`, then `python build_deck.py`.
