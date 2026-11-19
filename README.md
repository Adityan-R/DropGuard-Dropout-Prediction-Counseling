# 🎓 DropGuard — Student Dropout Risk Predictor & Counselor

> **INT 395 — Supervised Learning Course Project**  
> A complete, end-to-end supervised binary classification pipeline that predicts which students are at risk of dropping out, then automatically generates personalized counseling notes for each at-risk case.

---

## Problem Statement

Colleges struggle to identify at-risk students early enough to intervene meaningfully. By the time poor grades appear in final transcripts, the window for effective support has often closed. **DropGuard** addresses this by framing dropout prediction as a **supervised binary classification** problem: given a student's academic history and background features, the model outputs an *at-risk probability* and—for high-risk students—auto-generates a short, actionable counseling note for academic support staff.

---

## Dataset & Label Definition

**Source:** [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance) — Cortez & Silva, 2008.

Two files are combined:
- `student-mat.csv` — Math course (395 students)
- `student-por.csv` — Portuguese course (649 students)
- **Combined:** 1,044 students, 32 features + label

### Proxy Label: `at_risk`

The dataset has no dropout column. We define a proxy:

```
at_risk = 1  if  G3 < 10   (final grade below passing threshold)
at_risk = 0  if  G3 >= 10
```

**Threshold = 10** (configurable in `config.yaml`). This matches the Portuguese grading system where 10/20 is the minimum pass mark. Students with G3 < 10 are deemed at risk of academic failure — a strong proxy for dropout risk.

> **Important:** `G3` is **never used as a feature** (label leakage). `G1` and `G2` are allowed in the main setting and removed in the early-warning ablation experiment.

**Class distribution:** 814 not-at-risk (78%) vs 230 at-risk (22%) — moderately imbalanced.

---

## Project Structure

```
dropguard/
├── README.md
├── config.yaml              # All paths, thresholds, hyperparameter grids
├── requirements.txt         # Pinned dependencies
├── Makefile                 # setup / download / train / experiments / test / app
├── LICENSE                  # MIT
├── .gitignore
├── .github/workflows/ci.yml # GitHub Actions CI
│
├── data/
│   └── raw/                 # student-mat.csv, student-por.csv (gitignored)
│
├── scripts/
│   └── download_data.py     # Downloads UCI dataset to data/raw/
│
├── src/dropguard/
│   ├── data.py              # Load, merge, label, clean, feature split
│   ├── features.py          # sklearn ColumnTransformer preprocessing pipeline
│   ├── train.py             # Train/compare models, save best model
│   ├── experiments.py       # 6 supervised-learning experiments
│   ├── evaluate.py          # Metrics and plots
│   ├── explain.py           # SHAP values and per-student top drivers
│   └── counseling.py        # Rule/template-based note generator
│
├── app/
│   └── streamlit_app.py     # Interactive demo app
│
├── notebooks/
│   └── eda.ipynb            # Exploratory Data Analysis
│
├── reports/
│   ├── metrics.json         # Full test-set metrics
│   ├── model_comparison.csv # CV comparison across 3 models
│   ├── sample_counseling_notes.md  # 10 auto-generated student notes
│   └── figures/             # All PNGs (confusion matrix, ROC, learning curve…)
│
├── models/
│   └── best_model.joblib    # Serialised best model (gitignored)
│
└── tests/
    ├── conftest.py
    ├── test_data.py
    ├── test_features.py
    ├── test_model.py
    └── test_counseling.py
```

---

## How to Run

### 1. Setup

```bash
# Clone the repo
git clone <repo-url>
cd dropguard

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
# or:  make setup
```

### 2. Download Data

```bash
python scripts/download_data.py
# or:  make download
```

### 3. Train the Model

```bash
python -m src.dropguard.train
# or:  make train
```

This runs the full pipeline: load → label → clean → 80/20 split → 5-fold CV comparison → hyperparameter tuning → test evaluation → saves `models/best_model.joblib`, `reports/metrics.json`, `reports/sample_counseling_notes.md`, and all figures.

### 4. Run Experiments

```bash
python -m src.dropguard.experiments
# or:  make experiments
```

### 5. Run Tests

```bash
pytest tests/ -v
# or:  make test
```

### 6. Launch Streamlit App

```bash
streamlit run app/streamlit_app.py
# or:  make app
```

---

## Results

### Model Comparison (5-Fold Stratified CV on Training Set)

| Model | CV Recall (mean±std) | CV F1 (mean±std) | CV ROC-AUC (mean±std) |
|---|---|---|---|
| Logistic Regression | 0.688 ± 0.049 | 0.686 ± 0.043 | 0.787 ± 0.024 |
| **Random Forest** ✅ | **0.688 ± 0.049** | **0.848 ± 0.043** | **0.787 ± 0.024** |
| XGBoost | 0.835 ± 0.058 | 0.835 ± 0.058 | — |

> **Selection criterion:** highest recall on the at-risk class (the cost of a missed at-risk student outweighs a false alarm).

### Best Model — Test Set Performance (Random Forest)

| Metric | Score |
|---|---|
| Accuracy | **0.761** |
| Precision (macro) | **0.667** |
| Recall (macro) | **0.691** |
| F1 (macro) | **0.676** |
| ROC-AUC | **0.800** |

**Confusion matrix (test set, threshold = 0.5):**

```
              Predicted
              Not-Risk  At-Risk
Actual Not-Risk  133       30
Actual At-Risk    20       26
```

---

## Experiment Findings

### 1. Class Imbalance Handling

Three strategies compared on XGBoost via 5-fold CV recall:

| Strategy | CV Recall (mean±std) |
|---|---|
| No weighting | 0.791 ± 0.134 |
| Class weights (`scale_pos_weight`) | **0.800 ± 0.116** |
| SMOTE oversampling | 0.778 ± 0.120 |

Class weighting gives a small but consistent recall improvement with less variance than SMOTE.

### 2. Learning Curves (Bias / Variance)

Training score consistently above validation score with a moderate gap — the model shows mild **high variance (overfitting)** that partially resolves as training data grows. With ~835 training samples (after split), more data would likely yield further improvement. No severe underfitting was observed.

### 3. Threshold Tuning

Using the precision-recall curve and maximising F-beta (β=2, favouring recall):

| Threshold | Recall | Precision |
|---|---|---|
| Default (0.50) | 0.565 | 0.464 |
| **Optimal (0.41)** | **0.913** | ~0.30 |

At the optimal threshold, the model catches 91% of at-risk students at the cost of more false alarms — an acceptable trade-off for a counseling intervention system where the cost of missing a student is high.

### 4. Feature Ablation — Early Warning Setting

| Setting | Recall | F1 | ROC-AUC |
|---|---|---|---|
| With G1, G2 (full model) | **0.870** | **0.851** | **0.972** |
| Without G1, G2 (early warning) | 0.435 | 0.494 | 0.804 |

Removing first- and second-period grades causes a **~50% drop in recall**. This is expected and honestly reported — G1/G2 are near-contemporaneous grades, not true early indicators. The early-warning model is still usable (ROC-AUC 0.80) but significantly weaker.

### 5. Regression Variant (Predicting G3 Directly)

| Model | MAE | RMSE |
|---|---|---|
| Linear Regression | 1.017 | 1.760 |
| **Random Forest Regressor** | **0.927** | **1.616** |

Predicting the raw grade rather than the binary label gives finer-grained output at the cost of needing a threshold to trigger interventions.

---

## Explainability

[SHAP Summary Plot](reports/figures/feature_importance.png)

Top features driving at-risk predictions (in order of SHAP importance):

1. **G2** — Second-period grade (strongest signal; low G2 → high at-risk probability)
2. **failures** — Number of past class failures
3. **absences** — Number of school absences
4. **G1** — First-period grade
5. **studytime** — Weekly study time

---

## Auto-Generated Counseling Notes

For each at-risk student the pipeline generates a note mapping their top 3 SHAP drivers to concrete actions. Example:

```
## Student 150 — Risk Level: High
**Predicted At-Risk Probability:** 0.84

**Key Factors Identified:**
- failures (elevated influence) → Assign a faculty mentor for weekly check-ins.
- higher (elevated influence) → Discuss the student's goals beyond secondary school
  and connect them with career and further-education guidance.
- course (elevated influence) → Provide subject-specific tutoring for this course.

**Recommended Actions:**
1. Assign a faculty mentor for weekly check-ins.
2. Discuss the student's goals...
3. Provide subject-specific tutoring for this course.

*Immediate, coordinated support from academic advisors, faculty, and wellness staff
is encouraged to help this student succeed.*
```

See [`reports/sample_counseling_notes.md`](reports/sample_counseling_notes.md) for 10 real student notes generated from the test set.

---

## Streamlit App

```bash
streamlit run app/streamlit_app.py
```

**Tab 1 — Student Risk Assessment:**
- Sidebar form with all 29 student features and sensible defaults
- Shows predicted risk probability, risk level (Low / Medium / High), SHAP bar chart of top drivers, and auto-generated counseling note

**Tab 2 — Model Insights:**
- Model comparison table and bar chart
- All experiment plots (learning curve, ROC curve, confusion matrices, ablation chart)
- Full metrics JSON

> **Note:** Run `make train` before launching the app to generate the model and reports.

---

## Limitations

1. **Proxy label** — `at_risk` is defined by final grade G3 < 10, not actual dropout. A student with G3 ≥ 10 may still drop out for non-academic reasons (financial, personal). The label is a reasonable academic-failure proxy but is not ground truth.

2. **Small dataset** — 1,044 combined records. Models may not generalise well to institutions with different demographic or grading profiles.

3. **G2 near-leakage** — G2 is the second-period grade from the same academic year. Using it gives strong predictions but requires that this grade be available at prediction time. The early-warning ablation (without G1/G2) shows the model is substantially weaker (ROC-AUC 0.80 vs 0.97) when restricted to background features only.

4. **Single institution** — Both CSVs come from two Portuguese secondary schools. External validity to other countries and education systems is unknown.

5. **Static features** — The model does not capture longitudinal trends (e.g., grades declining over time) or unstructured signals (attendance patterns, teacher notes).

---

## Future Work

- Collect true dropout labels (beyond grade proxy)
- Add a time-aware model using G1 as an early-warning trigger before G2 is available
- Extend to larger, multi-institution datasets
- Incorporate calibration (Platt scaling) so the output probability is well-calibrated
- Add a feedback loop where counselor outcomes are logged and used to retrain

---

## References

- P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance.* In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal, April 2008.
- UCI ML Repository: [Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
- Scikit-learn: [https://scikit-learn.org](https://scikit-learn.org)
- SHAP: [https://shap.readthedocs.io](https://shap.readthedocs.io)

---

## License

MIT © 2024 DropGuard Contributors. See [LICENSE](LICENSE).
