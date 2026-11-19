"""
app/streamlit_app.py
--------------------
DropGuard - Student Dropout Risk Predictor

Run from project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root resolution - must come before any local imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Third-party / stdlib imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit("streamlit is not installed. Run: pip install streamlit") from exc

try:
    import numpy as np
    import pandas as pd
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _CORE_IMPORTS_OK = True
except ImportError as _core_err:
    _CORE_IMPORTS_OK = False
    _CORE_ERR_MSG = str(_core_err)

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
try:
    from dropguard import data as _data_mod
    _LOCAL_IMPORTS_OK = True
except ImportError as _local_err:
    _LOCAL_IMPORTS_OK = False
    _LOCAL_ERR_MSG = str(_local_err)

# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------
CONFIG_PATH   = PROJECT_ROOT / "config.yaml"
MODEL_PATH    = PROJECT_ROOT / "models" / "best_model.joblib"
REPORTS_DIR   = PROJECT_ROOT / "reports"
FIGURES_DIR   = REPORTS_DIR / "figures"
METRICS_PATH  = REPORTS_DIR / "metrics.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    import yaml
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@st.cache_resource(show_spinner=False)
def _load_model():
    """Return (pipeline, True) or (None, False) if model not found."""
    if not MODEL_PATH.exists():
        return None, False
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline, True
    except Exception:
        return None, False


# ---------------------------------------------------------------------------
# SHAP helper
# ---------------------------------------------------------------------------

def _shap_bar_chart(
    pipeline,
    student_df: "pd.DataFrame",
    feature_names_raw: list[str],
    top_n: int = 5,
) -> "plt.Figure":
    """Return a matplotlib Figure with a horizontal SHAP bar chart."""
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.linear_model import LogisticRegression

    # Transform through preprocessor (all steps except last classifier)
    if hasattr(pipeline, "steps") and len(pipeline.steps) > 1:
        preprocessor = SkPipeline(pipeline.steps[:-1])
        X_transformed = preprocessor.transform(student_df)
    else:
        X_transformed = student_df.values

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_transformed = np.asarray(X_transformed, dtype=float)

    clf = pipeline[-1] if hasattr(pipeline, "__getitem__") else pipeline.steps[-1][1]

    # Pick explainer
    if isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, X_transformed)
    else:
        explainer = shap.TreeExplainer(clf)

    shap_vals = explainer.shap_values(X_transformed)

    # For binary classifiers shap_vals may be a list [class0, class1] or a
    # (n_samples, n_features, n_classes) array depending on the SHAP version.
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[-1]  # class 1 = at-risk
    shap_vals = np.asarray(shap_vals, dtype=float)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[..., -1]

    shap_row = np.asarray(shap_vals, dtype=float).ravel()

    # Try to get feature names from the fitted pipeline
    try:
        preprocessor_step = pipeline.named_steps.get("preprocessor")
        if preprocessor_step is not None and hasattr(preprocessor_step, "get_feature_names_out"):
            feat_names = list(preprocessor_step.get_feature_names_out())
        else:
            feat_names = [f"feature_{i}" for i in range(len(shap_row))]
    except Exception:
        feat_names = [f"feature_{i}" for i in range(len(shap_row))]

    # Sort by absolute SHAP value and pick top_n
    abs_shap = np.abs(shap_row)
    top_idx = np.argsort(abs_shap)[-top_n:][::-1]

    top_names  = [feat_names[i] if i < len(feat_names) else f"f{i}" for i in top_idx]
    top_values = [shap_row[i] for i in top_idx]

    # Build figure
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#e05c5c" if v > 0 else "#4c8cbf" for v in top_values]
    ax.barh(top_names[::-1], top_values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value  (-> increases risk)", fontsize=10)
    ax.set_title(f"Top {top_n} Risk Drivers", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Counseling note builder
# ---------------------------------------------------------------------------

def _counseling_note(prob: float, student: dict) -> str:
    """Return a markdown-formatted counseling recommendation."""
    risk_pct = int(prob * 100)
    lines: list[str] = [f"### 📋 Counseling Note\n"]

    if prob >= 0.7:
        lines.append(
            f"**High risk ({risk_pct}%)** - Immediate intervention recommended.\n"
        )
        lines.append("**Suggested actions:**")
        lines.append("- Schedule a meeting with the student's academic advisor this week.")
        lines.append("- Review attendance patterns - absences may be a key driver.")
        if student.get("failures", 0) > 0:
            lines.append(
                f"- Note {student['failures']} prior course failure(s); "
                "consider remedial support or tutoring."
            )
        if student.get("studytime", 2) <= 1:
            lines.append("- Low study time reported; discuss time management strategies.")
        if student.get("higher", "yes") == "no":
            lines.append(
                "- Student has not expressed intent for higher education; "
                "career counseling may help motivate engagement."
            )
    elif prob >= 0.4:
        lines.append(
            f"**Moderate risk ({risk_pct}%)** - Proactive check-in advised.\n"
        )
        lines.append("**Suggested actions:**")
        lines.append("- Send a friendly check-in email and offer office hours.")
        lines.append("- Monitor grade trajectory over the next two weeks.")
        if student.get("goout", 3) >= 4:
            lines.append("- High social activity noted; ensure it doesn't interfere with study.")
        if student.get("Walc", 1) >= 3:
            lines.append(
                "- Weekend alcohol consumption may be elevated; "
                "wellness resources are available."
            )
    else:
        lines.append(f"**Low risk ({risk_pct}%)** - Student appears to be on track.\n")
        lines.append("**Suggested actions:**")
        lines.append("- Routine check-in at mid-semester.")
        lines.append("- Encourage participation in enrichment activities.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="DropGuard - Student Dropout Risk Predictor",
        page_icon="🎓",
        layout="wide",
    )

    # Guard: core imports
    if not _CORE_IMPORTS_OK:
        st.error(f"Missing core dependency: {_CORE_ERR_MSG}")
        st.stop()

    if not _LOCAL_IMPORTS_OK:
        st.error(
            f"Could not import local modules: {_LOCAL_ERR_MSG}\n\n"
            "Make sure you run from the project root: `streamlit run app/streamlit_app.py`"
        )
        st.stop()

    # Header
    st.title("🎓 DropGuard - Student Dropout Risk Predictor")
    st.markdown(
        """
        **DropGuard** uses machine learning on the UCI Student Performance dataset to identify
        students at risk of dropping out before they fall behind. Enter a student's profile in
        the sidebar and click **Assess Student Risk** to get an instant risk score, the top
        contributing factors, and tailored counseling recommendations.
        """,
        unsafe_allow_html=False,
    )
    st.divider()

    config = _load_config()
    pipeline, model_ok = _load_model()

    # ------------------------------------------------------------------ TABS
    tab_assess, tab_insights = st.tabs(
        ["🧑‍🎓 Student Risk Assessment", "📊 Model Insights"]
    )

    # ==================================================================
    # TAB 1: Student Risk Assessment
    # ==================================================================
    with tab_assess:
        # -------------------------------------------------------------- SIDEBAR
        with st.sidebar:
            st.header("🔎 Student Profile")
            with st.form("student_form"):
                st.subheader("Demographics")
                school    = st.selectbox("School",    ["GP", "MS"],           index=0)
                sex       = st.selectbox("Sex",       ["F", "M"],             index=0)
                age       = st.slider(  "Age",        15, 22, 17)
                address   = st.selectbox("Address",   ["U", "R"],             index=0)
                famsize   = st.selectbox("Family size", ["GT3", "LE3"],       index=0)
                Pstatus   = st.selectbox("Parent status", ["T", "A"],         index=0)

                st.subheader("Parents' Education")
                Medu = st.slider("Mother's education (0-4)", 0, 4, 2)
                Fedu = st.slider("Father's education (0-4)", 0, 4, 2)

                st.subheader("Family Background")
                _jobs = ["teacher", "health", "services", "at_home", "other"]
                Mjob     = st.selectbox("Mother's job", _jobs, index=4)
                Fjob     = st.selectbox("Father's job", _jobs, index=4)
                guardian = st.selectbox("Guardian", ["mother", "father", "other"], index=0)
                reason   = st.selectbox(
                    "Reason for choosing this school",
                    ["home", "reputation", "course", "other"],
                    index=2,
                )

                st.subheader("School & Study")
                traveltime = st.slider("Travel time (1-4)",  1, 4, 1)
                studytime  = st.slider("Weekly study time (1-4)", 1, 4, 2)
                failures   = st.slider("Past class failures", 0, 3, 0)
                schoolsup  = st.selectbox("School extra support", ["yes", "no"], index=1)
                famsup     = st.selectbox("Family support",        ["yes", "no"], index=0)
                paid       = st.selectbox("Extra paid classes",    ["yes", "no"], index=1)
                activities = st.selectbox("Extra-curricular activities", ["yes", "no"], index=0)
                nursery    = st.selectbox("Attended nursery",      ["yes", "no"], index=0)
                higher     = st.selectbox("Wants higher education", ["yes", "no"], index=0)
                internet   = st.selectbox("Internet at home",      ["yes", "no"], index=0)
                romantic   = st.selectbox("In a romantic relationship", ["yes", "no"], index=1)

                st.subheader("Lifestyle")
                famrel   = st.slider("Family relationship quality (1-5)", 1, 5, 4)
                freetime = st.slider("Free time after school (1-5)",      1, 5, 3)
                goout    = st.slider("Going out with friends (1-5)",      1, 5, 3)
                Dalc     = st.slider("Workday alcohol use (1-5)",         1, 5, 1)
                Walc     = st.slider("Weekend alcohol use (1-5)",         1, 5, 1)
                health   = st.slider("Current health status (1-5)",       1, 5, 3)
                absences = st.slider("Number of school absences",         0, 93, 4)

                st.subheader("Grades & Course")
                G1     = st.slider("Period 1 grade (0-20)", 0, 20, 10)
                G2     = st.slider("Period 2 grade (0-20)", 0, 20, 10)
                course = st.selectbox("Course", ["math", "portuguese"], index=1)

                submitted = st.form_submit_button("🎯 Assess Student Risk", type="primary")

        # -------------------------------------------------------------- MAIN PANEL
        if not submitted:
            st.info(
                "👈 Fill in the student profile in the sidebar and click **Assess Student Risk**."
            )
        else:
            # 1. Build raw one-row DataFrame (mimicking raw CSV row)
            raw_input: dict = {
                "school": school, "sex": sex, "age": age, "address": address,
                "famsize": famsize, "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
                "Mjob": Mjob, "Fjob": Fjob, "guardian": guardian, "reason": reason,
                "traveltime": traveltime, "studytime": studytime, "failures": failures,
                "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
                "activities": activities, "nursery": nursery, "higher": higher,
                "internet": internet, "romantic": romantic, "famrel": famrel,
                "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc,
                "health": health, "absences": absences, "G1": G1, "G2": G2,
                "G3": 0,        # placeholder - excluded from features
                "course": course,
            }

            student_df_raw = pd.DataFrame([raw_input])

            # 2. clean_data encodes binary/categorical columns
            try:
                student_df_clean = _data_mod.clean_data(student_df_raw, config)
            except Exception as clean_err:
                st.error(f"Data cleaning error: {clean_err}")
                st.stop()

            # Drop target/G3 so columns match what the model was trained on
            exclude_cols = {"G3", "at_risk"}
            feature_cols = [c for c in student_df_clean.columns if c not in exclude_cols]
            X_student = student_df_clean[feature_cols]

            # 3. Predict
            if not model_ok:
                st.warning(
                    "⚠️ Model file not found at `models/best_model.joblib`. "
                    "Please run training first (`make train`)."
                )
            else:
                try:
                    prob_arr = pipeline.predict_proba(X_student)
                    prob = float(prob_arr[0, 1])
                    prediction = int(pipeline.predict(X_student)[0])
                except Exception as pred_err:
                    st.error(
                        f"Prediction error: {pred_err}\n\n"
                        "The saved model may have been trained on a different feature set. "
                        "Re-run `make train` to rebuild the model."
                    )
                    st.stop()

                # 4. Risk level display
                st.subheader("📈 Risk Assessment Result")
                col_metric, col_gauge = st.columns([1, 2])
                with col_metric:
                    risk_pct = int(prob * 100)
                    st.metric(
                        label="Dropout Risk Score",
                        value=f"{risk_pct}%",
                        delta=None,
                    )
                    if prob >= 0.7:
                        st.error(f"🔴 **HIGH RISK** - Immediate action needed ({risk_pct}%)")
                    elif prob >= 0.4:
                        st.warning(f"🟡 **MODERATE RISK** - Monitor closely ({risk_pct}%)")
                    else:
                        st.success(f"🟢 **LOW RISK** - Student appears on track ({risk_pct}%)")

                with col_gauge:
                    # Simple progress bar as a visual gauge
                    st.write("**Risk probability gauge**")
                    st.progress(prob)
                    st.caption(
                        f"Binary prediction: {'At-Risk' if prediction == 1 else 'Not At-Risk'}"
                    )

                # 5. SHAP chart
                st.subheader("🔍 Top Risk Drivers (SHAP)")
                if not _SHAP_AVAILABLE:
                    st.info(
                        "SHAP is not installed. Install it with `pip install shap` "
                        "to see feature importance."
                    )
                else:
                    try:
                        fig = _shap_bar_chart(pipeline, X_student, feature_cols, top_n=5)
                        st.pyplot(fig)
                        plt.close(fig)
                        st.caption(
                            "Red bars push risk **up** (toward at-risk); "
                            "blue bars push risk **down**."
                        )
                    except Exception as shap_err:
                        st.warning(
                            f"SHAP chart could not be generated: {shap_err}\n\n"
                            "This may happen with certain model types or versions."
                        )

                # 6. Counseling note
                st.subheader("💬 Counselor Recommendation")
                note = _counseling_note(prob, raw_input)
                st.markdown(note)

    # ==================================================================
    # TAB 2: Model Insights
    # ==================================================================
    with tab_insights:
        st.subheader("📊 Model Comparison")

        # --- Model comparison table
        if COMPARISON_PATH.exists():
            try:
                df_cmp = pd.read_csv(COMPARISON_PATH)
                st.dataframe(df_cmp, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load model_comparison.csv: {e}")
        else:
            st.info(
                "No model comparison report found yet. "
                "Run `make train` to generate `reports/model_comparison.csv`."
            )

        st.divider()

        # --- Metrics JSON
        st.subheader("🏆 Best Model Metrics")
        if METRICS_PATH.exists():
            try:
                with METRICS_PATH.open("r", encoding="utf-8") as fh:
                    metrics_data = json.load(fh)
                st.json(metrics_data)
            except Exception as e:
                st.warning(f"Could not load metrics.json: {e}")
        else:
            st.info(
                "No metrics file found yet. "
                "Run `make train` to generate `reports/metrics.json`."
            )

        st.divider()

        # --- Figures
        st.subheader("📉 Evaluation Plots")
        if FIGURES_DIR.exists():
            png_files = sorted(FIGURES_DIR.glob("*.png"))
            if png_files:
                for fig_path in png_files:
                    caption = fig_path.stem.replace("_", " ").title()
                    try:
                        st.image(str(fig_path), caption=caption, use_container_width=True)
                    except Exception as img_err:
                        st.warning(f"Could not display {fig_path.name}: {img_err}")
            else:
                st.info(
                    "No PNG plots found in `reports/figures/`. "
                    "Run `make train` to generate evaluation figures."
                )
        else:
            st.info(
                "`reports/figures/` directory does not exist yet. "
                "Run `make train` to create it."
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
