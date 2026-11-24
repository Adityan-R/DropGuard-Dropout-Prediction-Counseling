"""
app/streamlit_app.py
--------------------
DropGuard — Autonomous Student Dropout Risk Prediction & Counseling Engine
Cybernetic interface styled with standardized pixel typography and zero emojis.

Run from project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = str(PROJECT_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

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
    from dropguard import counseling as _counseling_mod
    from dropguard import explain as _explain_mod
    _LOCAL_IMPORTS_OK = True
except ImportError as _local_err:
    _LOCAL_IMPORTS_OK = False
    _LOCAL_ERR_MSG = str(_local_err)

# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
NOTES_PATH = REPORTS_DIR / "sample_counseling_notes.md"
THRESHOLD_PATH = REPORTS_DIR / "threshold_results.json"
ABLATION_PATH = REPORTS_DIR / "ablation_results.json"
REGRESSION_PATH = REPORTS_DIR / "regression_results.json"

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


@st.cache_data(show_spinner=False)
def _load_sample_notes() -> str:
    if NOTES_PATH.exists():
        try:
            return NOTES_PATH.read_text(encoding="utf-8")
        except Exception:
            return ""
    return ""


# ---------------------------------------------------------------------------
# SHAP visualization with Pixel / Dark Cybernetic Palette
# ---------------------------------------------------------------------------

def _shap_bar_chart(
    pipeline,
    student_df: pd.DataFrame,
    feature_names_raw: list[str],
    top_n: int = 5,
) -> plt.Figure:
    """Return a styled matplotlib Figure matching the pixel dark aesthetic."""
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.linear_model import LogisticRegression

    if hasattr(pipeline, "steps") and len(pipeline.steps) > 1:
        preprocessor = SkPipeline(pipeline.steps[:-1])
        X_transformed = preprocessor.transform(student_df)
    else:
        X_transformed = student_df.values

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_transformed = np.asarray(X_transformed, dtype=float)

    clf = pipeline[-1] if hasattr(pipeline, "__getitem__") else pipeline.steps[-1][1]

    if isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, X_transformed)
    else:
        explainer = shap.TreeExplainer(clf)

    shap_vals = explainer.shap_values(X_transformed)

    if isinstance(shap_vals, list):
        shap_vals = shap_vals[-1]
    shap_vals = np.asarray(shap_vals, dtype=float)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[..., -1]

    shap_row = np.asarray(shap_vals, dtype=float).ravel()

    try:
        preprocessor_step = pipeline.named_steps.get("preprocessor")
        if preprocessor_step is not None and hasattr(preprocessor_step, "get_feature_names_out"):
            feat_names = list(preprocessor_step.get_feature_names_out())
        else:
            feat_names = [f"feat_{i}" for i in range(len(shap_row))]
    except Exception:
        feat_names = [f"feat_{i}" for i in range(len(shap_row))]

    # Clean display names
    cleaned_names = []
    for fn in feat_names:
        clean = fn
        for prefix in ("num__", "cat__", "remainder__"):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        clean = clean.replace("_scaled", "").replace("_encoded", "").replace("_", " ").upper()
        cleaned_names.append(clean)

    abs_shap = np.abs(shap_row)
    top_idx = np.argsort(abs_shap)[-top_n:][::-1]

    top_names = [cleaned_names[i] if i < len(cleaned_names) else f"F{i}" for i in top_idx]
    top_values = [shap_row[i] for i in top_idx]

    # Plot styling
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor("#080D1A")
    ax.set_facecolor("#0A1124")

    # Neon colors: crimson for risk increase, cobalt blue for risk decrease
    colors = ["#F43F5E" if v > 0 else "#2563EB" for v in top_values]

    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_values[::-1], color=colors[::-1], height=0.55, edgecolor="rgba(255,255,255,0.15)", linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], color="#E2E8F0", fontsize=9.5, fontweight="bold", fontfamily="monospace")
    ax.tick_params(axis="x", colors="#94A3B8", labelsize=9)
    ax.axvline(0, color="#64748B", linewidth=1.0, linestyle="--", alpha=0.7)

    # Grid & borders
    ax.grid(axis="x", color="rgba(255,255,255,0.06)", linestyle="-", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color("rgba(255,255,255,0.12)")

    ax.set_xlabel("SHAP VALUE (<- PROTECTIVE | RISK ELEVATING ->)", color="#94A3B8", fontsize=9, labelpad=8, fontfamily="monospace")
    ax.set_title(f"TOP {top_n} MODEL DRIVERS [LOCAL SHAP VECTOR]", color="#F8FAFC", fontsize=10.5, fontweight="bold", pad=12, loc="left", fontfamily="monospace")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI Style Injection (Standardized Pixel Font, Cobalt Theme & Auto-Resize)
# ---------------------------------------------------------------------------

def _inject_custom_css():
    st.html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pixelify+Sans:wght@400;500;600;700&family=Silkscreen:wght@400;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography Reset with Pixelify Sans */
    html, body, .stApp, .stMarkdown, p, label, input, select, textarea {
        font-family: 'Pixelify Sans', monospace, sans-serif !important;
        font-size: 15.5px;
        letter-spacing: 0.02em;
    }

    /* Protect Streamlit icon ligatures, SVGs, and collapse buttons */
    [data-testid="stIconMaterial"],
    [data-testid="stIcon"],
    [data-testid="stExpanderToggleIcon"],
    [data-testid="stSidebarCollapseButton"],
    .material-symbols-rounded,
    .material-icons,
    [class*="material-symbols"],
    [class*="MaterialSymbols"],
    button[kind="header"] span,
    summary svg,
    summary span[aria-hidden="true"],
    [data-testid="stSidebarCollapseButton"] span,
    svg,
    svg * {
        font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
        font-style: normal !important;
        font-variant: normal !important;
        text-transform: none !important;
        line-height: 1 !important;
    }
    
    /* Code and Technical Tags */
    code, pre, .card-tag, .hero-badge, .hero-ticker {
        font-family: 'Silkscreen', 'JetBrains Mono', monospace !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6, .hero-title, .card-val {
        font-family: 'Pixelify Sans', 'Silkscreen', monospace !important;
        letter-spacing: 0.03em;
    }
    
    .stApp {
        background-color: #040711;
        color: #F8FAFC;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ========================================================= */
    /* CUSTOM DESIGNED SCROLLBAR & MENU (Cobalt Glow)            */
    /* ========================================================= */
    ::-webkit-scrollbar {
        width: 7px;
        height: 7px;
    }
    ::-webkit-scrollbar-track {
        background: #040711;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #1E40AF 0%, #2563EB 50%, #3B82F6 100%);
        border-radius: 4px;
        border: 1px solid rgba(96, 165, 250, 0.3);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #2563EB 0%, #3B82F6 50%, #60A5FA 100%);
        box-shadow: 0 0 10px rgba(37, 99, 235, 0.7);
    }

    /* Custom Designed Dropdown Popover & Select Scroll Menu */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    ul[role="listbox"] {
        background-color: #0A1124 !important;
        border: 1px solid rgba(37, 99, 235, 0.45) !important;
        border-radius: 6px !important;
        box-shadow: 0 12px 35px -5px rgba(0, 0, 0, 0.8), 0 0 20px rgba(37, 99, 235, 0.25) !important;
        max-height: 280px !important;
    }

    li[role="option"] {
        background: transparent !important;
        color: #E2E8F0 !important;
        font-family: 'Pixelify Sans', monospace !important;
        font-size: 0.88rem !important;
        padding: 0.55rem 0.85rem !important;
        transition: background 0.15s ease, color 0.15s ease !important;
        border-left: 2px solid transparent !important;
    }

    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background: rgba(37, 99, 235, 0.22) !important;
        color: #FFFFFF !important;
        border-left: 2px solid #3B82F6 !important;
    }

    /* ========================================================= */
    /* CLEAN SIDEBAR & RESPONSIVE FLUID HOMEPAGE LAYOUT          */
    /* ========================================================= */

    /* Style the sidebar content with the dark cobalt theme without overriding layout mechanics */
    [data-testid="stSidebarContent"] {
        background-color: #080D1A !important;
        border-right: 1px solid rgba(37, 99, 235, 0.3) !important;
    }

    /* Style the collapse button inside sidebar */
    [data-testid="stSidebarCollapseButton"] button {
        background: rgba(10, 17, 36, 0.85) !important;
        border: 1px solid rgba(37, 99, 235, 0.45) !important;
        border-radius: 4px !important;
        color: #60A5FA !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stSidebarCollapseButton"] button:hover {
        background: rgba(37, 99, 235, 0.3) !important;
        border-color: #60A5FA !important;
        box-shadow: 0 0 10px rgba(37, 99, 235, 0.5) !important;
    }

    /* Style the expand button when sidebar is collapsed */
    [data-testid="stExpandSidebarButton"] button {
        background: rgba(10, 17, 36, 0.95) !important;
        border: 1px solid rgba(37, 99, 235, 0.55) !important;
        border-radius: 4px !important;
        color: #60A5FA !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stExpandSidebarButton"] button:hover {
        background: rgba(37, 99, 235, 0.35) !important;
        border-color: #93C5FD !important;
        box-shadow: 0 0 14px rgba(37, 99, 235, 0.6) !important;
    }

    /* Homepage full-bleed width & smooth responsive scaling */
    .block-container {
        max-width: 100% !important;
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        box-sizing: border-box !important;
        transition: padding 0.3s cubic-bezier(0.4, 0, 0.2, 1), max-width 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    /* Smooth interior card transitions on width changes */
    .hero-container {
        transition: padding 0.3s ease, margin 0.3s ease !important;
    }

    .metric-grid-card {
        transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease !important;
    }

    /* Fix Expander in Sidebar so text and icon never overlap */
    details summary {
        display: flex !important;
        align-items: center !important;
        gap: 0.6rem !important;
        padding: 0.6rem 0.75rem !important;
        line-height: 1.35 !important;
        background: rgba(10, 17, 36, 0.65) !important;
        border-radius: 6px !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        margin-top: 0.6rem !important;
        margin-bottom: 0.8rem !important;
    }
    
    details summary:hover {
        border-color: rgba(37, 99, 235, 0.5) !important;
    }
    
    details summary p {
        margin: 0 !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        white-space: normal !important;
    }
    
    /* Hero Banner with Cobalt Digital Wave Gradient */
    .hero-container {
        position: relative;
        overflow: hidden;
        background: 
            radial-gradient(110% 120% at 50% 0%, rgba(37, 99, 235, 0.42) 0%, rgba(30, 64, 175, 0.22) 35%, rgba(4, 7, 17, 0.98) 75%),
            repeating-linear-gradient(0deg, rgba(255, 255, 255, 0.032) 0px, rgba(255, 255, 255, 0.032) 1px, transparent 1px, transparent 5px),
            linear-gradient(180deg, #0A142F 0%, #040711 100%);
        border: 1px solid rgba(37, 99, 235, 0.38);
        border-radius: 10px;
        padding: 2.8rem 2.2rem 2rem 2.2rem;
        margin-bottom: 2rem;
        box-shadow: 0 16px 45px -12px rgba(37, 99, 235, 0.38);
    }
    
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(37, 99, 235, 0.16);
        border: 1px solid rgba(96, 165, 250, 0.45);
        color: #BFDBFE;
        font-size: 0.72rem !important;
        font-weight: 700;
        letter-spacing: 0.08em;
        padding: 0.3rem 0.8rem;
        border-radius: 4px;
        margin-bottom: 1.1rem;
        text-transform: uppercase;
    }
    
    .hero-title {
        font-size: 2.35rem !important;
        font-weight: 700;
        line-height: 1.2;
        color: #FFFFFF;
        margin: 0 0 0.9rem 0;
        text-shadow: 0 2px 14px rgba(255, 255, 255, 0.12);
    }
    
    .hero-sub {
        font-size: 1.02rem !important;
        color: #94A3B8;
        max-width: 820px;
        line-height: 1.6;
        margin-bottom: 1.8rem;
    }
    
    .hero-ticker {
        display: flex;
        flex-wrap: wrap;
        gap: 1.2rem 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 1.2rem;
        font-size: 0.74rem !important;
        color: #94A3B8;
    }
    
    .hero-ticker strong {
        color: #F8FAFC;
    }

    /* Telemetry grid cards */
    .metric-grid-card {
        background: rgba(10, 17, 36, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 8px;
        padding: 1.15rem 1.3rem;
        height: 100%;
        backdrop-filter: blur(10px);
        transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
        cursor: help;
        position: relative;
    }
    
    .metric-grid-card:hover {
        border-color: rgba(96, 165, 250, 0.75) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 28px -4px rgba(37, 99, 235, 0.45) !important;
    }

    /* Custom Cursor-Following Floating Tooltip */
    .cursor-tooltip {
        position: fixed;
        top: 0;
        left: 0;
        pointer-events: none;
        z-index: 999999999;
        opacity: 0;
        transform: translate3d(-9999px, -9999px, 0);
        transition: opacity 0.14s cubic-bezier(0.4, 0, 0.2, 1);
        background: rgba(6, 10, 22, 0.95);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(59, 130, 246, 0.65);
        border-radius: 8px;
        padding: 0.9rem 1.15rem;
        max-width: 320px;
        width: 310px;
        box-shadow: 0 16px 36px -4px rgba(0, 0, 0, 0.85), 0 0 20px rgba(37, 99, 235, 0.4);
        color: #F8FAFC;
        line-height: 1.45;
        will-change: transform, opacity;
    }

    .cursor-tooltip .tooltip-header {
        font-size: 0.70rem !important;
        color: #60A5FA !important;
        letter-spacing: 0.08em;
        font-family: 'Silkscreen', monospace !important;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    .cursor-tooltip .tooltip-title {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        font-family: 'Pixelify Sans', monospace !important;
        letter-spacing: 0.02em;
        margin-bottom: 0.45rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        padding-bottom: 0.35rem;
    }

    .cursor-tooltip .tooltip-desc {
        font-size: 0.83rem !important;
        color: #CBD5E1 !important;
        font-family: 'Pixelify Sans', monospace !important;
        line-height: 1.45 !important;
    }

    .cursor-tooltip .tooltip-stat {
        margin-top: 0.5rem;
        padding-top: 0.35rem;
        border-top: 1px dashed rgba(59, 130, 246, 0.45);
        font-size: 0.72rem !important;
        color: #93C5FD !important;
        font-family: 'Silkscreen', monospace !important;
        letter-spacing: 0.04em;
    }
    
    .card-tag {
        font-size: 0.68rem !important;
        color: #60A5FA;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    
    .card-val {
        font-size: 1.55rem !important;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
    }
    
    .card-desc {
        font-size: 0.82rem !important;
        color: #94A3B8;
        line-height: 1.45;
    }

    /* Risk Status Banners */
    .risk-banner {
        padding: 1.3rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        border: 1px solid transparent;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .risk-banner.high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(10, 17, 36, 0.85) 100%);
        border-color: rgba(239, 68, 68, 0.45);
    }
    
    .risk-banner.med {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.18) 0%, rgba(10, 17, 36, 0.85) 100%);
        border-color: rgba(245, 158, 11, 0.45);
    }
    
    .risk-banner.low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18) 0%, rgba(10, 17, 36, 0.85) 100%);
        border-color: rgba(16, 185, 129, 0.45);
    }

    /* Buttons */
    .stButton>button {
        font-family: 'Silkscreen', 'Pixelify Sans', monospace !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.05em;
        border-radius: 4px !important;
        padding: 0.55rem 1rem !important;
        background: linear-gradient(180deg, #2563EB 0%, #1D4ED8 100%) !important;
        border: 1px solid #3B82F6 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3) !important;
    }
    .stButton>button:hover {
        background: linear-gradient(180deg, #3B82F6 0%, #2563EB 100%) !important;
        border-color: #60A5FA !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
    }
    
    /* Tabs styling */
    button[data-baseweb="tab"] {
        font-family: 'Silkscreen', 'Pixelify Sans', monospace !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.04em;
    }
    </style>
    """)

    # Cursor-following description popup script
    st.html("""
    <script>
    (function() {
        function setupDropGuardTooltip() {
            let tooltip = document.getElementById('dropguard-cursor-tooltip');
            if (!tooltip) {
                tooltip = document.createElement('div');
                tooltip.id = 'dropguard-cursor-tooltip';
                tooltip.className = 'cursor-tooltip';
                tooltip.setAttribute('aria-hidden', 'true');
                tooltip.innerHTML = `
                    <div class="tooltip-header" id="dg-tt-header"></div>
                    <div class="tooltip-title" id="dg-tt-title"></div>
                    <div class="tooltip-desc" id="dg-tt-desc"></div>
                    <div class="tooltip-stat" id="dg-tt-stat"></div>
                `;
                document.body.appendChild(tooltip);
            }

            const ttHeader = document.getElementById('dg-tt-header');
            const ttTitle = document.getElementById('dg-tt-title');
            const ttDesc = document.getElementById('dg-tt-desc');
            const ttStat = document.getElementById('dg-tt-stat');

            let isHovering = false;
            let currentX = 0, currentY = 0;

            function updatePosition() {
                if (!isHovering || !tooltip) return;
                const offset = 18;
                let targetX = currentX + offset;
                let targetY = currentY + offset;

                const rect = tooltip.getBoundingClientRect();
                const winW = window.innerWidth;
                const winH = window.innerHeight;

                if (targetX + rect.width > winW - 14) {
                    targetX = currentX - rect.width - 14;
                }
                if (targetY + rect.height > winH - 14) {
                    targetY = currentY - rect.height - 14;
                }
                if (targetX < 10) targetX = 10;
                if (targetY < 10) targetY = 10;

                tooltip.style.transform = 'translate3d(' + targetX + 'px, ' + targetY + 'px, 0)';
            }

            document.addEventListener('mouseover', function(e) {
                const card = e.target.closest('.metric-grid-card');
                if (card) {
                    const tag = card.getAttribute('data-tooltip-tag') || 'TELEMETRY DIAGNOSTIC';
                    const title = card.getAttribute('data-tooltip-title') || '';
                    const desc = card.getAttribute('data-tooltip-desc') || '';
                    const stat = card.getAttribute('data-tooltip-stat') || '';

                    if (ttHeader) ttHeader.textContent = tag;
                    if (ttTitle) ttTitle.textContent = title;
                    if (ttDesc) ttDesc.textContent = desc;
                    if (ttStat) {
                        if (stat) {
                            ttStat.textContent = stat;
                            ttStat.style.display = 'block';
                        } else {
                            ttStat.style.display = 'none';
                        }
                    }

                    isHovering = true;
                    tooltip.style.opacity = '1';
                    updatePosition();
                }
            });

            document.addEventListener('mousemove', function(e) {
                currentX = e.clientX;
                currentY = e.clientY;
                if (isHovering) {
                    updatePosition();
                }
            });

            document.addEventListener('mouseout', function(e) {
                const card = e.target.closest('.metric-grid-card');
                if (card && (!e.relatedTarget || !card.contains(e.relatedTarget))) {
                    isHovering = false;
                    if (tooltip) tooltip.style.opacity = '0';
                }
            });
        }

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', setupDropGuardTooltip);
        } else {
            setupDropGuardTooltip();
        }
    })();
    </script>
    """, unsafe_allow_javascript=True)


# ---------------------------------------------------------------------------
# Presets Data for Fast Inspection (Clean Titles, No Emojis)
# ---------------------------------------------------------------------------
PRESETS = {
    "Custom (Manual Input)": None,
    "High Risk: Student #150 (G2 Drop, Prior Failures)": {
        "school": "GP", "sex": "M", "age": 18, "address": "U", "famsize": "LE3", "Pstatus": "T",
        "Medu": 2, "Fedu": 1, "Mjob": "services", "Fjob": "other", "guardian": "mother", "reason": "course",
        "traveltime": 2, "studytime": 1, "failures": 2, "schoolsup": "no", "famsup": "yes", "paid": "no",
        "activities": "no", "nursery": "yes", "higher": "no", "internet": "yes", "romantic": "no",
        "famrel": 3, "freetime": 4, "goout": 4, "Dalc": 2, "Walc": 3, "health": 3, "absences": 12,
        "G1": 6, "G2": 5, "course": "math"
    },
    "High Risk: Student #18 (High Absences, Chronic Struggles)": {
        "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3", "Pstatus": "T",
        "Medu": 3, "Fedu": 2, "Mjob": "services", "Fjob": "services", "guardian": "mother", "reason": "course",
        "traveltime": 1, "studytime": 1, "failures": 1, "schoolsup": "yes", "famsup": "yes", "paid": "no",
        "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
        "famrel": 5, "freetime": 3, "goout": 4, "Dalc": 2, "Walc": 4, "health": 5, "absences": 22,
        "G1": 7, "G2": 6, "course": "portuguese"
    },
    "Moderate Risk: Student #10 (Borderline G2, Moderate Engagement)": {
        "school": "GP", "sex": "F", "age": 16, "address": "U", "famsize": "GT3", "Pstatus": "T",
        "Medu": 3, "Fedu": 4, "Mjob": "services", "Fjob": "other", "guardian": "mother", "reason": "home",
        "traveltime": 1, "studytime": 2, "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "yes",
        "activities": "no", "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
        "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 2, "health": 4, "absences": 6,
        "G1": 10, "G2": 9, "course": "math"
    },
    "Low Risk: Student #42 (High Study Time, Strong Grades)": {
        "school": "GP", "sex": "F", "age": 16, "address": "U", "famsize": "GT3", "Pstatus": "T",
        "Medu": 4, "Fedu": 4, "Mjob": "teacher", "Fjob": "services", "guardian": "father", "reason": "reputation",
        "traveltime": 1, "studytime": 3, "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
        "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
        "famrel": 5, "freetime": 2, "goout": 2, "Dalc": 1, "Walc": 1, "health": 4, "absences": 2,
        "G1": 16, "G2": 16, "course": "portuguese"
    },
}


# ---------------------------------------------------------------------------
# App Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="DropGuard // Autonomous Student Risk Core",
        page_icon=":material/shield:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_custom_css()

    if not _CORE_IMPORTS_OK:
        st.error(f"Missing core dependency: {_CORE_ERR_MSG}")
        st.stop()

    if not _LOCAL_IMPORTS_OK:
        st.error(
            f"Could not import local modules: {_LOCAL_ERR_MSG}\n\n"
            "Run from the project root: `streamlit run app/streamlit_app.py`"
        )
        st.stop()

    config = _load_config()
    pipeline, model_ok = _load_model()

    # ------------------------------------------------------------------ HERO
    st.html("""
    <div class="hero-container">
        <div class="hero-badge">INT395 // ML INFERENCE CORE • RANDOM FOREST (AUC 0.963)</div>
        <h1 class="hero-title">Dropout risk, predicted before it happens.</h1>
        <p class="hero-sub">
            DropGuard continuously evaluates academic trajectories and socioeconomic drivers across secondary students.
            By computing localized SHAP influence vectors, it surfaces high-risk attrition candidates and auto-generates
            tailored intervention protocols for counselors.
        </p>
        <div class="hero-ticker">
            <span><strong>COHORT:</strong> 1,044 UCI STUDENTS (MATH + POR)</span>
            <span><strong>HELD-OUT ACCURACY:</strong> 90.0%</span>
            <span><strong>TUNED RECALL:</strong> 100% (46/46 CAUGHT)</span>
            <span><strong>DECISION BOUNDARY:</strong> 0.338</span>
            <span><strong>STATUS:</strong> ONLINE</span>
        </div>
    </div>
    """)

    # ------------------------------------------------------------------ METRIC TILES
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.html("""
        <div class="metric-grid-card"
             data-tooltip-tag="01 // ARCHITECTURE TELEMETRY"
             data-tooltip-title="Random Forest Classifier"
             data-tooltip-desc="Trained under 5-Fold Stratified Cross-Validation on 835 training students. Uses 100 decision trees constrained with max_depth=8 and min_samples_leaf=2 to curb overfitting, with balanced class weights applied to resolve the 78:22 pass-to-risk imbalance."
             data-tooltip-stat="CV RECALL: 91.3% // CV ROC-AUC: 0.965">
            <div class="card-tag">01 // CLASSIFIER ARCHITECTURE</div>
            <div class="card-val">Random Forest</div>
            <div class="card-desc">Stratified 5-Fold GridCV with balanced class weights & early-pruned estimators.</div>
        </div>
        """)
    with col_m2:
        st.html("""
        <div class="metric-grid-card"
             data-tooltip-tag="02 // DISCRIMINATION POWER"
             data-tooltip-title="Held-Out Test ROC-AUC: 0.963"
             data-tooltip-desc="Evaluated strictly on the held-out test cohort of 209 unseen students. An AUC of 0.963 proves near-optimal class separation and rank-ordering between passing and at-risk students across all classification thresholds."
             data-tooltip-stat="TEST COHORT: 209 STUDENTS // CLASS SEPARATION: 96.3%">
            <div class="card-tag">02 // TEST ROC-AUC</div>
            <div class="card-val">0.963</div>
            <div class="card-desc">Class separation power on held-out test cohort (209 unseen students).</div>
        </div>
        """)
    with col_m3:
        st.html("""
        <div class="metric-grid-card"
             data-tooltip-tag="03 // RETENTION OPTIMIZATION"
             data-tooltip-title="100.0% At-Risk Sensitivity"
             data-tooltip-desc="Optimized via out-of-fold F2-score threshold tuning (tau = 0.338). At this decision boundary, the model successfully caught all 46 out of 46 at-risk students in the unseen test set with 0 False Negatives, ensuring no student is left without support."
             data-tooltip-stat="TUNED THRESHOLD: 0.338 // ZERO FALSE NEGATIVES (46/46)">
            <div class="card-tag">03 // AT-RISK SENSITIVITY</div>
            <div class="card-val">100.0%</div>
            <div class="card-desc">Recall achieved at OOF-optimized F2 threshold (0.338). Zero false negatives.</div>
        </div>
        """)
    with col_m4:
        st.html("""
        <div class="metric-grid-card"
             data-tooltip-tag="04 // EXPLAINABLE AI & SYNTHESIS"
             data-tooltip-title="TreeSHAP Intervention Engine"
             data-tooltip-desc="Computes exact Shapley attribution vectors for every individual student prediction. The counseling synthesizer deterministically maps the top 3 adverse drivers (e.g., low G2 scores, prior failures, high unexcused absences) into structured, actionable advising protocols."
             data-tooltip-stat="LOCAL SHAP DECOMPOSITION // ZERO GENERATIVE HALLUCINATIONS">
            <div class="card-tag">04 // INTERVENTION ENGINE</div>
            <div class="card-val">SHAP Driver</div>
            <div class="card-desc">Rule-based translation of top 3 Shapley vectors into academic counseling briefs.</div>
        </div>
        """)

    st.write("")
    st.write("")

    # ------------------------------------------------------------------ TABS
    tab_assess, tab_insights, tab_dossiers = st.tabs(
        ["Live Inference & Risk Telemetry", "Model Benchmarks & Ablation", "Test Case Dossiers"]
    )

    # ==================================================================
    # TAB 1: Live Inference & Risk Telemetry
    # ==================================================================
    with tab_assess:
        # Sidebar Controls
        with st.sidebar:
            st.header("Telemetry Inputs")
            
            # Preset selector
            selected_preset_name = st.selectbox(
                "Quick Load Profile Preset",
                list(PRESETS.keys()),
                index=0,
                help="Select a benchmark student to test immediate risk scoring."
            )
            preset_vals = PRESETS[selected_preset_name]

            with st.form("student_profile_form"):
                st.caption("Primary Academic Drivers (Highest SHAP Weights)")

                def _val(key, default):
                    if preset_vals and key in preset_vals:
                        return preset_vals[key]
                    return default

                col1, col2 = st.columns(2)
                with col1:
                    G1 = st.slider("Period 1 Grade", 0, 20, _val("G1", 10))
                    studytime = st.slider("Study Time (1-4)", 1, 4, _val("studytime", 2))
                    failures = st.slider("Past Failures", 0, 3, _val("failures", 0))
                    course = st.selectbox("Course", ["math", "portuguese"], index=0 if _val("course", "portuguese") == "math" else 1)
                with col2:
                    G2 = st.slider("Period 2 Grade", 0, 20, _val("G2", 10))
                    absences = st.slider("Absences", 0, 93, _val("absences", 4))
                    higher = st.selectbox("Higher Ed Intent", ["yes", "no"], index=0 if _val("higher", "yes") == "yes" else 1)
                    sex = st.selectbox("Sex", ["F", "M"], index=0 if _val("sex", "F") == "F" else 1)

                age = st.slider("Age", 15, 22, _val("age", 17))

                with st.expander("More Student Attributes", expanded=False):
                    school = st.selectbox("School Campus", ["GP", "MS"], index=0 if _val("school", "GP") == "GP" else 1)
                    address = st.selectbox("Address Setting", ["U", "R"], index=0 if _val("address", "U") == "U" else 1)
                    famsize = st.selectbox("Family Size", ["GT3", "LE3"], index=0 if _val("famsize", "GT3") == "GT3" else 1)
                    Pstatus = st.selectbox("Parent Cohabitation", ["T", "A"], index=0 if _val("Pstatus", "T") == "T" else 1)
                    Medu = st.slider("Mother's Education (0-4)", 0, 4, _val("Medu", 2))
                    Fedu = st.slider("Father's Education (0-4)", 0, 4, _val("Fedu", 2))
                    _jobs = ["teacher", "health", "services", "at_home", "other"]
                    Mjob = st.selectbox("Mother's Occupation", _jobs, index=_jobs.index(_val("Mjob", "other")))
                    Fjob = st.selectbox("Father's Occupation", _jobs, index=_jobs.index(_val("Fjob", "other")))
                    guardian = st.selectbox("Primary Guardian", ["mother", "father", "other"], index=["mother", "father", "other"].index(_val("guardian", "mother")))
                    reason = st.selectbox("Enrollment Reason", ["home", "reputation", "course", "other"], index=["home", "reputation", "course", "other"].index(_val("reason", "course")))
                    traveltime = st.slider("Commute Time (1-4)", 1, 4, _val("traveltime", 1))
                    schoolsup = st.selectbox("School Academic Support", ["yes", "no"], index=0 if _val("schoolsup", "no") == "yes" else 1)
                    famsup = st.selectbox("Family Educational Support", ["yes", "no"], index=0 if _val("famsup", "yes") == "yes" else 1)
                    paid = st.selectbox("Paid Supplementary Classes", ["yes", "no"], index=0 if _val("paid", "no") == "yes" else 1)
                    activities = st.selectbox("Extracurricular Engagement", ["yes", "no"], index=0 if _val("activities", "yes") == "yes" else 1)
                    nursery = st.selectbox("Attended Nursery", ["yes", "no"], index=0 if _val("nursery", "yes") == "yes" else 1)
                    internet = st.selectbox("High-Speed Internet Access", ["yes", "no"], index=0 if _val("internet", "yes") == "yes" else 1)
                    romantic = st.selectbox("In Romantic Relationship", ["yes", "no"], index=0 if _val("romantic", "no") == "yes" else 1)
                    famrel = st.slider("Family Relationship Quality (1-5)", 1, 5, _val("famrel", 4))
                    freetime = st.slider("Post-School Free Time (1-5)", 1, 5, _val("freetime", 3))
                    goout = st.slider("Socializing / Going Out (1-5)", 1, 5, _val("goout", 3))
                    Dalc = st.slider("Workday Alcohol Consumption (1-5)", 1, 5, _val("Dalc", 1))
                    Walc = st.slider("Weekend Alcohol Consumption (1-5)", 1, 5, _val("Walc", 1))
                    health = st.slider("Self-Reported Health (1-5)", 1, 5, _val("health", 3))

                st.write("")
                submitted = st.form_submit_button("Run Risk Assessment & Diagnostics", type="primary", use_container_width=True)

        # Main Telemetry Viewport
        should_run = submitted or (selected_preset_name != "Custom (Manual Input)")

        if not should_run:
            st.info(
                "Adjust telemetry controls or pick a **Quick Load Preset** in the sidebar, then click **Run Risk Assessment**."
            )
        else:
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
                "G3": 0,
                "course": course,
            }

            student_df_raw = pd.DataFrame([raw_input])

            try:
                student_df_clean = _data_mod.clean_data(student_df_raw, config)
            except Exception as clean_err:
                st.error(f"Data cleaning pipeline exception: {clean_err}")
                st.stop()

            exclude_cols = {"G3", "at_risk"}
            feature_cols = [c for c in student_df_clean.columns if c not in exclude_cols]
            X_student = student_df_clean[feature_cols]

            if not model_ok:
                st.warning("Serialized model missing at `models/best_model.joblib`. Execute `make train` first.")
            else:
                try:
                    prob_arr = pipeline.predict_proba(X_student)
                    prob = float(prob_arr[0, 1])
                    prediction = int(pipeline.predict(X_student)[0])
                except Exception as pred_err:
                    st.error(f"Prediction pipeline error: {pred_err}")
                    st.stop()

                # Assessment Outcome Banner
                risk_pct = int(prob * 100)
                if prob >= 0.6:
                    banner_cls = "high"
                    status_title = "CRITICAL ATTRITION RISK"
                    status_sub = "Immediate advisory intervention and multi-stakeholder follow-up strongly advised."
                    badge_color = "#F43F5E"
                elif prob >= 0.3:
                    banner_cls = "med"
                    status_title = "ELEVATED RISK TRAJECTORY"
                    status_sub = "Student exhibits early friction indicators; proactive academic check-in recommended."
                    badge_color = "#F59E0B"
                else:
                    banner_cls = "low"
                    status_title = "NOMINAL ENGAGEMENT (ON TRACK)"
                    status_sub = "Low attrition probability. Continue routine mid-semester academic monitoring."
                    badge_color = "#10B981"

                st.html(f"""
                <div class="risk-banner {banner_cls}">
                    <div>
                        <div style="font-family:'Silkscreen',monospace; font-size:0.75rem; color:{badge_color}; font-weight:700; letter-spacing:0.08em;">
                            DIAGNOSTIC STATUS // {status_title}
                        </div>
                        <div style="font-size:1.45rem; font-weight:700; color:#FFFFFF; margin-top:0.2rem;">
                            Predicted Dropout Probability: {risk_pct}%
                        </div>
                        <div style="font-size:0.85rem; color:#CBD5E1; margin-top:0.3rem;">
                            {status_sub}
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <span style="font-family:'Silkscreen',monospace; font-size:2.2rem; font-weight:700; color:{badge_color};">
                            {prob:.3f}
                        </span>
                        <div style="font-size:0.75rem; color:#94A3B8;">CLASSIFICATION CUTOFF: 0.338</div>
                    </div>
                </div>
                """)

                # Visual split: SHAP chart + Counselor dossier
                col_left, col_right = st.columns([1.1, 0.9])

                with col_left:
                    st.subheader("Local Feature Attribution (SHAP)")
                    if not _SHAP_AVAILABLE:
                        st.info("SHAP package not found. Feature attributions disabled.")
                    else:
                        try:
                            fig = _shap_bar_chart(pipeline, X_student, feature_cols, top_n=5)
                            st.pyplot(fig)
                            plt.close(fig)
                            st.caption("Magenta bars exert upward pressure toward attrition; violet bars indicate stabilizing academic factors.")
                        except Exception as shap_err:
                            st.warning(f"SHAP attribution notice: {shap_err}")

                with col_right:
                    st.subheader("Counselor Action Brief")
                    top_drivers = []
                    try:
                        feat_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
                        shap_vals, _ = _explain_mod.compute_shap_values(pipeline, X_student, feat_names)
                        top_drivers = _explain_mod.get_top_drivers(shap_vals[0], feat_names, top_n=3)
                    except Exception:
                        top_drivers = []

                    note = _counseling_mod.generate_note(
                        student_idx=0, probability=prob, top_drivers=top_drivers
                    )

                    with st.container(border=True):
                        st.markdown(note)

    # ==================================================================
    # TAB 2: Model Benchmarks & Ablation
    # ==================================================================
    with tab_insights:
        st.subheader("Cross-Validation Model Matrix (5-Fold Stratified)")

        if COMPARISON_PATH.exists():
            try:
                df_cmp = pd.read_csv(COMPARISON_PATH)
                st.dataframe(df_cmp, use_container_width=True)
            except Exception as e:
                st.warning(f"Unable to parse model comparison: {e}")

        st.divider()

        # Telemetry columns: Metrics JSON & Threshold Tuning
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("Optimal Decision Boundary Tuning")
            if THRESHOLD_PATH.exists():
                try:
                    with THRESHOLD_PATH.open("r", encoding="utf-8") as fh:
                        thresh_data = json.load(fh)
                    st.json(thresh_data)
                except Exception as e:
                    st.warning(f"Threshold data missing: {e}")
            else:
                st.info("No threshold tuning results found.")

        with col_t2:
            st.subheader("Test Evaluation Payload")
            if METRICS_PATH.exists():
                try:
                    with METRICS_PATH.open("r", encoding="utf-8") as fh:
                        metrics_data = json.load(fh)
                    st.json(metrics_data)
                except Exception as e:
                    st.warning(f"Metrics payload error: {e}")
            else:
                st.info("No metrics file found.")

        st.divider()

        # Research Plots Gallery
        st.subheader("Experiment Research Figures")
        if FIGURES_DIR.exists():
            png_files = sorted(FIGURES_DIR.glob("*.png"))
            if png_files:
                cols = st.columns(2)
                for idx, fig_path in enumerate(png_files):
                    caption = fig_path.stem.replace("_", " ").title()
                    with cols[idx % 2]:
                        st.image(str(fig_path), caption=f"Figure: {caption}", use_container_width=True)
            else:
                st.info("No PNG figures located in `reports/figures/`.")

    # ==================================================================
    # TAB 3: Test Case Dossiers
    # ==================================================================
    with tab_dossiers:
        st.subheader("Test Cohort Case Studies & Sample Notes")
        st.markdown(
            "Auto-generated intelligence notes for the **top 10 highest-risk students** evaluated on the unseen test set."
        )

        sample_notes_md = _load_sample_notes()
        if sample_notes_md:
            with st.container(border=True):
                st.markdown(sample_notes_md)
        else:
            st.info("`reports/sample_counseling_notes.md` not found. Execute `python -m src.dropguard.train`.")


if __name__ == "__main__":
    main()
