from html import escape
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from analysis_functions.constants import (
    ALPHA, DOCUMENTATION_METRIC_COLUMNS, DOCUMENTATION_QUALITY_SCORE_COLUMN,
    H0, H1, MAINTENANCE_SCORE_COLUMN, MAINTENANCE_VARIABLE_COLUMNS,
    RESEARCH_QUESTION, RESEARCH_TITLE, readable_name,
)
from analysis_functions.hypothesis_evaluation import format_number
from visualization_charts.correlation_charts import (
    create_focused_correlation_heatmap, create_full_correlation_heatmap,
    create_ranked_correlation_bar_chart,
)
from visualization_charts.regression_charts import create_regression_scatter_plot

TAB_NAMES = [
    "Research Overview", "Dataset Overview", "Correlation Analysis",
    "Regression Analysis", "Hypothesis Evaluation",
]

def apply_dashboard_style(): 
    st.markdown(""" 
    <style> 
      .stApp {
        --dashboard-canvas: #F5F6F3;
        --dashboard-surface: #FFFFFF;
        --dashboard-surface-tint: #F7FAF9;
        --dashboard-sidebar: #EAF0ED;
        --dashboard-ink: #183247;
        --dashboard-muted: #526775;
        --dashboard-border: #D8E3DF;
        --dashboard-accent: #0F766E;
        --dashboard-accent-dark: #115E59;
        --dashboard-accent-soft: #E3F1EB;
        --dashboard-accent-border: #BCD8CD;
        --dashboard-shadow: #1832470A;
        background: var(--dashboard-canvas);
        color: var(--dashboard-ink);
        font-family: Arial, sans-serif;
      }
      .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp input, 
      .stApp button, .stApp textarea, .stApp select, .stApp label, 
      .stApp [data-testid="stMarkdownContainer"] { font-family: Arial, sans-serif; } 

      [data-testid="stHeader"] { visibility: hidden; height: 0; background: transparent; } 
      [data-testid="stDecoration"], #MainMenu, footer { display: none; } 

      [data-testid="stToolbar"] { pointer-events: none; } 

      [data-testid="stExpandSidebarButton"] { 
        visibility: visible; pointer-events: auto; position: fixed; top: 10px; left: 10px; 
        background: var(--dashboard-surface); border: 1px solid var(--dashboard-border); border-radius: 8px; 
      } 

      .block-container { max-width: 1480px; padding-top: 1.5rem; padding-bottom: 4rem; } 

      h1, h2, h3 { color: var(--dashboard-ink); letter-spacing: -.02em; } 

      [data-testid="stSidebar"] { 
        background: var(--dashboard-sidebar); 
        border-right: 1px solid var(--dashboard-border); 
      } 

      [data-testid="stSidebar"] h2 { 
        font-size: 1.1rem; 
        color: var(--dashboard-accent-dark); 
      } 

      [data-testid="stCaptionContainer"] { color: var(--dashboard-muted); } 

      [data-testid="stTabs"] [role="tablist"] {
        gap: 1.5rem;
        padding: 0 0 12px; 
      } 

      [data-testid="stTabs"] [data-baseweb="tab"] { 
        background: var(--dashboard-surface); 
        border: 1px solid var(--dashboard-border); 
        border-radius: 10px; 
        height: auto; 
        padding: 12px 18px; 
        color: var(--dashboard-muted); 
        font-weight: 600; 
      } 

      [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] { 
        background: var(--dashboard-accent); 
        border-color: var(--dashboard-accent); 
        color: var(--dashboard-surface); 
      } 

      [data-testid="stTabs"] [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background: var(--dashboard-accent-soft);
        border-color: var(--dashboard-accent-border);
        color: var(--dashboard-accent-dark);
      }

      [data-testid="stTabs"] [data-baseweb="tab"]:focus-visible {
        outline: 2px solid var(--dashboard-accent);
        outline-offset: 3px;
      }

      [data-testid="stTabs"] [data-baseweb="tab-highlight"] { 
        background: var(--dashboard-accent-dark); 
      } 

      [data-testid="stTabs"] [role="tabpanel"] { padding-top: 12px; } 

      [data-testid="stExpander"] { 
        background: var(--dashboard-surface); 
        border: 1px solid var(--dashboard-border); 
        border-radius: 12px; 
      } 

      [data-testid="stDataFrame"] { border-radius: 8px; } 

      [data-testid="stImage"] { 
        background: var(--dashboard-surface); 
        border: 1px solid var(--dashboard-border); 
        border-radius: 12px; 
        padding: 10px; 
      } 

      .research-hero { 
        box-sizing: border-box; 
        border-radius: 18px; 
        border: 1px solid #244C5C;
        padding: 28px 32px; 
        margin-bottom: 24px; 
        color: #F6FAF8;
        background: linear-gradient(115deg, #183247, #205C60);
      } 

      .research-hero h1 { 
        color: #F6FAF8;
        font-family: Arial, sans-serif; 
        font-weight: 600; 
        font-size: clamp(1.4rem, 2vw, 1.95rem); 
        line-height: 1.4; 
        letter-spacing: -.02em; 
        margin: 12px 0 18px; 
        padding: 0; 
        max-width: 1240px; 
        overflow-wrap: break-word; 
      } 

      .eyebrow { 
        font-size: .7rem; 
        letter-spacing: .11em; 
        font-weight: 700; 
        color: var(--dashboard-accent); 
      } 

      .research-hero .eyebrow { color: #B6DDD3; }

      .hero-status { 
        display: inline-block; 
        font-size: .8rem; 
        line-height: 1.5; 
        color: #E1F0EA;
        border: 1px solid #FFFFFF33;
        border-radius: 30px; 
        background: #FFFFFF12;
        padding: 7px 12px; 
      } 

      .section-intro { margin: 4px 0 18px; } 

      .section-intro h2 { 
        font-size: 1.35rem; 
        margin: 0 0 6px; 
        padding: 0; 
      } 

      .section-intro p { 
        color: var(--dashboard-muted); 
        font-size: .92rem; 
        line-height: 1.55; 
        margin: 0; 
      } 

      .st-key-research-overview { padding: 8px 0 24px; } 
      .st-key-research-overview .section-intro { margin: 0 0 24px; } 
      .st-key-research-overview .section-intro h2 { margin-bottom: 10px; } 

      .research-question { padding-top: 24px; } 

      .research-question h3 { 
        font-size: 1rem; 
        margin: 0 0 10px; 
        padding: 0; 
      } 

      .research-question p { 
        font-size: .95rem; 
        line-height: 1.65; 
        margin: 0; 
        padding: 0; 
      } 

      .st-key-dataset-table { padding-top: 16px; } 
      .st-key-correlation-results { padding-top: 24px; } 
      .st-key-regression-chart, .st-key-regression-results { padding-top: 20px; } 

      .insight { 
        background: var(--dashboard-accent-soft); 
        border: 1px solid var(--dashboard-accent-border); 
        border-radius: 12px; 
        padding: 14px 18px; 
        color: var(--dashboard-accent-dark); 
        margin: 6px 0 16px; 
        line-height: 1.55; 
      } 

      .metric-grid, .study-grid { 
        display: grid; 
        grid-template-columns: repeat(var(--columns, 3), minmax(0, 1fr)); 
        gap: 16px; 
        align-items: stretch; 
        margin: 4px 0 18px; 
      } 

      .stat-card, .study-card { 
        box-sizing: border-box; 
        min-width: 0; 
        background: var(--dashboard-surface); 
        border: 1px solid var(--dashboard-border); 
        border-radius: 16px; 
        padding: 20px 22px; 
        box-shadow: 0 6px 20px var(--dashboard-shadow); 
      } 

      .stat-card { 
        display: flex; 
        flex-direction: column; 
        min-height: 0; 
        padding: 16px 20px; 
        background: linear-gradient(135deg, var(--dashboard-surface), var(--dashboard-surface-tint)); 
      } 

      .stat-label { 
        color: var(--dashboard-muted); 
        font-size: .82rem; 
        font-weight: 600; 
        line-height: 1.45; 
        display: flex; 
        align-items: center; 
        justify-content: space-between; 
        gap: 10px; 
      } 

      .stat-label::after { 
        content: ''; 
        flex: 0 0 7px; 
        height: 7px; 
        border-radius: 50%; 
        background: var(--dashboard-accent); 
      } 

      .stat-value { 
        color: var(--dashboard-accent); 
        font-size: clamp(1.5rem, 2.2vw, 1.95rem); 
        font-weight: 600; 
        line-height: 1.2; 
        font-variant-numeric: tabular-nums; 
        margin: 8px 0 0; 
        overflow-wrap: anywhere; 
      } 

      .stat-context { 
        color: var(--dashboard-muted); 
        font-size: .77rem; 
        line-height: 1.45; 
        margin-top: 4px; 
      } 

      .stat-context:empty { display: none; } 

      .metric-grid.compact { 
        grid-template-columns: repeat(var(--columns, 5), minmax(145px, 1fr)); 
        gap: 12px; 
        overflow-x: auto; 
      } 

      .metric-grid.compact .stat-card { 
        min-height: 0; 
        padding: 14px 16px; 
        border-radius: 12px; 
      } 

      .metric-grid.compact .stat-value { 
        font-size: .95rem; 
        margin: 8px 0 0; 
      } 

      .study-card { min-height: 152px; } 

      .study-card h3 { 
        font-size: 1rem; 
        margin: 13px 0 8px; 
        padding: 0; 
        line-height: 1.4; 
      } 

      .study-card p { 
        color: var(--dashboard-muted); 
        font-size: .9rem; 
        line-height: 1.6; 
        margin: 0; 
        padding: 0; 
      } 

      .study-card .eyebrow { 
        display: inline-block; 
        color: var(--dashboard-accent-dark); 
        background: var(--dashboard-accent-soft); 
        border-radius: 6px; 
        padding: 5px 8px; 
        font-size: .66rem; 
      } 

      .decision { 
        box-sizing: border-box; 
        background: var(--dashboard-surface); 
        border: 1px solid var(--dashboard-border); 
        border-radius: 16px; 
        padding: 24px; 
        margin: 10px 0 20px; 
        box-shadow: 0 6px 20px var(--dashboard-shadow); 
      } 

      .decision.supported { background: var(--dashboard-surface-tint); } 

      .decision .eyebrow { color: var(--dashboard-accent); } 

      .decision h2 { 
        font-family: Arial, sans-serif; 
        font-size: 1.5rem; 
        margin: 12px 0; 
        padding: 0; 
        line-height: 1.4; 
      } 

      .decision p { 
        color: var(--dashboard-muted); 
        margin: 0; 
        line-height: 1.6; 
      } 

      @media (max-width: 1000px) { 
        .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } 
      } 

      @media (max-width: 650px) { 
        .block-container { padding-top: 1rem; } 
        .research-hero { padding: 24px 22px 26px; } 
        .metric-grid, .study-grid { grid-template-columns: 1fr; gap: 12px; } 
        .stat-card, .study-card { padding: 18px 20px; } 
        [data-testid="stTabs"] [data-baseweb="tab"] { padding: 10px 12px; } 
      } 
    </style> 
    """, unsafe_allow_html=True)

def render_header(report=None, filename=None):
    status = "Awaiting Dataset"
    if report is not None:
        status = f"{len(report['raw']):,} repository rows | {filename or 'Dataset loaded'}"
    st.markdown(
        '<div class="research-hero"><span class="eyebrow">EMPIRICAL RESEARCH · GITHUB REPOSITORIES</span>'
        f'<h1>{escape(RESEARCH_TITLE)}</h1>'
        f'<span class="hero-status">{escape(status)}</span></div>', unsafe_allow_html=True,
    )


def section_intro(title, subtitle):
    st.markdown(
        f'<div class="section-intro"><h2>{escape(title)}</h2><p>{escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def insight(text):
    st.markdown(f'<div class="insight">{escape(text)}</div>', unsafe_allow_html=True)


def metric_cards(items, columns=3, compact=False):
    cards = []
    for item in items:
        context = escape(str(item[2])) if len(item) > 2 else ""
        cards.append(
            f'<div class="stat-card" role="group" aria-label="{escape(str(item[0]), quote=True)}">'
            f'<div class="stat-label">{escape(str(item[0]))}</div>'
            f'<div class="stat-value">{escape(str(item[1]))}</div>'
            f'<div class="stat-context">{context}</div></div>'
        )
    grid_class = "metric-grid compact" if compact else "metric-grid"
    st.markdown(f'<div class="{grid_class}" style="--columns:{columns}">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def study_card(label, title, description):
    return (
        f'<div class="study-card"><span class="eyebrow">{escape(label)}</span>'
        f'<h3>{escape(title)}</h3><p>{escape(description)}</p></div>'
    )


def research_statements():
    cards = study_card("H1", "Alternative Hypothesis", H1) + study_card("H0", "Null Hypothesis", H0)
    st.markdown('<div class="study-grid" style="--columns:2">' + cards + '</div>', unsafe_allow_html=True)


def render_overview():
    with st.container(key="research-overview", gap=None):
        section_intro("Research Overview", "The impact of Technical Documentation Quality on Software Maintenance Efficiency in Enterprise-Level Software Projects.")
        st.markdown(
            '<div class="research-question"><h3>Research Question</h3>'
            f'<p>{escape(RESEARCH_QUESTION)}</p></div>', unsafe_allow_html=True,
        )
    research_statements()
    cards = [study_card(label, title, description) for label, title, description in [
        (
            "MEASURE", "Technical Documentation Quality",
            "Assess technical documentation quality using 15 predefined metrics derived through keyword-based analysis."
        ),
        (
            "GATHER", "Software Maintenance Efficiency",
            "Collect repository-level maintenance efficiency metrics from GitHub issue and pull request activity."
        ),
        (
            "ANALYZE", "Spearman Correlation Analysis",
            "Analyze the individual and overall statistical relationships between documentation quality metrics and software maintenance efficiency."
        ),
        (
            "EVALUATE", "Simple Linear Regression",
            "Evaluate the overall statistical relationship between documentation quality and software maintenance efficiency at a 0.05 significance level."
        ),
    ]]
    st.markdown('<div class="study-grid" style="--columns:4">' + ''.join(cards) + '</div>', unsafe_allow_html=True)


def render_dataset(report):
    raw, prepared = report["raw"], report["prepared"]
    regression_count = int(prepared[[DOCUMENTATION_QUALITY_SCORE_COLUMN, MAINTENANCE_SCORE_COLUMN]].notna().all(axis=1).sum())
    section_intro("Dataset Overview", "Information and Overview of the Final Integrated Dataset.")
    metric_cards([
        ("Repository Rows", f"{len(raw):,}"),
        ("Valid Regression Pairs", regression_count),
        ("Documentation Metrics", len(DOCUMENTATION_METRIC_COLUMNS)),
        ("Maintenance Variables", sum(column in raw for column in MAINTENANCE_VARIABLE_COLUMNS)),
    ], columns=4)
    with st.container(key="dataset-table"):
        st.dataframe(
            raw, hide_index=True, height=380, width="stretch",
            column_config={column: readable_name(column) if column != "repo_id" else "Repository" for column in raw.columns},
        )
    if report["score_created"]:
        partial = int(prepared[DOCUMENTATION_METRIC_COLUMNS].notna().sum(axis=1).between(1, 14).sum())
        if partial:
            st.warning(f"The composite averages available documentation metrics; {partial} rows have fewer than 15 observed metrics.")


def valid_correlations(report):
    return report["correlations"].dropna(subset=["spearman_correlation", "p_value"])


def show_figure(factory, *arguments):
    figure = None
    try:
        figure = factory(*arguments)
        st.pyplot(figure, width="stretch")
    except Exception:
        st.warning("This chart could not be rendered. Review its valid observations and results.")
    finally:
        if figure is not None:
            plt.close(figure)


def render_visualizations(report):
    charts = {
        "Ranked Correlation Bar Chart": (create_ranked_correlation_bar_chart, report["correlations"]),
        "Focused Spearman Correlation Heatmap": (create_focused_correlation_heatmap, report["correlations"]),
        "Full Spearman Correlation Heatmap": (create_full_correlation_heatmap, report["prepared"]),
    }
    view = st.radio("Visualizations", list(charts), horizontal=True)
    factory, data = charts[view]
    show_figure(factory, data)


def render_overall_correlation(report):
    result = report["overall_correlation"]
    section_intro(
        "Correlation Analysis",
        "Spearman Correlation between Technical Documentation Quality "
        "and Software Maintenance Efficiency.",
    )
    metric_cards([
        ("Spearman Correlation (ρ)", format_number(result["spearman_correlation"])),
        ("P-value", format_number(result["p_value"], p_value=True), result["significance_result"]),
        ("Valid Pairs", int(result["sample_size"])),
        ("Relationship Strength", result["strength"], result["direction"]),
    ], columns=4)
    if result["status"] != "Calculated":
        st.warning(result["status"])


def render_correlation_strength_guide():
    section_intro("Correlation Strength Guide", "Strength is based on the absolute Spearman correlation, |ρ|.")
    metric_cards([
        ("Very Weak", "0.00 ≤ |ρ| < 0.20"),
        ("Weak", "0.20 ≤ |ρ| < 0.40"),
        ("Moderate", "0.40 ≤ |ρ| < 0.60"),
        ("Strong", "0.60 ≤ |ρ| < 0.80"),
        ("Very Strong", "0.80 ≤ |ρ| ≤ 1.00"),
    ], columns=5, compact=True)


def render_correlations(report):
    render_overall_correlation(report)
    st.divider()
    results = report["correlations"]
    valid = valid_correlations(report)
    section_intro("Individual Correlation Analysis", "The individual correlations between technical documentation quality metrics and software maintenance efficiency.")
    strongest = valid.loc[valid["spearman_correlation"].abs().idxmax()] if not valid.empty else None
    metric_cards([
        ("Evaluable Metrics", f"{len(valid)} / {len(results)}"),
        ("Significant Relationships", int((valid["p_value"] < ALPHA).sum()), "p < 0.05"),
        ("Strongest Association", format_number(strongest["spearman_correlation"]) if strongest is not None else "Unavailable",
         readable_name(strongest["metric_name"]) if strongest is not None else "Insufficient observations or variation"),
    ], columns=4)
    if len(valid) != len(results):
        st.warning(f"{len(results) - len(valid)} metrics are unavailable. Reasons are listed with the results below.")
    render_visualizations(report)
    with st.container(key="correlation-results"):
        with st.expander("All Metric Results"):
            display = results.copy()
            display["metric_name"] = display["metric_name"].map(readable_name)
            st.dataframe(display, hide_index=True, height=560, width="stretch", column_config={
                "metric_name": st.column_config.TextColumn("Documentation Metric", width="large"),
                "spearman_correlation": st.column_config.NumberColumn("Spearman Correlation", format="%.4f"),
                "p_value": st.column_config.NumberColumn("P-Value", format="%.4g"),
                "sample_size": st.column_config.NumberColumn("Valid Pairs", format="%d"),
                "direction": "Direction", "strength": "Strength",
                "significance_result": "Significance", "status": "Calculation Status",
            })
    st.divider()
    render_correlation_strength_guide()


def results_table(result):
    labels = {
        "sample_size": "Valid Observations", "intercept": "Intercept (β₀)",
        "regression_coefficient": "Documentation Quality Coefficient (β₁)",
        "coefficient_p_value": "Coefficient P-Value", "coefficient_standard_error": "Coefficient Standard Error",
        "coefficient_t_statistic": "Coefficient T-Statistic", "r_squared": "R-Squared",
        "adjusted_r_squared": "Adjusted R-Squared", "f_statistic": "F-Statistic", "f_p_value": "F-Test P-Value",
        "significance_result": "Statistical Significance", "hypothesis_decision": "Hypothesis Decision",
    }
    return pd.DataFrame([
        {"Statistic": labels.get(key, readable_name(key)),
         "Value": str(value) if isinstance(value, (str, int)) else format_number(value, p_value="p_value" in key)}
        for key, value in result.items()
    ])


def render_regression(report):
    section_intro("Simple Linear Regression", "Technical Documentation Quality in relation to Software Maintenance Efficiency.")
    if report["regression_error"]:
        st.warning(report["regression_error"])
        return
    rows, model, result = report["regression"]
    metric_cards([
        ("Regression Coefficient (β₁)", format_number(result["regression_coefficient"])),
        ("P-Value", format_number(result["coefficient_p_value"], p_value=True)),
        ("R-Squared Value", format_number(result["r_squared"]), f"{result['r_squared'] * 100:.2f}% of observed variation"),
        ("Valid Observations", result["sample_size"]),
    ], columns=4)
    direction = "Positive" if result["regression_coefficient"] > 0 else "Negative" if result["regression_coefficient"] < 0 else "Non-Directional"
    if np.isfinite(result["coefficient_p_value"]):
        insight(f"The Fitted Relationship is {direction} but {result['significance_result']} at alpha = {ALPHA:.2f}.")
    else:
        st.warning("The coefficient p-value is undefined; statistical significance cannot be evaluated.")
    with st.container(key="regression-chart"):
        show_figure(create_regression_scatter_plot, rows, model)
    st.caption(f"{len(rows)} complete observations · {len(report['raw']) - len(rows)} incomplete pairs excluded · Shading: 95% confidence interval for the mean response.")
    with st.container(key="regression-results"):
        with st.expander("Full Regression Results"):
            st.dataframe(results_table(result), hide_index=True, width="stretch")
            st.caption("Software Maintenance efficiency = β₀ + β₁ × Technical Documentation Quality")


def render_final(report):
    section_intro(
        "Hypothesis Evaluation",
        "The hypothesis decision uses the regression coefficient p-value, "
        "with overall Spearman correlation as supporting evidence.",
    )
    result = report["regression"][2] if report["regression"] else None
    if result is None or not np.isfinite(result["coefficient_p_value"]):
        st.warning("Hypothesis cannot be evaluated. " + (report["regression_error"] or "The coefficient p-value is unavailable."))
    else:
        supported = result["coefficient_p_value"] < ALPHA
        decision = "Alternative Hypothesis Supported" if supported else "Alternative Hypothesis Not Supported"
        explanation = (
            "The dataset provides evidence of a statistically significant relationship."
            if supported else "The selected dataset provides insufficient evidence of a statistically significant relationship."
        )
        st.markdown(
            f'<div class="decision {"supported" if supported else ""}"><span class="eyebrow">RESEARCH FINDING</span>'
            f'<h2>{escape(decision)}</h2><p>{escape(explanation)}</p></div>', unsafe_allow_html=True,
        )
    regression = result or {}
    variation = regression.get("r_squared")
    metric_cards([
        ("Regression P-Value", format_number(regression.get("coefficient_p_value"), p_value=True),
         regression.get("significance_result", "Not Available")),
        ("Regression Coefficient", format_number(regression.get("regression_coefficient"))),
        ("Variation Explained", f"{variation * 100:.2f}%" if variation is not None and np.isfinite(variation) else "Not Available"),
        ("Valid Regression Observations", regression.get("sample_size", "Not Available")),
    ], columns=4)
    correlation = report["overall_correlation"]
    if correlation["status"] != "Calculated":
        st.warning("Overall Spearman correlation is unavailable. " + correlation["status"])
    metric_cards([
        ("Spearman Correlation (ρ)", format_number(correlation["spearman_correlation"])),
        ("Spearman P-Value", format_number(correlation["p_value"], p_value=True), correlation["significance_result"]),
        ("Correlation Strength", correlation["strength"], correlation["direction"]),
        ("Valid Correlation Pairs", correlation["sample_size"]),
    ], columns=4)


def render_dashboard(report=None, dataset_error=None):
    tabs = st.tabs(TAB_NAMES)
    with tabs[0]:
        render_overview()
    for index, (tab, renderer) in enumerate(zip(tabs[1:], [render_dataset, render_correlations, render_regression, render_final]), start=1):
        with tab:
            if dataset_error:
                st.error(dataset_error)
            elif report is None:
                section_intro(TAB_NAMES[index], "Your results will appear here after you upload a dataset.")
                st.info("Upload the final integrated dataset in the sidebar.")
            else:
                renderer(report)
