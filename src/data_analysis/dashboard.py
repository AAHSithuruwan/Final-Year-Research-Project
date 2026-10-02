from pathlib import Path
import sys
dashboard_directory = str(Path(__file__).resolve().parent)
if dashboard_directory not in sys.path:
    sys.path.insert(0, dashboard_directory)
import streamlit as st
from analysis_functions.constants import ALPHA, DOCUMENTATION_QUALITY_SCORE_COLUMN
from analysis_functions.correlation_analysis import (
    calculate_correlation_for_metric, calculate_correlations_for_all_metrics, full_correlation_matrix,
)
from analysis_functions.regression_analysis import analyze_regression
from dashboard_ui import apply_dashboard_style, render_dashboard, render_header
from utils.dataset_loader import load_dataset
from utils.dataset_validation import prepare_dataset


@st.cache_data(show_spinner=False, max_entries=3)
def analyze_uploaded_dataset(file_bytes, filename):
    raw = load_dataset(file_bytes, filename)
    prepared, conversions, score_created = prepare_dataset(raw)
    correlations = calculate_correlations_for_all_metrics(prepared)
    overall_correlation = calculate_correlation_for_metric(prepared, DOCUMENTATION_QUALITY_SCORE_COLUMN)
    matrix, matrix_samples = full_correlation_matrix(prepared)
    regression, regression_error = None, None
    try:
        regression = analyze_regression(prepared)
    except ValueError as error:
        regression_error = str(error)
    except Exception:
        regression_error = "Regression could not be completed. Review the numeric values and valid X/Y observations."
    return {
        "raw": raw, "prepared": prepared, "conversions": conversions,
        "score_created": score_created, "correlations": correlations,
        "overall_correlation": overall_correlation,
        "matrix": matrix, "matrix_samples": matrix_samples,
        "regression": regression, "regression_error": regression_error,
    }


def main():
    st.set_page_config(page_title="Documentation Quality | Research Dashboard", page_icon=":material/analytics:", layout="wide", initial_sidebar_state="expanded")
    apply_dashboard_style()
    with st.sidebar:
        st.header("Research Workspace")
        uploaded = st.file_uploader("Upload Final Integrated Dataset", type=["parquet", "csv"])
        st.divider()
        st.markdown("**Study Overview**")
        st.caption(f"Spearman Correlation\n\nSimple Linear Regression\n\nHypothesis Evaluation\n\nSignificance Level: {ALPHA:.2f}")
    report, dataset_error = None, None
    if uploaded is not None:
        try:
            with st.spinner("Preparing your research results..."):
                report = analyze_uploaded_dataset(uploaded.getvalue(), uploaded.name)
        except ValueError as error:
            dataset_error = str(error)
        except Exception:
            dataset_error = "The dataset could not be analyzed. Check its column names and numeric analysis values."
        if dataset_error:
            st.error(dataset_error)
        else:
            st.sidebar.success(f"{len(report['raw']):,} repository rows loaded")
            invalid = report["conversions"]["Invalid Values Converted to Missing"].sum()
            if invalid:
                st.warning(
                    f"{invalid} non-numeric or infinite analysis values were treated as missing."
                )
    render_header(report, uploaded.name if uploaded is not None else None)
    render_dashboard(report, dataset_error)


if __name__ == "__main__":
    main()
