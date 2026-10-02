from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import seaborn as sns
from analysis_functions.constants import ALPHA, readable_name
from analysis_functions.correlation_analysis import full_correlation_matrix


_INK = "#183247"
_MUTED = "#526775"
_POSITIVE = "#0F766E"
_NEGATIVE = "#B85C38"
_CORRELATION_COLORS = LinearSegmentedColormap.from_list(
    "documentation_correlation", [_NEGATIVE, "#F7F8F6", _POSITIVE]
)


def _label(column, width=24):
    return fill(readable_name(column), width=width, break_long_words=False)


def _set_chart_font(figure):
    for text in figure.findobj(match=Text):
        text.set_fontfamily("Arial")


def _style_axis(axis):
    axis.set_facecolor("#ffffff")
    axis.tick_params(axis="both", colors=_MUTED, labelsize=11, length=0, pad=7)
    axis.xaxis.label.set_color(_MUTED)
    axis.yaxis.label.set_color(_MUTED)
    for spine in axis.spines.values():
        spine.set_visible(False)


def _style_colorbar(axis):
    colorbar = axis.collections[0].colorbar
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    colorbar.ax.tick_params(labelsize=10.5, colors=_MUTED, length=0, pad=7)
    colorbar.ax.yaxis.label.set_color(_MUTED)
    colorbar.ax.yaxis.label.set_size(12)
    colorbar.outline.set_visible(False)


# Create a full Spearman correlation heatmap
def create_full_correlation_heatmap(dataframe, metric_columns=None):
    matrix, _ = full_correlation_matrix(dataframe, metric_columns)
    labels = matrix.rename(index=lambda column: _label(column, 20), columns=lambda column: _label(column, 20))
    # Each unique pair appears once; the diagonal remains a useful reference.
    hidden = np.triu(np.ones(matrix.shape, dtype=bool), k=1)
    figure, axis = plt.subplots(figsize=(14, 12.5), dpi=180, layout="constrained")
    figure.set_facecolor("#ffffff")
    sns.heatmap(
        labels, ax=axis, mask=hidden, annot=True, fmt=".2f",
        annot_kws={"fontsize": 9}, cmap=_CORRELATION_COLORS, vmin=-1, vmax=1,
        center=0, linewidths=0.8, linecolor="#ffffff", square=True,
        cbar_kws={"pad": 0.025, "shrink": 0.65, "label": "Spearman correlation (ρ)"},
    )
    # Seaborn masks missing entries. A dash distinguishes these from the hidden triangle.
    for row, column in np.argwhere(~np.isfinite(matrix.to_numpy()) & ~hidden):
        axis.text(column + 0.5, row + 0.5, "—", ha="center", va="center", color=_MUTED, fontsize=9)
    axis.set_title("Full Spearman Correlation Heatmap", loc="left", pad=24, fontsize=15, fontweight="bold", color=_INK)
    axis.set_xticklabels(axis.get_xticklabels(), rotation=55, ha="right", rotation_mode="anchor", fontsize=11)
    axis.set_yticklabels(axis.get_yticklabels(), rotation=0, fontsize=11)
    axis.set_xlabel("")
    axis.set_ylabel("")
    _style_axis(axis)
    _style_colorbar(axis)
    _set_chart_font(figure)
    return figure


# Create a focused Spearman correlation heatmap for documentation metrics vs maintenance score
def create_focused_correlation_heatmap(results):
    ranked = results.sort_values("spearman_correlation", ascending=False, na_position="last", kind="stable")
    values = ranked.set_index("metric_name")[["spearman_correlation"]]
    values.index = values.index.map(lambda column: _label(column, 32))
    values.columns = ["Software Maintenance\nEfficiency"]
    figure, axis = plt.subplots(figsize=(8.8, max(7.5, len(values) * 0.53)), dpi=180, layout="constrained")
    figure.set_facecolor("#ffffff")
    sns.heatmap(
        values, ax=axis, annot=True, fmt=".3f", annot_kws={"fontsize": 11.5},
        cmap=_CORRELATION_COLORS, vmin=-1, vmax=1, center=0, linewidths=1,
        linecolor="#ffffff", cbar_kws={"pad": 0.045, "shrink": 0.8, "label": "Spearman correlation (ρ)"},
    )
    for row, value in enumerate(values.iloc[:, 0]):
        if not np.isfinite(value):
            axis.text(0.5, row + 0.5, "Unavailable", ha="center", va="center", color=_MUTED, fontsize=11.5)
    axis.set_title("Documentation Quality Metrics vs\nSoftware Maintenance Efficiency", loc="left", pad=24, fontsize=14, fontweight="bold", color=_INK)
    axis.set_ylabel("")
    axis.set_xlabel("")
    axis.set_xticklabels(axis.get_xticklabels(), rotation=0)
    axis.set_yticklabels(axis.get_yticklabels(), rotation=0)
    _style_axis(axis)
    _style_colorbar(axis)
    _set_chart_font(figure)
    return figure


# Create a ranked bar chart of Spearman correlations for documentation metrics vs maintenance score
def create_ranked_correlation_bar_chart(results):
    ranked = results.sort_values("spearman_correlation", ascending=False, na_position="last", kind="stable")
    values = ranked["spearman_correlation"].to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    figure, axis = plt.subplots(figsize=(11, max(7.6, len(ranked) * 0.51)), dpi=180, layout="constrained")
    figure.set_facecolor("#ffffff")
    positions = np.arange(len(ranked))
    colors = [_POSITIVE if value >= 0 else _NEGATIVE for value in values]
    axis.barh(positions, np.where(np.isfinite(values), values, 0), color=colors, height=0.58, zorder=3)
    axis.set_yticks(positions, ranked["metric_name"].map(lambda column: _label(column, 34)))
    axis.invert_yaxis()
    if len(finite_values) and np.any(finite_values != 0):
        lower = min(0.0, float(finite_values.min()))
        upper = max(0.0, float(finite_values.max()))
        span = max(upper - lower, 0.02)
        axis.set_xlim(lower - span * 0.2, upper + span * 0.23)
    else:
        span = 0.2
        axis.set_xlim(-0.05, 0.15)
    padding = span * 0.025
    significance = (
        ranked["p_value"].to_numpy(dtype=float) < ALPHA
        if "p_value" in ranked else np.zeros(len(ranked), dtype=bool)
    )
    for position, value, significant in zip(positions, values, significance):
        if not np.isfinite(value):
            axis.text(padding, position, "Unavailable", va="center", color=_MUTED, fontsize=10.5)
        else:
            axis.text(
                value + (padding if value >= 0 else -padding), position,
                f"{value:.3f}{' *' if significant else ''}", va="center",
                ha="left" if value >= 0 else "right", color=_INK, fontsize=10.5,
            )
    axis.axvline(0, color=_MUTED, linewidth=1.25, zorder=2)
    axis.set_title("Documentation Quality Metrics\nRanked by Correlation", loc="left", pad=24, fontsize=15, fontweight="bold", color=_INK)
    footer = "Spearman correlation (ρ)"
    if significance.any():
        footer += f"\n* p < {ALPHA:g}"
    axis.set_xlabel(footer, labelpad=14, fontsize=12)
    axis.set_ylabel("")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    axis.xaxis.set_major_formatter(formatter)
    axis.grid(axis="x", color="#E1E8E7", linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    _style_axis(axis)
    _set_chart_font(figure)
    return figure
