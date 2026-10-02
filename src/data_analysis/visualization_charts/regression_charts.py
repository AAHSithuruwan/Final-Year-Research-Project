import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
import statsmodels.api as sm
from analysis_functions.constants import ALPHA, DOCUMENTATION_QUALITY_SCORE_COLUMN, MAINTENANCE_SCORE_COLUMN


# Create the regression scatter plot
def create_regression_scatter_plot(regression_df, model):
    x = regression_df[DOCUMENTATION_QUALITY_SCORE_COLUMN]
    y = regression_df[MAINTENANCE_SCORE_COLUMN]
    grid = pd.DataFrame({DOCUMENTATION_QUALITY_SCORE_COLUMN: np.linspace(x.min(), x.max(), 150)})
    prediction = model.get_prediction(sm.add_constant(grid, has_constant="add")).summary_frame(alpha=ALPHA)
    figure, axis = plt.subplots(figsize=(10.2, 7.4), dpi=180, layout="constrained")
    figure.set_facecolor("#ffffff")
    axis.set_facecolor("#ffffff")
    axis.scatter(
        x, y, color="#0F766E", alpha=0.76, s=52, edgecolors="#ffffff", linewidths=0.45,
        label="Repository Observations", zorder=3,
    )
    axis.plot(
        grid[DOCUMENTATION_QUALITY_SCORE_COLUMN], prediction["mean"], color="#B85C38",
        linewidth=3.1, label="Regression Line", zorder=4,
    )
    if np.isfinite(prediction[["mean_ci_lower", "mean_ci_upper"]].to_numpy()).all():
        axis.fill_between(
            grid[DOCUMENTATION_QUALITY_SCORE_COLUMN], prediction["mean_ci_lower"], prediction["mean_ci_upper"],
            color="#B85C38", alpha=0.14, linewidth=0,
            label=f"{100 * (1 - ALPHA):g}% Confidence Interval (Mean)", zorder=2,
        )
    axis.set_title(
        "Technical Documentation Quality vs\nSoftware Maintenance Efficiency", loc="left", pad=24,
        fontsize=15, fontweight="bold", color="#183247",
    )
    axis.set_xlabel("Technical Documentation Quality Score", fontsize=12, labelpad=12, color="#526775")
    axis.set_ylabel("Software Maintenance Efficiency", fontsize=12, labelpad=12, color="#526775")
    axis.tick_params(axis="both", colors="#526775", labelsize=11, length=0, pad=8)
    axis.legend(
        fontsize=10.5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
        frameon=False, ncol=2, labelcolor="#183247", columnspacing=1.4, handlelength=2.2,
    )
    axis.margins(x=0.035, y=0.08)
    axis.grid(color="#E1E8E7", linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for text in figure.findobj(match=Text):
        text.set_fontfamily("Arial")
    return figure
