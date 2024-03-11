from pathlib import Path

import keras
import pandas as pd
import shap
import shap.maskers
from matplotlib import pyplot as plt

FIGURES_DPI = 150
IMPORTANCE_EXTRACTION_SAMPLE_SIZE = 100


def extract_feature_importance_using_shap(
    model: keras.Model,
    X: pd.DataFrame,
    output_summary_figure_path: Path,
    show_progress_bar: bool = False
):
    output_summary_figure_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_figure_path.unlink(missing_ok=True)

    shap_masker = shap.maskers.Independent(data=X)
    shap_explainer = shap.Explainer(model, masker=shap_masker, feature_names=X.columns.tolist())

    X_sample = X.sample(n=IMPORTANCE_EXTRACTION_SAMPLE_SIZE)
    model_shap_values = shap_explainer(X_sample, silent=not show_progress_bar)
    try:
        shap.summary_plot(model_shap_values, feature_names=X.columns.tolist(), show=False)
        plt.savefig(str(output_summary_figure_path), dpi=FIGURES_DPI, bbox_inches='tight')
    finally:
        plt.close()
