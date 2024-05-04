from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.core.plot_settings import SQUARE_FIGURES_SIZE, SAVE_FIGURES_DPI


def plot_folds_eval_sets_prediction_deviations_graph(
    folds_eval_set_y_true_and_pred_df: pd.DataFrame,
    session_context: TrainSessionContext,
) -> Path:
    folds_eval_set_y_true_and_pred_df = folds_eval_set_y_true_and_pred_df.copy()

    folds_eval_sets_prediction_deviations_graph_path = \
        session_context.wip_outputs_folder_path / 'folds_eval_sets_prediction_deviations.jpeg'

    folds_eval_set_y_true_and_pred_df.sort_values(by='fold_key', inplace=True)
    y_true_array = folds_eval_set_y_true_and_pred_df['y_true'].to_numpy()
    fold_model_y_pred_array = folds_eval_set_y_true_and_pred_df['fold_model_y_pred'].to_numpy()
    y_folds_eval_set_prediction_deviations = fold_model_y_pred_array - y_true_array
    x_sample_indices = range(len(y_folds_eval_set_prediction_deviations))

    fig, ax = plt.subplots(figsize=SQUARE_FIGURES_SIZE, constrained_layout=True)
    try:
        fig.set_dpi(SAVE_FIGURES_DPI)

        ax.axhline(0, color='black', linewidth=1, label='Zero deviation')
        ax.scatter(x_sample_indices, y_folds_eval_set_prediction_deviations, color='blue', label='Deviations')
        ax.set_xlabel('Sample Index (sorted by evaluation folds)')
        ax.set_ylabel('Prediction Deviation')
        ax.legend()

        fig.savefig(str(folds_eval_sets_prediction_deviations_graph_path))
    finally:
        plt.close(fig)

    return folds_eval_sets_prediction_deviations_graph_path
