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
    samples_count = folds_eval_set_y_true_and_pred_df.shape[0]
    y_true_array = folds_eval_set_y_true_and_pred_df['y_true'].to_numpy()
    fold_model_y_pred_array = folds_eval_set_y_true_and_pred_df['fold_model_y_pred'].to_numpy()
    folds_eval_set_prediction_deviations = fold_model_y_pred_array - y_true_array

    try_ratio = (y_true_array.std() * 5) / samples_count

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, figsize=SQUARE_FIGURES_SIZE, constrained_layout=True,
    )
    try:
        fig.set_dpi(SAVE_FIGURES_DPI)

        zero_deviations_line_x = np.array(range(len(folds_eval_set_prediction_deviations)))
        zero_deviations_line_y = zero_deviations_line_x * try_ratio

        actual_deviations_scatter_x = zero_deviations_line_x
        actual_deviations_scatter_y = zero_deviations_line_y + folds_eval_set_prediction_deviations

        ax1.plot(zero_deviations_line_x, zero_deviations_line_y, color='black', linewidth=1, label='Zero deviation')
        ax1.scatter(actual_deviations_scatter_x, actual_deviations_scatter_y, color='blue', label='Deviations')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Prediction Deviation')
        ax1.set_title('Option 1')
        ax1.legend()

        ax2.axhline(0, color='black', linewidth=1, label='Zero deviation')
        ax2.scatter(range(len(folds_eval_set_prediction_deviations)), folds_eval_set_prediction_deviations, color='blue', label='Deviations')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Prediction Deviation')
        ax2.set_title('Option 2')
        ax2.legend()
        # ax2.grid(True)

        fig.savefig(str(folds_eval_sets_prediction_deviations_graph_path))
    finally:
        plt.close(fig)

    return folds_eval_sets_prediction_deviations_graph_path
