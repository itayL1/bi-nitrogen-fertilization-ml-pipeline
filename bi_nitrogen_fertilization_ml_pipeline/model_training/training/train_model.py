from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from keras.callbacks import History, EarlyStopping
from keras.models import Model
from sklearn.model_selection import train_test_split

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams
from bi_nitrogen_fertilization_ml_pipeline.model_training.evaluation.random_guess_predictions import \
    calculate_evaluation_metric_for_random_guess_predictions
from bi_nitrogen_fertilization_ml_pipeline.model_training.training.train_process_graphs import \
    plot_evaluation_value_per_training_epoch_graph
from bi_nitrogen_fertilization_ml_pipeline.model_training.utils.train_params_to_keras_api_conversions import \
    eval_func_to_keras_metric


def train_model(
    model: Model,
    X: pd.DataFrame,
    y: pd.Series,
    train_params: TrainParams,
    output_figures_folder_path: Path,
) -> None:
    if train_params.early_stopping is not None:
        applied_validation_set_split = True
        train_history, y_train, y_validation = _fit_model_with_early_stopping(
            model, X, y, train_params)
    else:
        y_train = y
        y_validation = None
        applied_validation_set_split = False
        train_history = _fit_model(model, X, y, train_params)

    _plot_train_graphs(
        train_history,
        train_params,
        output_figures_folder_path,
        y_train,
        applied_validation_set_split,
        y_validation,
    )


def _fit_model(
    model: Model,
    X: pd.DataFrame,
    y: pd.Series,
    train_params: TrainParams,
) -> History:
    train_history = model.fit(
        X,
        y,
        shuffle=True,
        epochs=train_params.epochs_count,
        verbose=0 if train_params.silent_models_fitting else 'auto',
    )
    return train_history


def _fit_model_with_early_stopping(
    model: Model,
    X: pd.DataFrame,
    y: pd.Series,
    train_params: TrainParams,
) -> tuple[History, pd.Series, pd.Series]:
    validation_set_fraction_size = train_params.early_stopping.validation_set_fraction_size
    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, test_size=validation_set_fraction_size, shuffle=True)

    train_history = model.fit(
        X_train.to_numpy(np.float32),
        y_train.to_numpy(np.float32),
        shuffle=True,
        epochs=train_params.epochs_count,
        validation_data=(
            X_validation.to_numpy(np.float32),
            y_validation.to_numpy(np.float32),
        ),
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                patience=train_params.early_stopping.tolerance_epochs_count,
            ),
        ],
        verbose=0 if train_params.silent_models_fitting else 'auto',
    )
    return train_history, y_train, y_validation


def _plot_train_graphs(
    train_history: History,
    train_params: TrainParams,
    output_figures_folder_path: Path,
    y_train: pd.Series,
    applied_validation_set_split: bool,
    y_validation: Optional[pd.Series] = None
) -> None:
    if applied_validation_set_split:
        assert y_validation is not None, 'when applied_validation_set_split is enabled, '\
                                         'the y_validation argument must be provided'

    train_result_values_per_epoch = train_history.history
    actual_train_epochs_count = len(train_result_values_per_epoch['loss'])

    loss_function = train_params.loss_function
    loss_func_random_guess_value = calculate_evaluation_metric_for_random_guess_predictions(
        y_train,
        y_validation if applied_validation_set_split else y_train,
        loss_function,
    )
    group_name_to_ordered_epoch_loss_values = {
        'train': train_result_values_per_epoch['loss'],
        **(
            {'validation': train_result_values_per_epoch['val_loss']}
            if applied_validation_set_split else {}
        ),
        'random guess': [loss_func_random_guess_value] * actual_train_epochs_count,
    }
    plot_evaluation_value_per_training_epoch_graph(
        group_name_to_ordered_epoch_loss_values,
        eval_value_func=loss_function,
        is_loss_function=True,
        output_graph_jpeg_file_path=output_figures_folder_path / 'loss_per_epoch.jpeg',
        y_axis_min_limit=0.0,
        y_axis_max_limit=loss_func_random_guess_value * 1.5,
    )

    eval_metric = train_params.evaluation_metric
    keras_name_of_eval_metric = eval_func_to_keras_metric(train_params.evaluation_metric).name
    eval_metric_random_guess_value = calculate_evaluation_metric_for_random_guess_predictions(
        y_train,
        y_validation if applied_validation_set_split else y_train,
        loss_function,
    )
    group_name_to_ordered_epoch_eval_metric_values = {
        'train': train_result_values_per_epoch[keras_name_of_eval_metric],
        **(
            {'validation': train_result_values_per_epoch[f'val_{keras_name_of_eval_metric}']}
            if applied_validation_set_split else {}
        ),
        'random guess': [eval_metric_random_guess_value] * actual_train_epochs_count,
    }
    plot_evaluation_value_per_training_epoch_graph(
        group_name_to_ordered_epoch_eval_metric_values,
        eval_value_func=eval_metric,
        is_loss_function=False,
        output_graph_jpeg_file_path=output_figures_folder_path / f'{eval_metric.value}_per_epoch.jpeg',
        y_axis_min_limit=0.0,
        y_axis_max_limit=eval_metric_random_guess_value * 1.5,
    )
