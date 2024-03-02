from pathlib import Path

import numpy as np
import pandas as pd
from keras.models import Model
from keras.callbacks import History, EarlyStopping

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams
from bi_nitrogen_fertilization_ml_pipeline.model_training.evaluation.random_guess_predictions import \
    calculate_evaluation_metric_for_random_guess_predictions
from bi_nitrogen_fertilization_ml_pipeline.model_training.training.train_process_graphs import \
    plot_evaluation_value_per_training_epoch_graph
from bi_nitrogen_fertilization_ml_pipeline.model_training.utils.train_params_to_keras_api_conversions import \
    eval_func_to_keras_loss, eval_func_to_keras_metric
from bi_nitrogen_fertilization_ml_pipeline.model_training.utils.keras_utils import is_model_compiled


def fit_keras_model(
    model: Model,
    X: pd.DataFrame,
    y: pd.Series,
    train_params: TrainParams,
    output_figures_folder_path: Path,
) -> None:
    _compile_model(model, train_params)
    if train_params.early_stopping is not None:
        train_history, applied_validation_set_split = _fit_model_with_early_stopping(
            model, X, y, train_params)
    else:
        train_history, applied_validation_set_split = _fit_model(
            model, X, y, train_params)
    _plot_train_graphs(
        train_history,
        train_params,
        applied_validation_set_split,
        output_figures_folder_path,
        y,
    )


def _fit_model(
    model: Model,
    X: pd.DataFrame,
    y: pd.Series,
    train_params: TrainParams,
) -> tuple[History, bool]:
    applied_validation_set_split = False
    train_history = model.fit(
        X, y, epochs=train_params.epochs_count,
    )
    return train_history, applied_validation_set_split


def _fit_model_with_early_stopping(
    model: Model,
    X: pd.DataFrame,
    y: pd.Series,
    train_params: TrainParams,
) -> tuple[History, bool]:
    applied_validation_set_split = True
    train_history = model.fit(
        X.to_numpy(np.float32),
        y.to_numpy(np.float32),
        epochs=train_params.epochs_count,
        validation_split=train_params.early_stopping.validation_set_fraction_size,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                patience=train_params.early_stopping.tolerance_epochs_count,
            ),
        ])
    return train_history, applied_validation_set_split


def _plot_train_graphs(
    train_history: History,
    train_params: TrainParams,
    applied_validation_set_split: bool,
    output_figures_folder_path: Path,
    y: pd.Series,
) -> None:
    train_result_values_per_epoch = train_history.history
    actual_train_epochs_count = len(train_result_values_per_epoch['loss'])

    loss_function = train_params.loss_function
    loss_func_random_guess_value = calculate_evaluation_metric_for_random_guess_predictions(y, loss_function)
    group_name_to_ordered_epoch_loss_values = {
        f'train {loss_function.value}': train_result_values_per_epoch['loss'],
        **(
            {f'validation {loss_function.value}': train_result_values_per_epoch['val_loss']}
            if applied_validation_set_split else {}
        ),
        f'random guess {loss_function.value}': [loss_func_random_guess_value] * actual_train_epochs_count,
    }
    plot_evaluation_value_per_training_epoch_graph(
        group_name_to_ordered_epoch_loss_values,
        eval_value_display_name=f'loss ({loss_function.value})',
        output_graph_jpeg_file_path=output_figures_folder_path / 'loss_per_epoch.jpeg',
        set_y_axis_min_limit_to_0=True,
    )

    eval_metric = train_params.evaluation_metric
    keras_name_of_eval_metric = eval_func_to_keras_metric(train_params.evaluation_metric).name
    eval_metric_random_guess_value = \
        calculate_evaluation_metric_for_random_guess_predictions(y, eval_metric)
    group_name_to_ordered_epoch_eval_metric_values = {
        f'train {eval_metric.value}': train_result_values_per_epoch[keras_name_of_eval_metric],
        **(
            {f'validation {eval_metric.value}': train_result_values_per_epoch[f'val_{keras_name_of_eval_metric}']}
            if applied_validation_set_split else {}
        ),
        f'random guess {eval_metric.value}': [eval_metric_random_guess_value] * actual_train_epochs_count,
    }
    plot_evaluation_value_per_training_epoch_graph(
        group_name_to_ordered_epoch_eval_metric_values,
        eval_value_display_name=eval_metric.value,
        output_graph_jpeg_file_path=output_figures_folder_path / f'{eval_metric.value}_per_epoch.jpeg',
        set_y_axis_min_limit_to_0=True,
    )


def _compile_model(model: Model, train_params: TrainParams) -> None:
    assert not is_model_compiled(model), 'the input model must not be compiled already'

    loss = eval_func_to_keras_loss(train_params.loss_function)
    evaluation_metric = eval_func_to_keras_metric(train_params.evaluation_metric)
    model.compile(
        optimizer=train_params.optimizer,
        loss=loss,
        metrics=[evaluation_metric],
    )
