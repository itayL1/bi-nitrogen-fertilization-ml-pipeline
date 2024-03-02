from pathlib import Path

import pandas as pd
from keras.models import Model
from keras.callbacks import EarlyStopping

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams
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
        train_history = model.fit(
            X,
            y,
            validation_split=train_params.early_stopping.validation_set_fraction_size,
            callbacks=[
                EarlyStopping(
                    monitor='val_loss',
                    patience=train_params.early_stopping.tolerance_epochs_count,
                ),
            ])
        train_history.history


def _compile_model(model: Model, train_params: TrainParams) -> None:
    assert not is_model_compiled(model), 'the input model must not be compiled already'

    loss = eval_func_to_keras_loss(train_params.loss_function)
    evaluation_metric = eval_func_to_keras_metric(train_params.evaluation_metric)
    model.compile(
        optimizer=train_params.optimizer,
        loss=loss,
        metrics=[evaluation_metric],
    )
