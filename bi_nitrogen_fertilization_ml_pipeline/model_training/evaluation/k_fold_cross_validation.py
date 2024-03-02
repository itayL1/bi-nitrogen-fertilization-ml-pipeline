from pathlib import Path
from typing import Generator

import pandas as pd
from keras import Model
from tqdm import tqdm

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.k_fold_cross_validation import DatasetFoldSplit, \
    SingleFoldResults, KFoldCrossValidationResults
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.preprocessed_datasets import PreprocessedTrainDataset
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import ModelTraining
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.model_training.training.model_setup import prepare_new_model_for_training
from bi_nitrogen_fertilization_ml_pipeline.model_training.training.train_model import train_model


def key_based_k_fold_cross_validation(
    preprocessed_train_dataset: PreprocessedTrainDataset,
    session_context: TrainSessionContext,
) -> None:
    folds_train_figures_folder = session_context.temp_wip_outputs_folder_path / 'folds_train_figures'
    model_input_features_count = preprocessed_train_dataset.get_train_features_count()

    folds_count = _calc_folds_count(preprocessed_train_dataset)
    folds_results = []
    with tqdm('Evaluation cross validation folds loop', total=folds_count) as pbar:
        for fold_split in _split_dataset_to_folds_based_on_key_col(preprocessed_train_dataset):
            fold_key = fold_split.fold_key
            fold_train_output_figures_folder = _setup_fold_train_output_figures_folder(
                fold_key, folds_train_figures_folder, session_context)
            fold_results = _run_fold_iteration(
                fold_split, model_input_features_count, fold_train_output_figures_folder, session_context,
            )
            folds_results.append(fold_results)
            pbar.update()

    _populate_report_with_k_fold_evaluation_results(
        folds_results, folds_train_figures_folder, session_context)


def _calc_folds_count(preprocessed_train_dataset: PreprocessedTrainDataset) -> int:
    return len(preprocessed_train_dataset.evaluation_folds_key_col.unique())


def _split_dataset_to_folds_based_on_key_col(
    preprocessed_train_dataset: PreprocessedTrainDataset,
) -> Generator[DatasetFoldSplit, None, None]:
    X = preprocessed_train_dataset.X
    y = preprocessed_train_dataset.y
    evaluation_folds_key_col = preprocessed_train_dataset.evaluation_folds_key_col

    for fold_key, group_segment in _group_series_by_values(evaluation_folds_key_col):
        yield DatasetFoldSplit(
            fold_key=str(fold_key),
            X_train=X.iloc[~group_segment.index],
            y_train=y.iloc[~group_segment.index],
            X_evaluation=X.iloc[group_segment.index],
            y_evaluation=y.iloc[group_segment.index],
        )


def _group_series_by_values(series: pd.Series):
    return series.groupby(series)


def _setup_fold_train_output_figures_folder(
    fold_key: str,
    folds_train_figures_folder: Path,
    session_context: TrainSessionContext,
) -> Path:
    folds_key_field_name = session_context.params.evaluation_folds_key.column
    fold_train_output_figures_folder =\
        folds_train_figures_folder / f'fold__{folds_key_field_name}_{fold_key}'

    fold_train_output_figures_folder.mkdir(parents=True, exist_ok=True)
    return fold_train_output_figures_folder


def _run_fold_iteration(
    fold_split: DatasetFoldSplit,
    model_input_features_count: int,
    train_output_figures_folder: Path,
    session_context: TrainSessionContext,
) -> SingleFoldResults:
    fold_model = prepare_new_model_for_training(
        session_context.params, model_input_features_count)
    train_model(
        fold_model,
        X=fold_split.X_train,
        y=fold_split.y_train,
        train_params=session_context.params,
        output_figures_folder_path=train_output_figures_folder,
    )
    fold_results = _evaluate_fold_model(fold_model, fold_split)
    return fold_results


def _evaluate_fold_model(
    fold_model: Model,
    fold_split: DatasetFoldSplit,
) -> SingleFoldResults:
    loss_metric_name = 'loss'
    model_metric_names = set(fold_model.metrics_names)
    assert loss_metric_name in model_metric_names, 'unexpected state'
    assert len(model_metric_names) == 2, 'unexpected state'
    main_metric_name, = list(model_metric_names - {loss_metric_name})

    train_set_metrics = fold_model.evaluate(
        x=fold_split.X_train, y=fold_split.y_train, return_dict=True)
    evaluation_set_metrics = fold_model.evaluate(
        x=fold_split.X_evaluation, y=fold_split.y_evaluation, return_dict=True)
    return SingleFoldResults(
        evaluation_set_size=len(fold_split.y_evaluation),
        train_set_loss=train_set_metrics[loss_metric_name],
        train_set_main_metric=train_set_metrics[main_metric_name],
        evaluation_set_loss=evaluation_set_metrics[loss_metric_name],
        evaluation_set_main_metric=evaluation_set_metrics[main_metric_name],
    )


def _populate_report_with_k_fold_evaluation_results(
    folds_results: list[SingleFoldResults],
    folds_train_figures_folder: Path,
    session_context: TrainSessionContext,
) -> None:
    model_training = session_context.pipeline_report.model_training
    model_training.evaluation_folds_results = KFoldCrossValidationResults(fold_results=folds_results)
    model_training.evaluation_folds_train_figures_root_folder = folds_train_figures_folder
