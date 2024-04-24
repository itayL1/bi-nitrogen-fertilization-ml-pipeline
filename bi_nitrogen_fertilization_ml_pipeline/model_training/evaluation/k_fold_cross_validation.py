from contextlib import contextmanager
from pathlib import Path
from typing import Generator, ContextManager, Callable

import pandas as pd
from keras import Model
from keras.callbacks import History

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.k_fold_cross_validation import DatasetFoldSplit, \
    FoldModelEvaluationResults, KFoldCrossValidationResults
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.preprocessed_datasets import PreprocessedTrainDataset
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import PipelineModules
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.core.utils.statistical_tests import \
    values_distribution_gini_coefficient_test, DistributionBalance
from bi_nitrogen_fertilization_ml_pipeline.model_training.evaluation.random_guess_predictions import \
    calculate_evaluation_metric_for_random_guess_predictions
from bi_nitrogen_fertilization_ml_pipeline.model_training.training.model_setup import prepare_new_model_for_training
from bi_nitrogen_fertilization_ml_pipeline.model_training.training.train_model import train_model
from bi_nitrogen_fertilization_ml_pipeline.model_training.utils.keras_utils import extract_train_epochs_count


def key_based_k_fold_cross_validation(
    preprocessed_train_dataset: PreprocessedTrainDataset,
    session_context: TrainSessionContext,
) -> None:
    _population_report_with_k_fold_evaluation_stats(preprocessed_train_dataset, session_context)

    folds_train_figures_folder = session_context.wip_outputs_folder_path / 'folds_train_figures'
    model_input_features_count = preprocessed_train_dataset.get_train_features_count()

    folds_results = []
    with _folds_display_progress_bar(preprocessed_train_dataset, session_context) as advance_pbar:
        for fold_split in _split_dataset_to_folds_based_on_key_col(preprocessed_train_dataset):
            fold_key = fold_split.fold_key
            fold_train_output_figures_folder = _setup_fold_train_output_figures_folder(
                fold_key, folds_train_figures_folder, session_context)
            fold_results = _run_fold_iteration(
                fold_split, model_input_features_count, fold_train_output_figures_folder, session_context,
            )
            folds_results.append(fold_results)
            advance_pbar()

    _populate_report_with_k_fold_evaluation_results(
        folds_results, folds_train_figures_folder, session_context)


def _calc_folds_count(preprocessed_train_dataset: PreprocessedTrainDataset) -> int:
    return len(preprocessed_train_dataset.evaluation_folds_key_col.unique())


def _population_report_with_k_fold_evaluation_stats(
    preprocessed_train_dataset: PreprocessedTrainDataset,
    session_context: TrainSessionContext,
) -> None:
    pipeline_report = session_context.pipeline_report
    evaluation_folds_key_col = preprocessed_train_dataset.evaluation_folds_key_col

    eval_folds_count = _calc_folds_count(preprocessed_train_dataset)
    if eval_folds_count < 3:
        pipeline_report.add_warning(
            PipelineModules.model_training,
            "the number of evaluation folds is low",
            context={
                'evaluation_folds_count': eval_folds_count,
            },
        )
    elif eval_folds_count > 30:
        pipeline_report.add_warning(
            PipelineModules.model_training,
            "the number of evaluation folds is high",
            context={
                'evaluation_folds_count': eval_folds_count,
            },
        )

    eval_folds_gini_test_res = values_distribution_gini_coefficient_test(evaluation_folds_key_col)
    pipeline_report.model_training.evaluation_folds_distribution_gini_coefficient = eval_folds_gini_test_res.gini_coefficient
    if eval_folds_gini_test_res.distribution_balance != DistributionBalance.relatively_balanced:
        pipeline_report.add_warning(
            PipelineModules.model_training,
            "the distribution of the evaluation folds is considered imbalanced "
            "according to the gini coefficient test",
            context={
                'evaluation_folds_distribution_gini_coefficient_test_results': eval_folds_gini_test_res,
            },
        )


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


@contextmanager
def _folds_display_progress_bar(
    preprocessed_train_dataset: PreprocessedTrainDataset,
    session_context: TrainSessionContext,
) -> ContextManager[Callable[[], None]]:
    rich_progress = session_context.rich_progress

    folds_count = _calc_folds_count(preprocessed_train_dataset)
    folds_progress_rich_task_id = rich_progress.add_task(
        description="[green]Evaluation folds progress", total=folds_count, start=True)

    def _advance_pbar():
        rich_progress.update(folds_progress_rich_task_id, advance=1)

    try:
        yield _advance_pbar
    finally:
        rich_progress.remove_task(folds_progress_rich_task_id)


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
) -> FoldModelEvaluationResults:
    fold_model = prepare_new_model_for_training(
        session_context.params, model_input_features_count)
    train_history = train_model(
        fold_model,
        X=fold_split.X_train,
        y=fold_split.y_train,
        train_params=session_context.params,
        output_figures_folder_path=train_output_figures_folder,
    )
    fold_results = _evaluate_fold_model(
        fold_model, fold_split, train_history, session_context)
    return fold_results


def _evaluate_fold_model(
    fold_model: Model,
    fold_split: DatasetFoldSplit,
    train_history: History,
    session_context: TrainSessionContext,
) -> FoldModelEvaluationResults:
    train_set_loss, train_set_main_metric, = _get_fold_model_evaluation_metrics_for_dataset(
        fold_model, fold_split.X_train, fold_split.y_train)
    evaluation_set_loss, evaluation_set_main_metric, = _get_fold_model_evaluation_metrics_for_dataset(
        fold_model, fold_split.X_evaluation, fold_split.y_evaluation)

    train_params = session_context.params
    train_random_guess_on_evaluation_set_loss = calculate_evaluation_metric_for_random_guess_predictions(
        fold_split.y_train, fold_split.y_evaluation, train_params.loss_function)
    train_random_guess_on_evaluation_set_main_metric = calculate_evaluation_metric_for_random_guess_predictions(
        fold_split.y_train, fold_split.y_evaluation, train_params.evaluation_metric)

    return FoldModelEvaluationResults(
        fold_key=fold_split.fold_key,
        train_epochs_count=extract_train_epochs_count(train_history),
        evaluation_set_size=len(fold_split.y_evaluation),
        train_set_loss=train_set_loss,
        train_set_main_metric=train_set_main_metric,
        evaluation_set_loss=evaluation_set_loss,
        evaluation_set_main_metric=evaluation_set_main_metric,
        train_random_guess_on_evaluation_set_loss=train_random_guess_on_evaluation_set_loss,
        train_random_guess_on_evaluation_set_main_metric=train_random_guess_on_evaluation_set_main_metric,
    )


def _get_fold_model_evaluation_metrics_for_dataset(
    fold_model: Model, X: pd.DataFrame, y: pd.Series,
) -> tuple[float, float]:
    loss_metric_name, main_metric_name = _extract_evaluation_metric_names(fold_model)
    dataset_set_evaluation_metrics = fold_model.evaluate(
        x=X, y=y, return_dict=True, verbose=0)
    loss = dataset_set_evaluation_metrics[loss_metric_name]
    main_metric = dataset_set_evaluation_metrics[main_metric_name]
    return loss, main_metric


def _extract_evaluation_metric_names(fold_model: Model) -> tuple[str, str]:
    loss_metric_name = 'loss'
    model_metric_names = set(fold_model.metrics_names)

    if model_metric_names == {loss_metric_name}:
        return loss_metric_name, loss_metric_name
    elif len(model_metric_names) == 2 and loss_metric_name in model_metric_names:
        main_metric_name, = list(model_metric_names - {loss_metric_name})
        return loss_metric_name, main_metric_name
    else:
        raise ValueError('could not extract the evaluation metric names')


def _populate_report_with_k_fold_evaluation_results(
    folds_results: list[FoldModelEvaluationResults],
    folds_train_figures_folder: Path,
    session_context: TrainSessionContext,
) -> None:
    model_training = session_context.pipeline_report.model_training
    model_training.evaluation_folds_results = KFoldCrossValidationResults(fold_results=folds_results)
    model_training.evaluation_folds_train_figures_root_folder = folds_train_figures_folder
