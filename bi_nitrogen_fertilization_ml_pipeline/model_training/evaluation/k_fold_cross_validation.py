from typing import Generator

import pandas as pd
from tqdm import tqdm

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.k_fold_cross_validation import DatasetFoldSplit
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.preprocessed_datasets import PreprocessedTrainDataset
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.model_training.training.model_setup import prepare_new_model_for_training
from bi_nitrogen_fertilization_ml_pipeline.model_training.training.train_model import train_model


def key_based_k_fold_cross_validation(
    preprocessed_train_dataset: PreprocessedTrainDataset,
    session_context: TrainSessionContext,
) -> None:
    _populate_folds_split_ratios_in_pipeline_report(
        preprocessed_train_dataset, session_context)
    folds_train_figures_folder = session_context.temp_wip_outputs_folder_path / 'folds_train_figures'
    folds_key_field_name = session_context.params.evaluation_folds_key.column

    folds_count = _calc_folds_count(preprocessed_train_dataset)
    with tqdm('Evaluation cross validation folds loop', total=folds_count) as pbar:
        for fold_split in _split_dataset_to_folds_based_on_key_col(preprocessed_train_dataset):
            fold_model = prepare_new_model_for_training(
                session_context.params, preprocessed_train_dataset)
            train_output_figures_folder =\
                folds_train_figures_folder / f'fold__{folds_key_field_name}_{fold_split.fold_key}'
            train_output_figures_folder.mkdir(parents=True, exist_ok=True)
            train_model(
                fold_model,
                X=fold_split.X_train,
                y=fold_split.y_train,
                train_params=session_context.params,
                output_figures_folder_path=train_output_figures_folder,
            )
            aaaa = fold_model.evaluate(x=fold_split.X_evaluation, y=fold_split.y_evaluation)

            pbar.update()


def _calc_folds_count(preprocessed_train_dataset: PreprocessedTrainDataset) -> int:
    return len(preprocessed_train_dataset.evaluation_folds_key_col.unique())


def _split_dataset_to_folds_based_on_key_col(
    preprocessed_train_dataset: PreprocessedTrainDataset,
) -> Generator[DatasetFoldSplit, None, None]:
    X = preprocessed_train_dataset.X
    y = preprocessed_train_dataset.y
    evaluation_folds_key_col = preprocessed_train_dataset.evaluation_folds_key_col

    for fold_key, group_segment in _group_series_by_values(evaluation_folds_key_col):
        # try:
        yield DatasetFoldSplit(
            fold_key=str(fold_key),
            X_train=X.iloc[~group_segment.index],
            y_train=y.iloc[~group_segment.index],
            X_evaluation=X.iloc[group_segment.index],
            y_evaluation=y.iloc[group_segment.index],
        )
        # except Exception as ex:
        #     x = 1


def _populate_folds_split_ratios_in_pipeline_report(
    preprocessed_train_dataset: PreprocessedTrainDataset,
    session_context: TrainSessionContext,
) -> None:
    fold_key_to_evaluation_set_size = {
        fold_split.fold_key: fold_split.X_evaluation.shape[0]
        for fold_split
        in _split_dataset_to_folds_based_on_key_col(preprocessed_train_dataset)
    }


def _group_series_by_values(series: pd.Series):
    return series.groupby(series)
