from typing import Collection

import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.preprocessed_datasets import PreprocessedTrainDataset, \
    PreprocessedInferenceDataset
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing import imputation_manager
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing.feature_extraction import feature_extraction


def train_dataset_preprocessing(
    raw_train_dataset_df: pd.DataFrame,
    session_context: TrainSessionContext,
) -> PreprocessedTrainDataset:
    _validate_required_columns_present_in_input_dataset(
        raw_train_dataset_df, session_context.get_required_columns_for_training())

    original_raw_train_dataset_df = raw_train_dataset_df
    raw_train_dataset_df = raw_train_dataset_df.copy()

    imputation_manager.train_dataset_imputation(raw_train_dataset_df, session_context)
    feature_extraction.fit_train_feature_extraction_artifacts(raw_train_dataset_df, session_context)

    X = feature_extraction.extract_features(
        raw_train_dataset_df, session_context.artifacts, for_inference=False)
    y = feature_extraction.extract_train_target(
        raw_train_dataset_df, session_context.artifacts)
    evaluation_folds_key_col = feature_extraction.extract_evaluation_folds_key(
        raw_train_dataset_df, session_context.params)
    preprocessed_dataset = PreprocessedTrainDataset(X=X, y=y, evaluation_folds_key_col=evaluation_folds_key_col)

    _populate_preprocessing_general_details_in_pipeline_report(
        preprocessed_dataset, original_raw_train_dataset_df, session_context)
    return preprocessed_dataset


def inference_dataset_preprocessing(
    raw_inference_dataset_df: pd.DataFrame,
    training_artifacts: TrainArtifacts,
) -> PreprocessedInferenceDataset:
    # todo - move
    # assert training_artifacts.is_fitted, \
    #     "the provided train artifacts instance was not fitted, this isn't allowed."
    _validate_required_columns_present_in_input_dataset(
        raw_inference_dataset_df, training_artifacts.features_config.get_feature_columns())

    raw_inference_dataset_df = raw_inference_dataset_df.copy()

    imputation_manager.inference_dataset_validation(
        raw_inference_dataset_df, training_artifacts)
    X = feature_extraction.extract_features(
        raw_inference_dataset_df, training_artifacts, for_inference=True)
    return PreprocessedInferenceDataset(X=X)


def _validate_required_columns_present_in_input_dataset(
    input_dataset_df: pd.DataFrame,
    required_columns: Collection[str],
) -> None:
    for column in required_columns:
        assert column in input_dataset_df,\
            f"the required column '{column}' is missing in the input dataset"


def _populate_preprocessing_general_details_in_pipeline_report(
    preprocessed_dataset: PreprocessedTrainDataset,
    original_raw_train_dataset_df: pd.DataFrame,
    session_context: TrainSessionContext,
) -> None:
    session_context.pipeline_report.dataset_preprocessing.original_dataset = original_raw_train_dataset_df.copy()
    session_context.pipeline_report.dataset_preprocessing.preprocessed_dataset = \
        preprocessed_dataset.get_full_dataset()
    session_context.pipeline_report.dataset_preprocessing.required_columns_for_training = \
        session_context.get_required_columns_for_training()

    raw_dataset_columns_count = original_raw_train_dataset_df.shape[1]
    session_context.pipeline_report.dataset_preprocessing.unused_dropped_columns_count = \
        raw_dataset_columns_count - len(session_context.get_required_columns_for_training())
