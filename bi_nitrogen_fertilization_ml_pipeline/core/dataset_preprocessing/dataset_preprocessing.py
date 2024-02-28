import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing import imputation_manager
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing.feature_extraction import feature_extraction


def train_dataset_preprocessing(
    raw_train_dataset_df: pd.DataFrame,
    session_context: TrainSessionContext,
) -> pd.DataFrame:
    assert not session_context.artifacts.is_fitted, \
        "the provided train artifacts instance was fitted already, this isn't allowed."

    session_context.pipeline_report.dataset_preprocessing.original_dataset = raw_train_dataset_df.copy()
    preprocessed_train_dataset_df = raw_train_dataset_df.copy()

    _remove_unused_columns_from_dataset(
        preprocessed_train_dataset_df, features_config=session_context.artifacts.features_config)
    unused_dropped_columns_count = len(raw_train_dataset_df.columns) - len(preprocessed_train_dataset_df.columns)
    session_context.pipeline_report.dataset_preprocessing.unused_dropped_columns_count = unused_dropped_columns_count

    imputation_manager.train_dataset_imputation(preprocessed_train_dataset_df, session_context)
    feature_extraction.fit_train_feature_extraction_artifacts(preprocessed_train_dataset_df, session_context)
    feature_extraction.extract_features_from_raw_dataset(
        raw_dataset_df=preprocessed_train_dataset_df,
        dataset_preprocessing_artifacts=session_context.artifacts.dataset_preprocessing,
        for_inference=False,
    )

    session_context.artifacts.is_fitted = True
    session_context.pipeline_report.dataset_preprocessing.preprocessed_dataset = preprocessed_train_dataset_df.copy()
    return preprocessed_train_dataset_df


def inference_dataset_preprocessing(
    raw_inference_dataset_df: pd.DataFrame,
    training_artifacts: TrainArtifacts,
) -> pd.DataFrame:
    assert training_artifacts.is_fitted, \
        "the provided train artifacts instance was not fitted, this isn't allowed."

    preprocessed_inference_dataset_df = raw_inference_dataset_df.copy()

    _remove_unused_columns_from_dataset(
        preprocessed_inference_dataset_df, features_config=training_artifacts.features_config)
    imputation_manager.inference_dataset_validation(
        preprocessed_inference_dataset_df, training_artifacts)
    feature_extraction.extract_features_from_raw_dataset(
        raw_dataset_df=preprocessed_inference_dataset_df,
        dataset_preprocessing_artifacts=training_artifacts.dataset_preprocessing,
        for_inference=True,
    )

    return preprocessed_inference_dataset_df


def _remove_unused_columns_from_dataset(
    raw_dataset_df: pd.DataFrame,
    features_config: FeaturesConfig,
) -> None:
    columns_to_remove =\
        set(raw_dataset_df.columns) - set(features_config.get_features_and_target_columns())
    raw_dataset_df.drop(
        columns=list(columns_to_remove),
        inplace=True,
    )
