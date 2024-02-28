import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts, \
    DatasetPreprocessingArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing.feature_extraction.categorical_features_one_hot_encoding import \
    fit_categorical_features_one_hot_encoding, transform_categorical_features_one_hot_encoding


def fit_train_feature_extraction_artifacts(
    raw_train_dataset_df: pd.DataFrame,
    session_context: TrainSessionContext,
) -> None:
    fit_categorical_features_one_hot_encoding(raw_train_dataset_df, session_context)


def extract_features_from_raw_dataset(
    raw_dataset_df: pd.DataFrame,
    dataset_preprocessing_artifacts: DatasetPreprocessingArtifacts,
    for_inference: bool,
) -> None:
    transform_categorical_features_one_hot_encoding(
        raw_dataset_df,
        dataset_preprocessing_artifacts.one_hot_encoded_features,
        for_inference,
    )
