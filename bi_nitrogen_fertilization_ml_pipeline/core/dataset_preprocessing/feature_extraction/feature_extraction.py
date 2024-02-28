import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing.feature_extraction.categorical_features_one_hot_encoding import \
    fit_categorical_features_one_hot_encoding, transform_categorical_features_one_hot_encoding


def training_feature_extraction(
    train_dataset_df: pd.DataFrame,
    session_context: TrainSessionContext,
) -> None:
    fit_categorical_features_one_hot_encoding(
        train_dataset_df, session_context)
    fitted_one_hot_encoded_features = session_context.artifacts.dataset_preprocessing.one_hot_encoded_features
    transform_categorical_features_one_hot_encoding(
        train_dataset_df, fitted_one_hot_encoded_features, for_inference=False)


def inference_feature_extraction(
    inference_dataset_df: pd.DataFrame,
    training_artifacts: TrainArtifacts,
) -> None:
    one_hot_encoded_features_artifacts = training_artifacts.dataset_preprocessing.one_hot_encoded_features
    transform_categorical_features_one_hot_encoding(
        inference_dataset_df, one_hot_encoded_features_artifacts, for_inference=True)
