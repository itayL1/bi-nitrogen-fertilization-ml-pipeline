import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import DatasetPreprocessingArtifacts, \
    TrainArtifacts, OneHotEncodedFeatures
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext


def training_feature_extraction(
    raw_train_dataset_df: pd.DataFrame,
    session_context: TrainSessionContext,
) -> pd.DataFrame:
    train_features_df = raw_train_dataset_df.copy()

    train_features_df, one_hot_encoded_features, encoding_details_for_report = \
        _apply_features_one_hot_encoding(train_features_df, session_context)
    return train_features_df


def inference_feature_extraction(
    raw_inference_dataset_df: pd.DataFrame,
    training_artifacts: TrainArtifacts,
) -> pd.DataFrame:
    raise NotImplementedError


def _apply_features_one_hot_encoding(
    raw_train_dataset_df: pd.DataFrame,
    session_context: TrainSessionContext,
) -> pd.DataFrame:
    features_config = session_context.artifacts.features_config
    raise NotImplementedError
