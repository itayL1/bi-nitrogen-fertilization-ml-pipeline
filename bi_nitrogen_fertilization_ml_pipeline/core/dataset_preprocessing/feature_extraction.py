import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes import train_pipeline_report
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import DatasetPreprocessingArtifacts, \
    TrainArtifacts, OneHotEncodedFeatures


def training_feature_extraction(
    raw_train_dataset_df: pd.DataFrame,
    features_config: FeaturesConfig,
) -> tuple[pd.DataFrame, DatasetPreprocessingArtifacts, pipeline_report.CategoricalFeaturesEncodingDetails]:
    train_features_df = raw_train_dataset_df.copy()

    train_features_df, one_hot_encoded_features, encoding_details_for_report = \
        _apply_features_one_hot_encoding(train_features_df, features_config)
    return train_features_df


def inference_feature_extraction(
    raw_inference_dataset_df: pd.DataFrame,
    training_artifacts: TrainArtifacts,
) -> pd.DataFrame:
    raise NotImplementedError


def _apply_features_one_hot_encoding(
    raw_train_dataset_df: pd.DataFrame,
    features_config: FeaturesConfig,
) -> tuple[pd.DataFrame, OneHotEncodedFeatures, pipeline_report.CategoricalFeaturesEncodingDetails]:
    raise NotImplementedError
