import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing.feature_extraction.categorical_features_one_hot_encoding import \
    fit_categorical_features_one_hot_encoding


def training_feature_extraction(
    raw_train_dataset_df: pd.DataFrame,
    session_context: TrainSessionContext,
) -> pd.DataFrame:
    ret_train_features_df = raw_train_dataset_df.copy()

    fit_categorical_features_one_hot_encoding(ret_train_features_df, session_context)
    return ret_train_features_df


def inference_feature_extraction(
    raw_inference_dataset_df: pd.DataFrame,
    training_artifacts: TrainArtifacts,
) -> pd.DataFrame:
    raise NotImplementedError
