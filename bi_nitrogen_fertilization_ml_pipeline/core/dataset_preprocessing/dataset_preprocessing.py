import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing import imputation_manager
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing.feature_extraction import feature_extraction


def train_dataset_preprocessing(
    raw_train_dataset_df: pd.DataFrame,
    session_context: TrainSessionContext,
) -> pd.DataFrame:
    preprocessed_train_dataset_df = raw_train_dataset_df.copy()

    imputation_manager.train_dataset_imputation(
        preprocessed_train_dataset_df, session_context)
    feature_extraction.training_feature_extraction(
        preprocessed_train_dataset_df, session_context)

    return preprocessed_train_dataset_df


def inference_dataset_preprocessing(
    raw_inference_dataset_df: pd.DataFrame,
    training_artifacts: TrainArtifacts,
) -> pd.DataFrame:
    preprocessed_inference_dataset_df = raw_inference_dataset_df.copy()

    imputation_manager.inference_dataset_validation(
        preprocessed_inference_dataset_df, training_artifacts)
    feature_extraction.inference_feature_extraction(
        preprocessed_inference_dataset_df, training_artifacts)

    return preprocessed_inference_dataset_df
