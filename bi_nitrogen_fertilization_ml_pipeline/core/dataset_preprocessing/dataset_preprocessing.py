import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import DatasetPreprocessingArtifacts, \
    TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing import imputation_manager


def train_dataset_preprocessing(
    raw_train_dataset_df: pd.DataFrame,
    session_context: TrainSessionContext,
) -> tuple[pd.DataFrame, DatasetPreprocessingArtifacts]:
    preprocessed_dataset_df = raw_train_dataset_df.copy()

    preprocessed_dataset_df = imputation_manager.train_dataset_imputation(
        preprocessed_dataset_df, session_context)

    raise NotImplementedError
    return preprocessed_dataset_df


def inference_dataset_preprocessing(
    raw_inference_dataset_df: pd.DataFrame,
    training_artifacts: TrainArtifacts,
) -> pd.DataFrame:
    raise NotImplementedError
