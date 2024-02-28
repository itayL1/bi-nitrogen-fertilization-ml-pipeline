import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import ImputationFunnel
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext


def train_dataset_imputation(
    train_dataset_df: pd.DataFrame,
    session_context: TrainSessionContext,
) -> None:
    features_config = session_context.artifacts.features_config

    rows_count_before_imputation = train_dataset_df.shape[0]
    train_dataset_df.dropna(
        subset=list(features_config.get_features_and_target_columns()), inplace=True)
    rows_count_after_imputation = train_dataset_df.shape[0]

    session_context.pipeline_report.dataset_preprocessing.imputation_funnel = ImputationFunnel(
        rows_count_before_imputation=rows_count_before_imputation,
        rows_count_after_imputation=rows_count_after_imputation,
    )


def inference_dataset_validation(raw_inference_dataset_df: pd.DataFrame) -> None:
    raise NotImplementedError
