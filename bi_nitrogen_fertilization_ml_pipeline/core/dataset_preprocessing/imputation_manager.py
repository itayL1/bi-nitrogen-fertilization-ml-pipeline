import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import ImputationFunnel
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext


def train_dataset_imputation(
    raw_train_dataset_df: pd.DataFrame,
    session_context: TrainSessionContext,
) -> pd.DataFrame:
    ret_dataset = raw_train_dataset_df.copy()
    features_config = session_context.artifacts.features_config

    rows_count_before_imputation = ret_dataset.shape[0]
    ret_dataset = ret_dataset.dropna(subset=list(features_config.get_features_and_target_columns()))
    rows_count_after_imputation = ret_dataset.shape[0]

    session_context.pipeline_report.dataset_preprocessing.imputation_funnel = ImputationFunnel(
        rows_count_before_imputation=rows_count_before_imputation,
        rows_count_after_imputation=rows_count_after_imputation,
    )
    return ret_dataset


def inference_dataset_validation(raw_inference_dataset_df: pd.DataFrame) -> None:
    raise NotImplementedError
