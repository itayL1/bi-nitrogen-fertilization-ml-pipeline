import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import ImputationFunnel
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.core.pipeline_report.display_utils import to_displayable_percentage


def train_dataset_imputation(
    train_dataset_df: pd.DataFrame,
    session_context: TrainSessionContext,
) -> None:
    features_config = session_context.artifacts.features_config

    rows_count_before_imputation = train_dataset_df.shape[0]
    train_dataset_df.replace('', None, inplace=True)
    train_dataset_df.dropna(
        subset=list(features_config.get_features_and_target_columns()), inplace=True)
    rows_count_after_imputation = train_dataset_df.shape[0]

    remaining_rows_percentage = (rows_count_after_imputation / rows_count_before_imputation) * 100
    session_context.pipeline_report.dataset_preprocessing.imputation_funnel = ImputationFunnel(
        remaining_rows_percentage=to_displayable_percentage(remaining_rows_percentage),
        rows_count_before_imputation=rows_count_before_imputation,
        rows_count_after_imputation=rows_count_after_imputation,
    )


def inference_dataset_validation(
    inference_dataset_df: pd.DataFrame,
    training_artifacts: TrainArtifacts,
) -> None:
    features_config = training_artifacts.features_config

    inference_dataset_relevant_columns_df = inference_dataset_df[
        list(features_config.get_features_and_target_columns())
    ]
    required_columns_with_empty_values = _find_columns_with_empty_values(
        inference_dataset_relevant_columns_df)
    assert not any(required_columns_with_empty_values),\
        f'during inference time, the feature and target of the input dataset ' \
        f'columns must not contain empty values. however, empty values were ' \
        f'detected in the following columns - {required_columns_with_empty_values}'


def _find_columns_with_empty_values(df: pd.DataFrame) -> set[str]:
    return set(
        df.columns[(df.isna() | df == '').any()]
    )
