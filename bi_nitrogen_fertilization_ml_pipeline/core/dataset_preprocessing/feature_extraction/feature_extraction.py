import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeatureKinds
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing.feature_extraction.categorical_features_one_hot_encoding import \
    fit_categorical_features_one_hot_encoding, transform_categorical_feature_one_hot_encoding
from bi_nitrogen_fertilization_ml_pipeline.core.utils.dataframe_utils import concat_dataframes_horizontally, \
    validate_dataframe_has_column


def fit_train_feature_extraction_artifacts(
    raw_train_dataset_df: pd.DataFrame,
    session_context: TrainSessionContext,
) -> None:
    fit_categorical_features_one_hot_encoding(raw_train_dataset_df, session_context)


def extract_features(
    raw_dataset_df: pd.DataFrame,
    train_artifacts: TrainArtifacts,
    for_inference: bool,
) -> pd.DataFrame:
    numeric_features_df = _extract_numeric_features(
        raw_dataset_df, train_artifacts)
    transformed_categorical_features = _extract_categorical_features(
        raw_dataset_df, train_artifacts, for_inference)

    transformed_dataset_df = concat_dataframes_horizontally([
        numeric_features_df,
        *transformed_categorical_features,
    ])
    return transformed_dataset_df


def extract_train_target(
    raw_train_dataset_df: pd.DataFrame,
    train_artifacts: TrainArtifacts,
) -> pd.Series:
    target_col_name = train_artifacts.features_config.target_column
    validate_dataframe_has_column(raw_train_dataset_df, target_col_name)
    target_col = raw_train_dataset_df[target_col_name].copy()
    return target_col


def extract_evaluation_folds_key(
    raw_train_dataset_df: pd.DataFrame,
    train_params: TrainParams,
) -> pd.Series:
    evaluation_folds_key_settings = train_params.evaluation_folds_key
    validate_dataframe_has_column(raw_train_dataset_df, evaluation_folds_key_settings.column)

    evaluation_folds_key_col = raw_train_dataset_df[evaluation_folds_key_settings.column].copy()
    # for simplicity’s sake, treat the values of this key column as strings
    evaluation_folds_key_col = evaluation_folds_key_col.astype(str)
    if evaluation_folds_key_settings.values_mapper is not None:
        evaluation_folds_key_col = evaluation_folds_key_col.apply(
            evaluation_folds_key_settings.values_mapper)

    return evaluation_folds_key_col


def _extract_categorical_features(
    raw_dataset_df: pd.DataFrame,
    train_artifacts: TrainArtifacts,
    for_inference: bool,
) -> list[pd.DataFrame]:
    one_hot_encoded_features_artifacts = train_artifacts.dataset_preprocessing.one_hot_encoded_features

    transformed_categorical_features = []
    for feature_col_name, one_hot_encoding_metadata in one_hot_encoded_features_artifacts.items():
        validate_dataframe_has_column(raw_dataset_df, feature_col_name)
        feature_col = raw_dataset_df[feature_col_name]
        one_hot_encoded_feature_df = transform_categorical_feature_one_hot_encoding(
            feature_col, feature_col_name, one_hot_encoding_metadata, for_inference,
        )
        transformed_categorical_features.append(one_hot_encoded_feature_df)
    return transformed_categorical_features


def _extract_numeric_features(
    raw_dataset_df,
    train_artifacts: TrainArtifacts,
) -> pd.DataFrame:
    feature_config = train_artifacts.features_config
    non_categorical_feature_columns = [
        feature.column
        for feature in feature_config.features
        if feature.kind != FeatureKinds.categorical
    ]

    for feature_col in non_categorical_feature_columns:
        validate_dataframe_has_column(raw_dataset_df, feature_col)
    raw_numeric_features_df = raw_dataset_df[non_categorical_feature_columns].copy()
    return raw_numeric_features_df
