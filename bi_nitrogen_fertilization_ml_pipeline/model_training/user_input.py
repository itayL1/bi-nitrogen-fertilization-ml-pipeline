import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig


def parse_input_features_config(features_config_dict: dict) -> FeaturesConfig:
    try:
        return FeaturesConfig.parse_obj(features_config_dict)
    except Exception as ex:
        raise AssertionError('the provided features config is invalid') from ex


def validate_input_train_dataset(raw_train_dataset_df: pd.DataFrame, features_config: FeaturesConfig) -> None:
    _validate_input_train_dataset_structure(raw_train_dataset_df)
    _validate_input_train_dataset_aligned_with_features_config(raw_train_dataset_df, features_config)


def invalid_input_train_dataset_error(reason: str) -> AssertionError:
    return AssertionError(f'the provided train dataset is invalid. reason: {reason}')


def _validate_input_train_dataset_structure(raw_train_dataset_df: pd.DataFrame) -> None:
    train_dataset_shape = raw_train_dataset_df.shape
    train_dataset_dimension = len(train_dataset_shape)
    if train_dataset_dimension != 2:
        raise invalid_input_train_dataset_error(
            f'the dataset dataframe must have exactly 2 dimensions, but the '
            f'provided dataframe has {train_dataset_dimension} dimensions'
        )

    train_dataset_rows_count, _ = train_dataset_shape
    if train_dataset_rows_count == 0:
        raise invalid_input_train_dataset_error('the input dataset must not be empty')


def _validate_input_train_dataset_aligned_with_features_config(
    raw_train_dataset_df: pd.DataFrame,
    features_config: FeaturesConfig,
) -> None:
    train_dataset_columns = set(raw_train_dataset_df.columns)
    features_config_columns = {
        features_config.target_column,
        *(feature_col for feature_col in features_config.features.keys())
    }

    feature_columns_missing_in_actual_train_dataset = train_dataset_columns - features_config_columns
    if any(feature_columns_missing_in_actual_train_dataset):
        raise invalid_input_train_dataset_error(
            f'detected feature columns that are missing in the input train dataset. the missing '
            f'feature columns are - {sorted(feature_columns_missing_in_actual_train_dataset)}'
        )
