import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams


def parse_input_features_config(features_config_dict: dict) -> FeaturesConfig:
    try:
        return FeaturesConfig.parse_obj(features_config_dict)
    except Exception as ex:
        raise AssertionError('the provided features config is invalid') from ex


def parse_input_train_params(train_params_dict: dict) -> TrainParams:
    try:
        return TrainParams.parse_obj(train_params_dict)
    except Exception as ex:
        raise AssertionError('the provided train params are invalid') from ex


def validate_input_train_dataset(raw_train_dataset_df: pd.DataFrame) -> None:
    _validate_input_train_dataset_structure(raw_train_dataset_df)


def invalid_input_train_dataset_error(reason: str) -> AssertionError:
    return AssertionError(f'the provided train dataset is invalid. reason: {reason}')


def _validate_input_train_dataset_structure(raw_train_dataset_df: pd.DataFrame) -> None:
    if raw_train_dataset_df.ndim != 2:
        raise invalid_input_train_dataset_error(
            f'the dataset dataframe must have exactly 2 dimensions, but the '
            f'provided dataframe has {raw_train_dataset_df.ndim} dimensions'
        )

    train_dataset_rows_count, _ = raw_train_dataset_df.shape
    if train_dataset_rows_count == 0:
        raise invalid_input_train_dataset_error('the input dataset must not be empty')
