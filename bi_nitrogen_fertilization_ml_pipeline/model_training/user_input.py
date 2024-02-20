import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig


def validate_input_train_dataset(train_dataset_df: pd.DataFrame) -> None:
    pass


def parse_input_features_config(features_config) -> FeaturesConfig:
    try:
        return FeaturesConfig.parse_obj(features_config)
    except Exception as ex:
        raise AssertionError('the provided features config is invalid') from ex
