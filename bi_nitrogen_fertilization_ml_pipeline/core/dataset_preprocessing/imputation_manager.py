import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig


def train_dataset_imputation(
    raw_train_dataset_df: pd.DataFrame,
    features_config: FeaturesConfig,
) -> pd.DataFrame:
    ret_dataset = raw_train_dataset_df.copy()

    ret_dataset = ret_dataset.dropna(subset=list(features_config.get_features_and_target_columns()))


def inference_dataset_validation(raw_inference_dataset_df: pd.DataFrame) -> None:
    raise NotImplementedError
