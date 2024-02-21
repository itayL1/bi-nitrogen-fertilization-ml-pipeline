import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig


class FeatureExtractionTransformer:
    def __init__(self, feature_config: FeaturesConfig):
        self.feature_config = feature_config

    def fit(self, raw_train_dataset_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def transform(self, raw_dataset_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
