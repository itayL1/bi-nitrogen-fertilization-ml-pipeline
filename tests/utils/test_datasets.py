from pathlib import Path

import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig, FeatureSettings, \
    FeatureKinds, OneHotEncodingSettings


def load_Nitrogen_with_Era5_and_NDVI_dataset() -> pd.DataFrame:
    current_file_path = Path(__file__)
    dataset_csv_file_path = current_file_path.parent / '../assets/Nitrogen_with_Era5_and_NDVI.xlsx - Sheet1.csv'
    column_dtypes = {
        'משק': str,
        'עומק': float,
        'קוד  קרקע': str,
        'Krab_key_eng': str,
        'YLD_Israel(1000T)': float,
        'Precip_Gilat': float,
    }
    return pd.read_csv(dataset_csv_file_path, low_memory=False, dtype=column_dtypes)


def default_Nitrogen_with_Era5_and_NDVI_dataset_features_config() -> FeaturesConfig:
    return FeaturesConfig(
        target_column='N kg/d',
        features=[
            FeatureSettings(
                column='משק',
                kind=FeatureKinds.categorical,
            ),
            FeatureSettings(
                column='עומק',
                kind=FeatureKinds.numeric,
            ),
            FeatureSettings(
                column='קוד  קרקע',
                kind=FeatureKinds.categorical,
            ),
            FeatureSettings(
                column='Krab_key_eng',
                kind=FeatureKinds.categorical,
            ),
            FeatureSettings(
                column='YLD_Israel(1000T)',
                kind=FeatureKinds.numeric,
            ),
            FeatureSettings(
                column='Precip_Gilat',
                kind=FeatureKinds.numeric,
            ),
        ]
    )
