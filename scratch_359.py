import numpy as np
import pandas as pd
import keras.optimizers.legacy

from bi_nitrogen_fertilization_ml_pipeline.assets.baseline_model import init_baseline_model
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig, FeatureSettings, \
    FeatureKinds
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams, \
    TrainEarlyStoppingSettings, EvaluationFoldsSplitSettings
from bi_nitrogen_fertilization_ml_pipeline.main_api import train_and_evaluate_model

column_dtypes = {
    'משק': str,
    'עומק': float,
    'קוד  קרקע': str,
    'Krab_key_eng': str,
    'YLD_Israel(1000T)': float,
    'Precip_Gilat': float,
}
train_dataset_df = pd.read_csv(
    '/Users/itaylotan/Downloads/Cutted_Hizuy_with_met_50percent.xlsx - Sheet1.csv',
    low_memory=False,
    dtype=column_dtypes,
)

train_params = TrainParams(
    model_builder=init_baseline_model,
    epochs_count=100,
    # epochs_count=5,
    evaluation_folds_split=EvaluationFoldsSplitSettings(
        by_key_column='fold_id',
        # values_mapper=lambda year_str: str(int(year_str.strip()) % 3),
    ),
    early_stopping=TrainEarlyStoppingSettings(
        validation_set_fraction_size=0.2,
        tolerance_epochs_count=9,
        # tolerance_epochs_count=2,
    ),
    optimizer_builder=keras.optimizers.legacy.Adam,
    random_seed=42,
    silent_models_fitting=True,
)


features_config = FeaturesConfig(
    target_column='N kg/d',
    features=[
        FeatureSettings(
            key_column='משק',
            kind=FeatureKinds.categorical,
        ),
        FeatureSettings(
            key_column='עומק',
            kind=FeatureKinds.numeric,
        ),
        FeatureSettings(
            key_column='קוד  קרקע',
            kind=FeatureKinds.categorical,
        ),
        FeatureSettings(
            key_column='Krab_key_eng',
            kind=FeatureKinds.categorical,
        ),
        # FeatureSettings(
        #     column='YLD_Israel(1000T)',
        #     kind=FeatureKinds.numeric,
        # ),
        FeatureSettings(
            key_column='Precip_Gilat',
            kind=FeatureKinds.numeric,
        ),
        # *[
        #     FeatureSettings(
        #         column=feature_name,
        #         kind=FeatureKinds.numeric,
        #     ) for feature_name in (
        #         'LST_-5', 'LST_-4', 'LST_-3', 'LST_-2', 'LST_-1', 'LST_0', 'LST_1', 'LST_2', 'LST_3', 'LST_4', 'LST_5', 'LST_6', 'temperature_2m_-5', 'temperature_2m_-4', 'temperature_2m_-3', 'temperature_2m_-2', 'temperature_2m_-1', 'temperature_2m_0', 'temperature_2m_1', 'temperature_2m_2', 'temperature_2m_3', 'temperature_2m_4', 'temperature_2m_5', 'temperature_2m_6', 'total_precipitation_sum_-5', 'total_precipitation_sum_-4', 'total_precipitation_sum_-3', 'total_precipitation_sum_-2', 'total_precipitation_sum_-1', 'total_precipitation_sum_0', 'total_precipitation_sum_1', 'total_precipitation_sum_2', 'total_precipitation_sum_3', 'total_precipitation_sum_4', 'total_precipitation_sum_5', 'total_precipitation_sum_6', 'NDVI_-5', 'NDVI_-4', 'NDVI_-3', 'NDVI_-2', 'NDVI_-1', 'NDVI_0', 'NDVI_1', 'NDVI_2', 'NDVI_3', 'NDVI_4', 'NDVI_5', 'NDVI_6'
        #     )
        # ]
    ]
)

# print('train_dataset_df.shape before', train_dataset_df.shape)
# train_dataset_df = train_dataset_df.dropna(subset=['קוד  קרקע'])
# print('train_dataset_df.shape after', train_dataset_df.shape)
# asdsad

FOLDS_COUNT = 5
rng = np.random.default_rng(seed=42)
train_dataset_df['fold_id'] = rng.integers(1, FOLDS_COUNT + 1, size=train_dataset_df.shape[0])

if __name__ == '__main__':
    train_and_evaluate_model(
        train_dataset_df,
        features_config_dict=features_config.dict(),
        train_params_dict=train_params.dict(),
        output_model_file_path='/Users/itaylotan/git/bi-nitrogen-fertilization-ml-pipeline/scratch_359_outputs4/model.zip',
    )
