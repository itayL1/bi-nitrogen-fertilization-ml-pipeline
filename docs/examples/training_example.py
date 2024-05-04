# ** Imports **
from pathlib import Path

import pandas as pd
import keras.optimizers.legacy

from bi_nitrogen_fertilization_ml_pipeline.main_api import train_and_evaluate_model
from bi_nitrogen_fertilization_ml_pipeline.assets import init_baseline_model

# ** Preparations **

# Load dataset

TRAIN_DATASET_FILE_PATH = Path(__file__).parent / 'datasets/sample_dataset.csv'

column_dtypes = {
    'N kg/d': float,
    'עומק': float,
    'קוד  קרקע': str,
    'Krab_key_eng': str,
    'YLD_Israel(1000T)': float,
    'Precip_Gilat': float,
    'year': str,
}
train_dataset_df = pd.read_csv(
    TRAIN_DATASET_FILE_PATH, low_memory=False, dtype=column_dtypes,
)

# Configuration definition

features_config = {
    'target_column': 'N kg/d',
    'features': [
        # {
        #     'column': 'משק',
        #     'kind': 'categorical',
        # },
        {
            'column': 'עומק',
            'kind': 'numeric',
        },
        {
            'column': 'קוד  קרקע',
            'kind': 'categorical',
        },
        {
            'column': 'Krab_key_eng',
            'kind': 'categorical',
        },
        # {
        #     'column': 'YLD_Israel(1000T)',
        #     'kind': 'numeric',
        # },
        {
            'column': 'Precip_Gilat',
            'kind': 'numeric',
        },
        # *[
        #     {
        #         'column': feature_name,
        #         'kind': 'numeric',
        #     } for feature_name in (
        #         'LST_-5', 'LST_-4', 'LST_-3', 'LST_-2', 'LST_-1', 'LST_0', 'LST_1', 'LST_2', 'LST_3', 'LST_4', 'LST_5', 'LST_6', 'temperature_2m_-5', 'temperature_2m_-4', 'temperature_2m_-3', 'temperature_2m_-2', 'temperature_2m_-1', 'temperature_2m_0', 'temperature_2m_1', 'temperature_2m_2', 'temperature_2m_3', 'temperature_2m_4', 'temperature_2m_5', 'temperature_2m_6', 'total_precipitation_sum_-5', 'total_precipitation_sum_-4', 'total_precipitation_sum_-3', 'total_precipitation_sum_-2', 'total_precipitation_sum_-1', 'total_precipitation_sum_0', 'total_precipitation_sum_1', 'total_precipitation_sum_2', 'total_precipitation_sum_3', 'total_precipitation_sum_4', 'total_precipitation_sum_5', 'total_precipitation_sum_6', 'NDVI_-5', 'NDVI_-4', 'NDVI_-3', 'NDVI_-2', 'NDVI_-1', 'NDVI_0', 'NDVI_1', 'NDVI_2', 'NDVI_3', 'NDVI_4', 'NDVI_5', 'NDVI_6'
        #     )
        # ]
    ]
}

train_params = dict(
    model_builder=init_baseline_model,
    loss_function='mse',
    evaluation_metric='rmse',
    epochs_count=100,
    evaluation_folds_split=dict(
        by_key_column='year',
        # by_folds_number=5,
    ),
    early_stopping=dict(
        validation_set_fraction_size=0.2,
        tolerance_epochs_count=9,
    ),
    optimizer_builder=keras.optimizers.legacy.Adam,
    random_seed=42,
    silent_models_fitting=True,
    create_dataset_eda_reports=True,
)

# ** Train and evaluate model **

OUTPUT_MODEL_FILE_PATH = str(Path(__file__).parent / 'trained_model/model.zip')

train_and_evaluate_model(
    raw_train_dataset_df=train_dataset_df,
    features_config_dict=features_config,
    train_params_dict=train_params,
    output_model_file_path=OUTPUT_MODEL_FILE_PATH,
)
