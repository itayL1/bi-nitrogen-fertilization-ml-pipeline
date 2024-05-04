# ** Imports **
from pathlib import Path

import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.main_api import predict_using_trained_model

# Load inference dataset

INFERENCE_DATASET_FILE_PATH = Path(__file__).parent / 'datasets/sample_dataset.csv'

column_dtypes = {
    'משק': str,
    'עומק': float,
    'קוד  קרקע': str,
    'Krab_key_eng': str,
    'YLD_Israel(1000T)': float,
    'Precip_Gilat': float,
    'year': str,
}
inference_dataset_df = pd.read_csv(
    INFERENCE_DATASET_FILE_PATH, low_memory=False, dtype=column_dtypes,
)

# ** Get model predictions **

TRAINED_MODEL_FILE_PATH = str(Path(__file__).parent / 'trained_model/model.zip')

y_pred = predict_using_trained_model(inference_dataset_df, TRAINED_MODEL_FILE_PATH)

# ** Optional - Save the inference dataset with the model predictions to a csv file **

OUTPUT_CSV_FILE_PATH = '/tmp/inference_dataset_with_predictions.csv'

inference_dataset_with_predictions_df = inference_dataset_df.copy()
inference_dataset_with_predictions_df['model_prediction'] = y_pred
inference_dataset_with_predictions_df.to_csv(OUTPUT_CSV_FILE_PATH, index=False)
