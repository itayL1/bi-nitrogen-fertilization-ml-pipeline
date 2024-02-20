import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.model_training.user_input import parse_input_features_config


def train_and_evaluate_model(train_dataset_df: pd.DataFrame, features_config: dict):
    parsed_features_config = parse_input_features_config(features_config)
