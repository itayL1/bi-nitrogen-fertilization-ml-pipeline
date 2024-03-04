from pathlib import Path

import numpy as np
import pandas as pd
from keras import Model

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.preprocessed_datasets import PreprocessedInferenceDataset
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing import dataset_preprocessing
from bi_nitrogen_fertilization_ml_pipeline.core import model_storage


def predict_using_trained_model(
    raw_inference_dataset_df: pd.DataFrame,
    stored_model_file_path: str,
) -> pd.Series:
    trained_model, train_artifacts = model_storage.load_trained_model(Path(stored_model_file_path))

    preprocessed_inference_dataset = dataset_preprocessing.inference_dataset_preprocessing(
        raw_inference_dataset_df, train_artifacts)
    _validate_input_dataset_matches_model_scheme(
        preprocessed_inference_dataset, train_artifacts)

    ordered_predictions_series = _make_predications_for_inference_dataset(
        trained_model, preprocessed_inference_dataset, raw_inference_dataset_df, train_artifacts)
    return ordered_predictions_series


def _make_predications_for_inference_dataset(
    trained_model: Model,
    preprocessed_inference_dataset: PreprocessedInferenceDataset,
    raw_inference_dataset_df: pd.DataFrame,
    train_artifacts: TrainArtifacts,
) -> pd.Series:
    X = preprocessed_inference_dataset.X
    assert X.index.equals(raw_inference_dataset_df.index),\
        "the indices of the input raw dataset and the preprocessed dataset are " \
        "expected to be exactly the same, but they aren't"

    ordered_predictions_array: np.array = trained_model.predict(X, verbose=0)

    target_column_name = train_artifacts.features_config.target_column
    ordered_predictions_series = pd.Series(
        ordered_predictions_array.flatten(), index=X.index, name=target_column_name)
    return ordered_predictions_series


def _validate_input_dataset_matches_model_scheme(
    preprocessed_inference_dataset: PreprocessedInferenceDataset,
    train_artifacts: TrainArtifacts,
) -> None:
    model_input_order_feature_columns = train_artifacts.model_training.model_input_ordered_feature_columns
    inference_dataset_ordered_feature_columns = list(preprocessed_inference_dataset.X.columns)
    if model_input_order_feature_columns != inference_dataset_ordered_feature_columns:
        raise AssertionError(
            f"the preprocessed inference dataset has a different structure then "
            f"the expected structure of the model input. this is not allowed. "
            f"model_input_order_feature_columns: {model_input_order_feature_columns} | "
            f"inference_dataset_ordered_feature_columns: {inference_dataset_ordered_feature_columns} "
        )
