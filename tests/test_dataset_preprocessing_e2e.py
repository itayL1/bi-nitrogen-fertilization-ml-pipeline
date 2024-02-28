import json
from datetime import datetime

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import TrainPipelineReport
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing import dataset_preprocessing
from tests.utils.test_datasets import load_Nitrogen_with_Era5_and_NDVI_dataset, \
    default_Nitrogen_with_Era5_and_NDVI_dataset_features_config
from sklearn.model_selection import train_test_split


def _init_train_session_context(features_config: FeaturesConfig) -> TrainSessionContext:
    return TrainSessionContext(
        artifacts=TrainArtifacts(
            features_config=features_config,
        ),
        pipeline_report=TrainPipelineReport(
            pipeline_start_timestamp=datetime.now(),
        )
    )


def test_dataset_preprocessing_e2e():
    # Arrange
    raw_dataset_df = load_Nitrogen_with_Era5_and_NDVI_dataset()
    train_session_context = _init_train_session_context(
        features_config=default_Nitrogen_with_Era5_and_NDVI_dataset_features_config(),
    )
    raw_train_dataset_df, raw_inference_dataset_df =\
        train_test_split(raw_dataset_df, test_size=0.2, random_state=42)

    # Act
    preprocessed_train_dataset_df = dataset_preprocessing.train_dataset_preprocessing(
        raw_train_dataset_df, train_session_context)
    preprocessed_inference_dataset_df = dataset_preprocessing.inference_dataset_preprocessing(
        raw_inference_dataset_df, train_session_context.artifacts)

    # Assert
    xxx = train_session_context.pipeline_report.copy_without_large_members().json(ensure_ascii=False, indent=4)
    print()
