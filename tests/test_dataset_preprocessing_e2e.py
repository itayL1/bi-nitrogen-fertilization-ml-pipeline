from datetime import datetime

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import TrainPipelineReport
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing import dataset_preprocessing
from tests.utils.test_datasets import load_Nitrogen_with_Era5_and_NDVI_dataset, \
    default_Nitrogen_with_Era5_and_NDVI_dataset_features_config


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
    raw_train_dataset_df = load_Nitrogen_with_Era5_and_NDVI_dataset()
    train_session_context = _init_train_session_context(
        features_config=default_Nitrogen_with_Era5_and_NDVI_dataset_features_config(),
    )

    # Act
    preprocessed_inference_dataset_df = dataset_preprocessing.train_dataset_preprocessing(
        raw_train_dataset_df, train_session_context)

    # Assert
    print(preprocessed_inference_dataset_df.shape)