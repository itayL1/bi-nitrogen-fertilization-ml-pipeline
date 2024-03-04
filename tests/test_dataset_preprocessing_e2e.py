import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from bi_nitrogen_fertilization_ml_pipeline.assets.baseline_model import init_baseline_model
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams, EvaluationFoldsKeySettings
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import TrainPipelineReportData, \
    PipelineExecutionTime
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
        params=TrainParams(
            model_builder=init_baseline_model,
            epochs_count=3,
            evaluation_folds_key=EvaluationFoldsKeySettings(
                column='year',
            ),
        ),
        pipeline_report=TrainPipelineReportData(
            pipeline_execution_time=PipelineExecutionTime(
                pipeline_start_timestamp=datetime.now(),
            ),
        ),
        wip_outputs_folder_path=Path(tempfile.mkdtemp()),
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
    try:
        preprocessed_train_dataset = dataset_preprocessing.train_dataset_preprocessing(
            raw_train_dataset_df, train_session_context)
        preprocessed_inference_dataset = dataset_preprocessing.inference_dataset_preprocessing(
            raw_inference_dataset_df, train_session_context.artifacts)
    finally:
        shutil.rmtree(train_session_context.wip_outputs_folder_path)

    # Assert
    assert 'y' not in preprocessed_train_dataset.X.columns
    assert list(preprocessed_train_dataset.X.columns) == list(preprocessed_inference_dataset.X.columns)

    pipeline_execution_time = train_session_context.pipeline_report.pipeline_execution_time
    pipeline_execution_time.pipeline_end_timestamp = datetime.now()
    pipeline_execution_time.populate_duration_field()

    train_pipeline_report_dump =\
        train_session_context.pipeline_report.copy_without_large_members().json(ensure_ascii=False, indent=4)
    with open('train_pipeline_report.json', 'w') as f:
        f.write(train_pipeline_report_dump)

    # train_session_context.pipeline_report.dataset_preprocessing.original_dataset.head(20).to_html('train_report.dataset_preprocessing.original_dataset.html')
    # train_session_context.pipeline_report.dataset_preprocessing.preprocessed_dataset.head(20).to_html('train_report.dataset_preprocessing.preprocessed_dataset.html')
    # preprocessed_train_dataset.get_full_dataset().head(20).to_html('preprocessed_train_dataset.html')
    # preprocessed_inference_dataset.X.head(20).to_html('preprocessed_inference_dataset.html')
