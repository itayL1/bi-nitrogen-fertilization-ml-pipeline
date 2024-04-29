import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from bi_nitrogen_fertilization_ml_pipeline.assets.baseline_model import init_baseline_model
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams, \
    EvaluationFoldsSplitSettings, TrainEarlyStoppingSettings
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import TrainPipelineReportData, \
    PipelineExecutionTime
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing import dataset_preprocessing
from bi_nitrogen_fertilization_ml_pipeline.model_training.training.model_setup import prepare_new_model_for_training
from bi_nitrogen_fertilization_ml_pipeline.model_training.training.train_model import train_model
from tests.utils.test_datasets import load_Nitrogen_with_Era5_and_NDVI_dataset, \
    default_Nitrogen_with_Era5_and_NDVI_dataset_features_config
from sklearn.model_selection import train_test_split

TEMP_OUTPUTS_FOLDER = Path('/Users/itaylotan/git/bi-nitrogen-fertilization-ml-pipeline/tests/temp_outputs/')


def _init_train_session_context(features_config: FeaturesConfig) -> TrainSessionContext:
    return TrainSessionContext(
        artifacts=TrainArtifacts(
            features_config=features_config,
        ),
        params=TrainParams(
            model_builder=init_baseline_model,
            epochs_count=10,
            evaluation_folds_split=EvaluationFoldsSplitSettings(
                by_key_column='year',
            ),
            early_stopping=TrainEarlyStoppingSettings(
                validation_set_fraction_size=0.2,
                tolerance_epochs_count=9,
            ),
        ),
        pipeline_report=TrainPipelineReportData(
            pipeline_execution_time=PipelineExecutionTime(
                pipeline_start_timestamp=datetime.now(),
            ),
        ),
        wip_outputs_folder_path=Path(tempfile.mkdtemp()),
    )


def test_train_keras_model_e2e():
    # Arrange
    raw_dataset_df = load_Nitrogen_with_Era5_and_NDVI_dataset()

    train_session_context = _init_train_session_context(
        features_config=default_Nitrogen_with_Era5_and_NDVI_dataset_features_config(),
    )
    raw_train_dataset_df, raw_inference_dataset_df = \
        train_test_split(raw_dataset_df, test_size=0.2, random_state=42)
    preprocessed_train_dataset = dataset_preprocessing.train_dataset_preprocessing(
        raw_train_dataset_df, train_session_context)

    # Act
    try:
        test_model = prepare_new_model_for_training(
            train_params=train_session_context.params,
            input_features_count=preprocessed_train_dataset.get_train_features_count(),
        )
        train_model(
            test_model,
            preprocessed_train_dataset.X,
            preprocessed_train_dataset.y,
            train_params=train_session_context.params,
            output_figures_folder_path=TEMP_OUTPUTS_FOLDER,
        )
    finally:
        shutil.rmtree(train_session_context.wip_outputs_folder_path)

    # # Assert
    # pipeline_execution_time = train_session_context.pipeline_report.pipeline_execution_time
    # pipeline_execution_time.pipeline_end_timestamp = datetime.now()
    # pipeline_execution_time.populate_duration_field()
    #
    # train_pipeline_report_dump = \
    #     train_session_context.pipeline_report.copy_without_large_members().json(ensure_ascii=False, indent=4)
    # with open('train_pipeline_report.json', 'w') as f:
    #     f.write(train_pipeline_report_dump)
