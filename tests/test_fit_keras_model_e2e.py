import json
from datetime import datetime
from pathlib import Path

from bi_nitrogen_fertilization_ml_pipeline.assets.baseline_model import init_baseline_model
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams, \
    EvaluationFoldsKeySettings, TrainEarlyStoppingSettings
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import TrainPipelineReport, \
    PipelineExecutionTime
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing import dataset_preprocessing
from bi_nitrogen_fertilization_ml_pipeline.model_training.training.fit_keras_model import fit_keras_model
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
            epochs_count=5,
            evaluation_folds_key=EvaluationFoldsKeySettings(
                column='year',
            ),
            early_stopping=TrainEarlyStoppingSettings(
                validation_set_fraction_size=0.2,
                tolerance_epochs_count=2,
            ),
        ),
        pipeline_report=TrainPipelineReport(
            pipeline_execution_time=PipelineExecutionTime(
                pipeline_start_timestamp=datetime.now(),
            ),
        ),
    )


def test_dataset_preprocessing_e2e():
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
    test_model = init_baseline_model(preprocessed_train_dataset.X.shape[1])
    fit_keras_model(
        test_model,
        preprocessed_train_dataset.X,
        preprocessed_train_dataset.y,
        train_params=train_session_context.params,
        output_figures_folder_path=TEMP_OUTPUTS_FOLDER,
    )

    # # Assert
    # pipeline_execution_time = train_session_context.pipeline_report.pipeline_execution_time
    # pipeline_execution_time.pipeline_end_timestamp = datetime.now()
    # pipeline_execution_time.populate_duration_field()
    #
    # train_pipeline_report_dump = \
    #     train_session_context.pipeline_report.copy_without_large_members().json(ensure_ascii=False, indent=4)
    # with open('train_pipeline_report.json', 'w') as f:
    #     f.write(train_pipeline_report_dump)
