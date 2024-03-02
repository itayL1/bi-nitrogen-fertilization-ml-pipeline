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
from bi_nitrogen_fertilization_ml_pipeline.model_training.api.train_and_evaluate_model import train_and_evaluate_model
from bi_nitrogen_fertilization_ml_pipeline.model_training.training.train_model import train_model
from tests.utils.test_datasets import load_Nitrogen_with_Era5_and_NDVI_dataset, \
    default_Nitrogen_with_Era5_and_NDVI_dataset_features_config
from sklearn.model_selection import train_test_split

TEMP_OUTPUTS_FOLDER = Path('/Users/itaylotan/git/bi-nitrogen-fertilization-ml-pipeline/tests/temp_outputs/')


def _get_test_train_params() -> TrainParams:
    return TrainParams(
        model_builder=init_baseline_model,
        epochs_count=5,
        evaluation_folds_key=EvaluationFoldsKeySettings(
            column='year',
        ),
        early_stopping=TrainEarlyStoppingSettings(
            validation_set_fraction_size=0.2,
            tolerance_epochs_count=2,
        ),
    )


def test_train_and_evaluate_model_e2e():
    # Arrange
    raw_train_dataset_df = load_Nitrogen_with_Era5_and_NDVI_dataset()

    # Act
    train_and_evaluate_model(
        raw_train_dataset_df,
        features_config_dict=default_Nitrogen_with_Era5_and_NDVI_dataset_features_config().dict(),
        train_params_dict=_get_test_train_params().dict()
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
