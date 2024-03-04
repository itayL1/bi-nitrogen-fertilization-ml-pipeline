from pathlib import Path

import keras.optimizers.legacy

from bi_nitrogen_fertilization_ml_pipeline.assets.baseline_model import init_baseline_model
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams, \
    EvaluationFoldsKeySettings, TrainEarlyStoppingSettings
from bi_nitrogen_fertilization_ml_pipeline.model_training.api.train_and_evaluate_model import train_and_evaluate_model
from tests.utils.test_datasets import load_Nitrogen_with_Era5_and_NDVI_dataset, \
    default_Nitrogen_with_Era5_and_NDVI_dataset_features_config

TEMP_OUTPUTS_FOLDER = Path('/Users/itaylotan/git/bi-nitrogen-fertilization-ml-pipeline/tests/temp_outputs/')


def _get_test_train_params() -> TrainParams:
    return TrainParams(
        model_builder=init_baseline_model,
        # epochs_count=100,
        epochs_count=5,
        evaluation_folds_key=EvaluationFoldsKeySettings(
            column='year',
            values_mapper=lambda year_str: str(int(year_str.strip()) % 3),
        ),
        early_stopping=TrainEarlyStoppingSettings(
            validation_set_fraction_size=0.2,
            # tolerance_epochs_count=9,
            tolerance_epochs_count=2,
        ),
        optimizer_builder=keras.optimizers.legacy.Adam,
        random_seed=42,
        silent_models_fitting=True,
    )


def test_train_and_evaluate_model_e2e():
    # Arrange
    raw_train_dataset_df = load_Nitrogen_with_Era5_and_NDVI_dataset()

    # Act
    output_model_file_path = './test_train_and_evaluate_model_e2e/outputs/output_model.zip'
    train_and_evaluate_model(
        raw_train_dataset_df,
        features_config_dict=default_Nitrogen_with_Era5_and_NDVI_dataset_features_config().dict(),
        train_params_dict=_get_test_train_params().dict(),
        output_model_file_path=output_model_file_path,
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
