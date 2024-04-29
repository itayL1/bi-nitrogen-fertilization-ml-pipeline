import keras.optimizers.legacy
from sklearn.model_selection import train_test_split

from bi_nitrogen_fertilization_ml_pipeline.assets.baseline_model import init_baseline_model
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams, \
    EvaluationFoldsSplitSettings, TrainEarlyStoppingSettings
from bi_nitrogen_fertilization_ml_pipeline.main_api import train_and_evaluate_model, predict_using_trained_model
from bi_nitrogen_fertilization_ml_pipeline.model_training.utils.train_params_to_keras_api_conversions import \
    eval_func_to_keras_metric
from tests.utils.test_datasets import load_Nitrogen_with_Era5_and_NDVI_dataset, \
    default_Nitrogen_with_Era5_and_NDVI_dataset_features_config


def _get_test_train_params() -> TrainParams:
    return TrainParams(
        model_builder=init_baseline_model,
        epochs_count=100,
        # epochs_count=5,
        evaluation_folds_split=EvaluationFoldsSplitSettings(
            by_key_column='year',
            # by_folds_number=5,
        ),
        early_stopping=TrainEarlyStoppingSettings(
            validation_set_fraction_size=0.2,
            tolerance_epochs_count=9,
            # tolerance_epochs_count=2,
        ),
        optimizer_builder=keras.optimizers.legacy.Adam,
        random_seed=43,
        silent_models_fitting=True,
        create_dataset_eda_reports=True,
    )


def test_train_and_evaluate_model_e2e():
    # Arrange
    features_config = default_Nitrogen_with_Era5_and_NDVI_dataset_features_config()
    train_params = _get_test_train_params()
    raw_dataset_df = load_Nitrogen_with_Era5_and_NDVI_dataset()
    # raw_dataset_df['year'] = raw_dataset_df['year'].astype(str).apply(lambda year_str: str(int(year_str.strip()) % 3))
    raw_dataset_df.dropna(subset=[features_config.target_column], inplace=True)
    raw_train_dataset_df, raw_inference_dataset_df = \
        train_test_split(raw_dataset_df, test_size=0.2, random_state=42)

    # Act
    output_model_file_path = './test_full_e2e/outputs/output_model.zip'
    train_and_evaluate_model(
        raw_train_dataset_df,
        features_config_dict=features_config.dict(),
        train_params_dict=train_params.dict(by_alias=True),
        output_model_file_path=output_model_file_path,
    )

    y_inference_pred = predict_using_trained_model(
        raw_inference_dataset_df, output_model_file_path)
    y_inference_true = raw_inference_dataset_df[features_config.target_column]

    eval_metric = eval_func_to_keras_metric(train_params.evaluation_metric)
    eval_metric.update_state(y_inference_true, y_inference_pred)
    metric_value_on_inference_set = float(eval_metric.result().numpy())
    print(dict(metric_value_on_inference_set=metric_value_on_inference_set))

    # # Assert
    # pipeline_execution_time = train_session_context.pipeline_report.pipeline_execution_time
    # pipeline_execution_time.pipeline_end_timestamp = datetime.now()
    # pipeline_execution_time.populate_duration_field()
    #
    # train_pipeline_report_dump = \
    #     train_session_context.pipeline_report.copy_without_large_members().json(ensure_ascii=False, indent=4)
    # with open('train_pipeline_report.json', 'w') as f:
    #     f.write(train_pipeline_report_dump)
