from datetime import datetime

import pandas as pd
from keras import Model

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.preprocessed_datasets import PreprocessedTrainDataset
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import ModelTrainingArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing import dataset_preprocessing
from bi_nitrogen_fertilization_ml_pipeline.model_training.api.setup_train_session_context import \
    setup_train_session_context
from bi_nitrogen_fertilization_ml_pipeline.model_training.api.user_input import parse_input_features_config, \
    validate_input_train_dataset, parse_input_train_params
from bi_nitrogen_fertilization_ml_pipeline.model_training.evaluation.k_fold_cross_validation import \
    key_based_k_fold_cross_validation
from bi_nitrogen_fertilization_ml_pipeline.model_training.training.model_setup import prepare_new_model_for_training
from bi_nitrogen_fertilization_ml_pipeline.model_training.training.train_model import train_model
from bi_nitrogen_fertilization_ml_pipeline.model_training.utils.random_seed import set_random_seed_globally


def train_and_evaluate_model(
    raw_train_dataset_df: pd.DataFrame,
    features_config_dict: dict,
    train_params_dict: dict,
):
    features_config = parse_input_features_config(features_config_dict)
    train_params = parse_input_train_params(train_params_dict)
    validate_input_train_dataset(raw_train_dataset_df)

    with setup_train_session_context(features_config, train_params) as session_context:
        _run_train_and_evaluation_session(raw_train_dataset_df, session_context)


def _run_train_and_evaluation_session(
    raw_train_dataset_df: pd.DataFrame,
    session_context: TrainSessionContext,
) -> None:
    if session_context.params.random_seed is not None:
        set_random_seed_globally(session_context.params.random_seed)

    preprocessed_train_dataset = dataset_preprocessing.train_dataset_preprocessing(
        raw_train_dataset_df, session_context)
    key_based_k_fold_cross_validation(
        preprocessed_train_dataset, session_context)
    final_model = _train_final_model_on_entire_dataset(
        preprocessed_train_dataset, session_context)

    session_context.artifacts.model_training = ModelTrainingArtifacts(
        model_input_order_feature_columns=list(preprocessed_train_dataset.X.columns),
    )

    _finalize_train_session(session_context)

    # todo - delete
    train_pipeline_report_dump = \
        session_context.pipeline_report.copy_without_large_members().json(ensure_ascii=False, indent=4)
    with open('train_and_evaluate_model.py.5.json', 'w') as f:
        f.write(train_pipeline_report_dump)
    # end of todo - delete

    # todo - add warnings for
    #  * k fold groups not evenly splitted
    #  * too many k fold groups
    #  * feature importance not proportional
    #  * high std in k fold
    #  * final training close to random guess


def _train_final_model_on_entire_dataset(
    preprocessed_train_dataset: PreprocessedTrainDataset,
    session_context: TrainSessionContext,
) -> Model:
    final_model_train_figures_folder =\
        session_context.temp_wip_outputs_folder_path / 'final_model_train_figures'
    final_model_train_figures_folder.mkdir(parents=True, exist_ok=True)
    model_input_features_count = preprocessed_train_dataset.get_train_features_count()

    final_model = prepare_new_model_for_training(
        session_context.params, model_input_features_count)
    train_model(
        final_model,
        X=preprocessed_train_dataset.X,
        y=preprocessed_train_dataset.y,
        train_params=session_context.params,
        output_figures_folder_path=final_model_train_figures_folder,
    )

    model_training = session_context.pipeline_report.model_training
    model_training.final_model_train_figures_folder = final_model_train_figures_folder
    return final_model


def _finalize_train_session(session_context: TrainSessionContext) -> None:
    pipeline_execution_time = session_context.pipeline_report.pipeline_execution_time
    pipeline_execution_time.pipeline_end_timestamp = datetime.now()
    pipeline_execution_time.populate_duration_field()
    session_context.artifacts.is_fitted = True
