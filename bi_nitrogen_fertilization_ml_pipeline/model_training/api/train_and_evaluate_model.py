import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from keras import Model

from bi_nitrogen_fertilization_ml_pipeline.core import model_storage
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.preprocessed_datasets import PreprocessedTrainDataset
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import ModelTrainingArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_logical_steps import \
    TrainPipelineLogicalSteps
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing import dataset_preprocessing
from bi_nitrogen_fertilization_ml_pipeline.model_training.api.setup_train_session_context import train_session_context
from bi_nitrogen_fertilization_ml_pipeline.model_training.api.user_input import parse_input_features_config, \
    validate_input_train_dataset, parse_input_train_params
from bi_nitrogen_fertilization_ml_pipeline.model_training.evaluation.k_fold_cross_validation import \
    key_based_k_fold_cross_validation
from bi_nitrogen_fertilization_ml_pipeline.model_training.train_pipeline_report import \
    create_and_save_train_pipeline_report
from bi_nitrogen_fertilization_ml_pipeline.model_training.training.model_setup import prepare_new_model_for_training
from bi_nitrogen_fertilization_ml_pipeline.model_training.training.train_model import train_model
from bi_nitrogen_fertilization_ml_pipeline.model_training.utils.random_seed import set_random_seed_globally


def train_and_evaluate_model(
    raw_train_dataset_df: pd.DataFrame,
    features_config_dict: dict,
    train_params_dict: dict,
    output_model_file_path: str,
) -> None:
    features_config = parse_input_features_config(features_config_dict)
    train_params = parse_input_train_params(train_params_dict)

    output_model_file_path = Path(output_model_file_path)
    model_storage.validate_storage_file_name(output_model_file_path)
    validate_input_train_dataset(raw_train_dataset_df)

    with train_session_context(
        features_config, train_params,
        first_pipeline_step=TrainPipelineLogicalSteps.preprocess_train_dataset,
    ) as session_context:
        trained_model = _run_train_and_evaluation_session(raw_train_dataset_df, session_context)

        wip_output_model_file = session_context.wip_outputs_folder_path / 'output_model.zip'
        model_storage.store_trained_model(trained_model, session_context.artifacts, wip_output_model_file)

        session_context.pipeline_report.pipeline_execution_time.pipeline_end_timestamp = datetime.now()

        session_context.pipeline_main_progress_bar.move_to_next_step(
            TrainPipelineLogicalSteps.generate_pipeline_report)
        wip_train_pipeline_report_folder = session_context.wip_outputs_folder_path / 'train_pipeline_report'
        create_and_save_train_pipeline_report(
            session_context.pipeline_report, wip_train_pipeline_report_folder)

        _move_wip_files_to_output_paths(
            output_model_file_path, wip_output_model_file, wip_train_pipeline_report_folder)

        # todo - add warnings for
        #  * k fold groups not evenly splitted
        #  * too many k fold groups
        #  * feature importance not proportional
        #  * high std in k fold
        #  * final training close to random guess


def _run_train_and_evaluation_session(
    raw_train_dataset_df: pd.DataFrame,
    session_context: TrainSessionContext,
) -> Model:
    if session_context.params.random_seed is not None:
        set_random_seed_globally(session_context.params.random_seed)
    main_progress_bar = session_context.pipeline_main_progress_bar

    preprocessed_train_dataset = dataset_preprocessing.train_dataset_preprocessing(
        raw_train_dataset_df, session_context)
    main_progress_bar.move_to_next_step(TrainPipelineLogicalSteps.model_k_fold_cross_valuation)
    key_based_k_fold_cross_validation(
        preprocessed_train_dataset, session_context)
    main_progress_bar.move_to_next_step(TrainPipelineLogicalSteps.final_model_training)
    final_model = _train_final_model_on_entire_dataset(
        preprocessed_train_dataset, session_context)

    session_context.artifacts.model_training = ModelTrainingArtifacts(
        model_input_ordered_feature_columns=list(preprocessed_train_dataset.X.columns),
    )

    session_context.artifacts.is_fitted = True
    return final_model


def _train_final_model_on_entire_dataset(
    preprocessed_train_dataset: PreprocessedTrainDataset,
    session_context: TrainSessionContext,
) -> Model:
    final_model_train_figures_folder = \
        session_context.wip_outputs_folder_path / 'final_model_train_figures'
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


def _move_wip_files_to_output_paths(
    output_model_file_path: Path,
    wip_output_model_file: Path,
    wip_train_pipeline_report_folder: Path,
) -> None:
    main_output_folder = output_model_file_path.parent
    output_train_report_root_folder = main_output_folder / 'train_pipeline_report'

    main_output_folder.mkdir(parents=True, exist_ok=True)
    output_model_file_path.unlink(missing_ok=True)
    if output_train_report_root_folder.is_dir():
        shutil.rmtree(output_train_report_root_folder)

    wip_output_model_file.rename(output_model_file_path)
    wip_train_pipeline_report_folder.rename(output_train_report_root_folder)
