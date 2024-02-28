from datetime import datetime

import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import TrainPipelineReport, \
    PipelineExecutionTime
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.model_training.user_input import parse_input_features_config, \
    validate_input_train_dataset


def train_and_evaluate_model(raw_train_dataset_df: pd.DataFrame, features_config_dict: dict):
    features_config = parse_input_features_config(features_config_dict)
    validate_input_train_dataset(raw_train_dataset_df, features_config)

    session_context = _init_train_session_context(features_config)


def _init_train_session_context(features_config: FeaturesConfig) -> TrainSessionContext:
    return TrainSessionContext(
        artifacts=TrainArtifacts(
            features_config=features_config,
        ),
        pipeline_report=TrainPipelineReport(
            pipeline_execution_time=PipelineExecutionTime(
                pipeline_start_timestamp=datetime.now(),
            )
        )
    )
