from datetime import datetime

import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import TrainPipelineReport, \
    PipelineExecutionTime
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.model_training.api.user_input import parse_input_features_config, \
    validate_input_train_dataset, parse_input_train_params


def train_and_evaluate_model(
    raw_train_dataset_df: pd.DataFrame,
    features_config_dict: dict,
    train_params_dict: dict,
):
    features_config = parse_input_features_config(features_config_dict)
    train_params = parse_input_train_params(train_params_dict)
    validate_input_train_dataset(raw_train_dataset_df)

    session_context = _init_train_session_context(features_config, train_params)


def _init_train_session_context(
    features_config: FeaturesConfig,
    train_params: TrainParams,
) -> TrainSessionContext:
    return TrainSessionContext(
        artifacts=TrainArtifacts(
            features_config=features_config,
        ),
        params=train_params,
        pipeline_report=TrainPipelineReport(
            pipeline_execution_time=PipelineExecutionTime(
                pipeline_start_timestamp=datetime.now(),
            )
        ),
    )
