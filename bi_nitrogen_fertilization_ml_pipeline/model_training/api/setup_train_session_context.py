import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import ContextManager

from rich.progress import Progress

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import TrainPipelineReportData, \
    PipelineExecutionTime
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.model_training.api.train_pipeline_progress_display import \
    TrainPipelineMainProgressBarManager
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_logical_steps import \
    TrainPipelineLogicalSteps


@contextmanager
def train_session_context(
    features_config: FeaturesConfig,
    train_params: TrainParams,
    first_pipeline_step: TrainPipelineLogicalSteps,
) -> ContextManager[TrainSessionContext]:
    with (
        tempfile.TemporaryDirectory() as session_wip_outputs_folder_path,
        Progress() as rich_progress,
    ):
        pipeline_main_progress_bar = TrainPipelineMainProgressBarManager(
            rich_progress=rich_progress,
            first_pipeline_step=first_pipeline_step,
        )
        yield TrainSessionContext(
            artifacts=TrainArtifacts(
                features_config=features_config,
            ),
            params=train_params,
            pipeline_report=TrainPipelineReportData(
                pipeline_execution_time=PipelineExecutionTime(
                    pipeline_start_timestamp=datetime.now(),
                )
            ),
            wip_outputs_folder_path=Path(session_wip_outputs_folder_path),
            rich_progress=rich_progress,
            pipeline_main_progress_bar=pipeline_main_progress_bar
        )
