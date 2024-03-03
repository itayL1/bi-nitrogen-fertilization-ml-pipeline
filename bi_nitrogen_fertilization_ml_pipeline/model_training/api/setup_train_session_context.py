import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import ContextManager

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import TrainPipelineReport, \
    PipelineExecutionTime
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext


@contextmanager
def setup_train_session_context(
    features_config: FeaturesConfig,
    train_params: TrainParams,
) -> ContextManager[TrainSessionContext]:
    session_wip_outputs_folder_path = _create_new_temp_folder()
    try:
        yield TrainSessionContext(
            artifacts=TrainArtifacts(
                features_config=features_config,
            ),
            params=train_params,
            pipeline_report=TrainPipelineReport(
                pipeline_execution_time=PipelineExecutionTime(
                    pipeline_start_timestamp=datetime.now(),
                )
            ),
            temp_wip_outputs_folder_path=session_wip_outputs_folder_path,
        )
    finally:
        # assert False, f'aaaaaaa {session_wip_outputs_folder_path}'
        shutil.rmtree(session_wip_outputs_folder_path)


def _create_new_temp_folder() -> Path:
    return Path(tempfile.mkdtemp())
