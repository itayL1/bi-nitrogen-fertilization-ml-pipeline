from pathlib import Path
from uuid import uuid4

from pydantic import Field
from rich.progress import Progress

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.base_model import BaseModel
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import TrainPipelineReportData
from bi_nitrogen_fertilization_ml_pipeline.core.display_utils.train_pipeline_progress_display import \
    TrainPipelineMainProgressBarManager


class TrainSessionContext(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid4().hex)
    params: TrainParams
    artifacts: TrainArtifacts
    pipeline_report: TrainPipelineReportData = Field(default_factory=TrainPipelineReportData)
    wip_outputs_folder_path: Path
    rich_progress: Progress
    pipeline_main_progress_bar: TrainPipelineMainProgressBarManager

    def get_raw_dataset_columns_required_for_training(self) -> tuple[str, ...]:
        evaluation_folds_split = self.params.evaluation_folds_split

        return (
            self.artifacts.features_config.target_column,
            *(
                [evaluation_folds_split.key_column] if evaluation_folds_split.key_column is not None else []
            ),
            *self.artifacts.features_config.get_feature_columns(),
        )
