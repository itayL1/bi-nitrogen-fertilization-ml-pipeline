from pathlib import Path
from uuid import uuid4

from pydantic import Field

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.base_model import BaseModel
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import TrainPipelineReportData


class TrainSessionContext(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid4().hex)
    params: TrainParams
    artifacts: TrainArtifacts
    pipeline_report: TrainPipelineReportData = Field(default_factory=TrainPipelineReportData)
    wip_outputs_folder_path: Path

    def get_raw_dataset_columns_required_for_training(self) -> tuple[str, ...]:
        return (
            self.artifacts.features_config.target_column,
            self.params.evaluation_folds_key.column,
            *self.artifacts.features_config.get_feature_columns(),
        )
