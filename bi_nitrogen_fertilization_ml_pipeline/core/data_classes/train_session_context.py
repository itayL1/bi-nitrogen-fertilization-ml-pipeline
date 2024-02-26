from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.base_model import BaseModel
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import TrainPipelineReport


class TrainSessionContext(BaseModel):
    session_id: str
    artifacts: TrainArtifacts
    pipeline_report: TrainPipelineReport
