from enum import Enum

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.base_model import BaseModel


class FeatureKinds(Enum):
    numeric = 'numeric'
    categorical = 'categorical'


class FeatureSettings(BaseModel):
    column: str
    kind: FeatureKinds


class FeaturesConfig(BaseModel):
    target_column: str
    features: list[FeatureSettings]
