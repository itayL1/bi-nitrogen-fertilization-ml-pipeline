from typing import Optional

from pydantic import Field

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.base_model import BaseModel
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig


class OneHotEncodingMetadata(BaseModel):
    original_raw_categories: list[str]
    categories_ordered_by_relative_offset: list[str]
    categories_mapped_to_other: list[str]
    allow_unknown_categories_during_inference: bool

    def get_known_categories(self) -> list[str]:
        return self.categories_ordered_by_relative_offset + self.categories_mapped_to_other


OneHotEncodedFeatures = dict[str, OneHotEncodingMetadata]


class DatasetPreprocessingArtifacts(BaseModel):
    one_hot_encoded_features: OneHotEncodedFeatures = Field(default_factory=dict)


class ModelTrainingArtifacts(BaseModel):
    model_input_order_feature_columns: list[str]


class TrainArtifacts(BaseModel):
    features_config: FeaturesConfig
    dataset_preprocessing: DatasetPreprocessingArtifacts = Field(default_factory=DatasetPreprocessingArtifacts)
    model_training: Optional[ModelTrainingArtifacts]
    is_fitted: bool = Field(default=False)
