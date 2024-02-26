from pydantic import Field

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.base_model import BaseModel
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeaturesConfig


class OneHotEncodingMetadata(BaseModel):
    OTHER_CATEGORY = 'other'
    categories_ordered_by_relative_offset = list[str]

    # def get_category_offset(self, category: str) -> int:
    #     if category == self.OTHER_CATEGORY:
    #         return len(self.categories_ordered_by_relative_offset)
    #     else:
    #         try:
    #             return self.categories_ordered_by_relative_offset.index(category)
    #         except ValueError as ex:
    #             raise Exception(f"the category '{category}' is unknown") from ex


OneHotEncodedFeatures = dict[str, OneHotEncodingMetadata]


class DatasetPreprocessingArtifacts(BaseModel):
    one_hot_encoded_features: OneHotEncodedFeatures = Field(default_factory=dict)


class TrainArtifacts(BaseModel):
    features_config: FeaturesConfig
    dataset_preprocessing: DatasetPreprocessingArtifacts = Field(default_factory=DatasetPreprocessingArtifacts)
