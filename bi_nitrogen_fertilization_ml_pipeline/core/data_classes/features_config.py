from enum import Enum
from typing import Any, Optional

from pydantic import validator, root_validator, confloat, PositiveInt

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.base_model import BaseModel
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.field_utils import not_empty_str
from bi_nitrogen_fertilization_ml_pipeline.core.utils.collection_utils import find_duplicates


class FeatureKinds(Enum):
    numeric = 'numeric'
    categorical = 'categorical'


class OneHotEncodingSettings(BaseModel):
    min_significant_category_percentage_threshold: Optional[confloat(gt=0, lt=100)]
    max_allowed_categories_count: Optional[PositiveInt]
    allow_unknown_categories_during_inference: Optional[bool]


class FeatureSettings(BaseModel):
    column: not_empty_str
    kind: FeatureKinds
    one_hot_encoding_settings: Optional[OneHotEncodingSettings]

    @root_validator()
    def _properties_combination_is_valid(cls, values: dict[str, Any]) -> dict[str, Any]:
        kind = values.get('kind', None)
        one_hot_encoding_settings = values.get('one_hot_encoding_settings', None)
        if any(prop is None for prop in (kind, one_hot_encoding_settings)):
            return values

        if one_hot_encoding_settings is not None and kind != FeatureKinds.categorical:
            raise ValueError('one_hot_encoding_settings can only be used for categorical features')
        return values


class FeaturesConfig(BaseModel):
    target_column: not_empty_str
    features: list[FeatureSettings]

    @validator('features')
    def _collection_must_not_be_empty(cls, features: list[FeatureSettings]):
        if len(features) == 0:
            raise ValueError('must not be empty')

        features_with_duplicate_definitions =\
            find_duplicates([feature.column for feature in features])
        if any(features_with_duplicate_definitions):
            raise ValueError(
                f'features cannot have multiple definitions, but these '
                f'features do - {features_with_duplicate_definitions}'
            )

        return features

    @root_validator()
    def _target_column_must_not_be_a_feature(cls, values: dict[str, Any]) -> dict[str, Any]:
        target_column = values.get('target_column', None)
        features = values.get('features', None)
        if any(prop is None for prop in (target_column, features)):
            return values

        feature_columns = [feature.column for feature in features]
        if target_column in feature_columns:
            raise ValueError('the target column cannot be used as a feature as well')
        return values

    def get_feature_columns(self) -> tuple[str, ...]:
        return tuple(feature.column for feature in self.features)
