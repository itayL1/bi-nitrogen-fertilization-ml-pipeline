from enum import Enum
from typing import Collection, Dict, Any

from pydantic import validator, root_validator

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.base_model import BaseModel


class FeatureKinds(Enum):
    numeric = 'numeric'
    categorical = 'categorical'


class FeatureSettings(BaseModel):
    kind: FeatureKinds


class FeaturesConfig(BaseModel):
    target_column: str
    features: Dict[str, FeatureSettings]

    @validator('features')
    def _collection_must_not_be_empty(cls, features: Dict[str, FeatureSettings]):
        if not isinstance(features, Collection):
            raise ValueError('must be a collection')
        if len(features) == 0:
            raise ValueError('must not be empty')
        return features

    @root_validator()
    def _target_column_must_not_be_a_feature(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        target_column = values['target_column']
        features = values['features']
        if target_column in features.keys():
            raise ValueError('the target column cannot be used as a feature as well')
        return values

    def get_features_and_target_columns(self) -> tuple[str, ...]:
        return (
            self.target_column,
            *(feature_col for feature_col in self.features.keys()),
        )


if __name__ == '__main__':
    FeaturesConfig.parse_obj(dict(
        target_column='a',
        features=dict(
            # a=dict(
            #     kind='numeric',
            # ),
            b=dict(
                kind='categorical',
            ),
            c=dict(
                kind='numeric',
            ),
        ),
    ))
