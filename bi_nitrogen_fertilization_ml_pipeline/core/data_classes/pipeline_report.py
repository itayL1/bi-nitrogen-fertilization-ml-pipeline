from enum import Enum
from typing import Optional

import pandas as pd
from pydantic import Field, validator

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.base_model import BaseModel


def _validate_percentage_str(value: str) -> None:
    if not value.endswith('%'):
        raise ValueError(f"the percentage string '{value}' must end with '%'")


def _validate_percentage_distribution_dict(percentage_distribution: dict[str, str]) -> None:
    for perc_val in percentage_distribution.values():
        _validate_percentage_str(perc_val)


class ImputationFunnel(BaseModel):
    rows_count_before_imputation: int
    rows_count_after_imputation: int


class CategoricalFeaturesEncodingMethod(str, Enum):
    one_hot_encoding = 'one_hot_encoding'


class OtherCategoryAggregationDetails(BaseModel):
    total_percentage: str
    aggregated_categories_threshold: str
    aggregated_categories_distribution: dict[str, str]

    @validator('total_percentage')
    def _total_percentage_validator(cls, total_percentage: str) -> str:
        _validate_percentage_str(total_percentage)
        return total_percentage

    @validator('aggregated_categories_distribution')
    def _aggregated_categories_distribution(
        cls, aggregated_categories_distribution: dict[str, str],
    ) -> dict[str, str]:
        _validate_percentage_distribution_dict(aggregated_categories_distribution)
        return aggregated_categories_distribution


class CategoricalFeatureEncodingDetails(BaseModel):
    encoding_method: CategoricalFeaturesEncodingMethod
    categories_distribution: dict[str, str]
    other_category_aggregation: OtherCategoryAggregationDetails

    @validator('categories_distribution')
    def _aggregated_categories_distribution(
        cls, categories_distribution: dict[str, str],
    ) -> dict[str, str]:
        _validate_percentage_distribution_dict(categories_distribution)
        return categories_distribution


class DatasetPreprocessing(BaseModel):
    original_dataset: Optional[pd.DataFrame]
    preprocessed_dataset: Optional[pd.DataFrame]
    imputation_funnel: Optional[ImputationFunnel]
    categorical_features_encoding_details = Optional[dict[str, CategoricalFeatureEncodingDetails]]


class ReportWarning(BaseModel):
    title: str
    description: str


class PipelineReport(BaseModel):
    dataset_preprocessing: DatasetPreprocessing = Field(default_factory=DatasetPreprocessing)
    warnings: list[ReportWarning] = Field(default_factory=list)
