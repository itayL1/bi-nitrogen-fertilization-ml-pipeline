from datetime import datetime
from enum import Enum
from typing import Optional

import humanize
import pandas as pd
from pydantic import Field, validator

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.base_model import BaseModel


def _validate_percentage_str(value: str, validate_between_0_and_100: bool) -> None:
    if not value.endswith('%'):
        raise ValueError(f"the percentage string '{value}' must end with '%'")

    actual_percentage_value = float(value.rstrip('%'))
    if validate_between_0_and_100:
        if not 0 <= actual_percentage_value <= 100:
            raise ValueError(f"the percentage string '{value}' must be between 0 and 100")


def _validate_percentage_distribution_dict(percentage_distribution: dict[str, str]) -> None:
    for perc_val in percentage_distribution.values():
        _validate_percentage_str(perc_val, validate_between_0_and_100=True)


class PipelineExecutionTime(BaseModel):
    duration: Optional[str]
    pipeline_start_timestamp: Optional[datetime]
    pipeline_end_timestamp: Optional[datetime]

    def populate_duration_field(self) -> None:
        assert self.pipeline_start_timestamp is not None
        assert self.pipeline_end_timestamp is not None
        assert self.duration is None, 'the duration field is already populated'
        self.duration = humanize.precisedelta(
            self.pipeline_start_timestamp - self.pipeline_end_timestamp
        )


class ImputationFunnel(BaseModel):
    remaining_rows_percentage: str
    rows_count_before_imputation: int
    rows_count_after_imputation: int

    @validator('remaining_rows_percentage')
    def _total_percentage_validator(cls, remaining_rows_percentage: str) -> str:
        _validate_percentage_str(remaining_rows_percentage, validate_between_0_and_100=True)
        return remaining_rows_percentage


class CategoricalFeaturesEncodingMethod(str, Enum):
    one_hot_encoding = 'one_hot_encoding'


class OtherCategoryAggregationDetails(BaseModel):
    total_percentage: str
    min_significant_category: str
    aggregated_categories_distribution: dict[str, str]

    @validator('total_percentage')
    def _total_percentage_validator(cls, total_percentage: str) -> str:
        _validate_percentage_str(total_percentage, validate_between_0_and_100=True)
        return total_percentage

    @validator('aggregated_categories_distribution')
    def _aggregated_categories_distribution(
        cls, aggregated_categories_distribution: dict[str, str],
    ) -> dict[str, str]:
        _validate_percentage_distribution_dict(aggregated_categories_distribution)
        return aggregated_categories_distribution


class FinalCategories(BaseModel):
    count: int
    values: list[str]


class CategoricalFeatureEncodingDetails(BaseModel):
    encoding_method: CategoricalFeaturesEncodingMethod
    final_categories: FinalCategories
    categories_distribution: dict[str, str]
    other_category_aggregation: OtherCategoryAggregationDetails

    @validator('categories_distribution')
    def _aggregated_categories_distribution(
        cls, categories_distribution: dict[str, str],
    ) -> dict[str, str]:
        _validate_percentage_distribution_dict(categories_distribution)
        return categories_distribution


CategoricalFeaturesEncodingDetails = dict[str, CategoricalFeatureEncodingDetails]


class DatasetPreprocessing(BaseModel):
    original_dataset: Optional[pd.DataFrame]
    preprocessed_dataset: Optional[pd.DataFrame]
    imputation_funnel: Optional[ImputationFunnel]
    unused_dropped_columns_count: Optional[int]
    categorical_features_encoding_details: Optional[CategoricalFeaturesEncodingDetails] = Field(default_factory=dict)

    def copy_without_large_members(self):
        return self.copy(deep=True, exclude={'original_dataset', 'preprocessed_dataset'})


class PipelineModules(str, Enum):
    dataset_preprocessing = 'dataset_preprocessing'
    model_training = 'model_training'


class ReportWarning(BaseModel):
    pipeline_module: PipelineModules
    description: str
    context: Optional[dict]


class TrainPipelineReport(BaseModel):
    pipeline_execution_time: PipelineExecutionTime = Field(default_factory=PipelineExecutionTime)
    dataset_preprocessing: DatasetPreprocessing = Field(default_factory=DatasetPreprocessing)
    warnings: list[ReportWarning] = Field(default_factory=list)

    def copy_without_large_members(self):
        ret_copy = self.copy(deep=True, exclude={'dataset_preprocessing'})
        ret_copy.dataset_preprocessing = self.dataset_preprocessing.copy_without_large_members()
        return ret_copy

    def add_warning(
        self, pipeline_module: PipelineModules,
        description: str, context: Optional[dict] = None,
    ) -> None:
        self.warnings.append(ReportWarning(
            pipeline_module=pipeline_module,
            description=description,
            context=context,
        ))

