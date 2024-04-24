from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import Field, validator

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.base_model import BaseModel
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.field_utils import validate_percentage_str, \
    validate_percentage_distribution_dict, validate_dataframe_has_2_dimensions
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.k_fold_cross_validation import \
    KFoldCrossValidationResults


class PipelineExecutionTime(BaseModel):
    pipeline_start_timestamp: Optional[datetime]
    pipeline_end_timestamp: Optional[datetime]


class ImputationFunnel(BaseModel):
    remaining_rows_percentage: str
    rows_count_before_imputation: int
    rows_count_after_imputation: int

    @validator('remaining_rows_percentage')
    def _total_percentage_validator(cls, remaining_rows_percentage: str) -> str:
        validate_percentage_str(remaining_rows_percentage, validate_between_0_and_100=True)
        return remaining_rows_percentage


class CategoricalFeaturesEncodingMethod(str, Enum):
    one_hot_encoding = 'one_hot_encoding'


class OtherCategoryAggregationDetails(BaseModel):
    total_percentage: str
    min_significant_category_percentage_threshold: str
    aggregated_categories_distribution: dict[str, str]

    @validator('total_percentage')
    def _total_percentage_validator(cls, total_percentage: str) -> str:
        validate_percentage_str(total_percentage, validate_between_0_and_100=True)
        return total_percentage

    @validator('aggregated_categories_distribution')
    def _aggregated_categories_distribution(
        cls, aggregated_categories_distribution: dict[str, str],
    ) -> dict[str, str]:
        validate_percentage_distribution_dict(aggregated_categories_distribution)
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
        validate_percentage_distribution_dict(categories_distribution)
        return categories_distribution


CategoricalFeaturesEncodingDetails = dict[str, CategoricalFeatureEncodingDetails]


class DatasetPreprocessing(BaseModel):
    original_input_dataset: Optional[pd.DataFrame]
    preprocessed_input_dataset: Optional[pd.DataFrame]
    raw_dataset_columns_required_for_training: Optional[tuple[str, ...]]
    imputation_funnel: Optional[ImputationFunnel]
    unused_dropped_columns_count: Optional[int]
    categorical_features_encoding_details: Optional[CategoricalFeaturesEncodingDetails] = Field(default_factory=dict)

    def copy_without_large_members(self):
        return self.copy(deep=True, exclude={'original_input_dataset', 'preprocessed_input_dataset'})

    @validator('original_input_dataset')
    def _validate_original_dataset_has_2_dimensions(
        cls, original_dataset: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        validate_dataframe_has_2_dimensions(original_dataset)
        return original_dataset

    @validator('preprocessed_input_dataset')
    def _validate_preprocessed_dataset_has_2_dimensions(
        cls, preprocessed_dataset: Optional[pd.DataFrame],
    ) -> Optional[pd.DataFrame]:
        validate_dataframe_has_2_dimensions(preprocessed_dataset)
        return preprocessed_dataset


class ModelTraining(BaseModel):
    evaluation_folds_results: Optional[KFoldCrossValidationResults]
    evaluation_folds_distribution_gini_coefficient: Optional[float]
    evaluation_folds_train_figures_root_folder: Optional[Path]
    final_model_train_figures_folder: Optional[Path]


class PipelineModules(str, Enum):
    dataset_preprocessing = 'dataset_preprocessing'
    model_training = 'model_training'


class ReportWarning(BaseModel):
    pipeline_module: PipelineModules
    description: str
    context: Optional[dict]


class TrainPipelineReportData(BaseModel):
    pipeline_execution_time: PipelineExecutionTime = Field(default_factory=PipelineExecutionTime)
    dataset_preprocessing: DatasetPreprocessing = Field(default_factory=DatasetPreprocessing)
    model_training: ModelTraining = Field(default_factory=ModelTraining)
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
