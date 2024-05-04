from typing import Optional, Callable, Any

from pydantic import validator, confloat, PositiveInt, Field, root_validator
import keras
from keras.optimizers import Optimizer

from bi_nitrogen_fertilization_ml_pipeline.core import defaults
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.base_model import BaseModel
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.evaluation_functions import EvaluationFunctions


class TrainEarlyStoppingSettings(BaseModel):
    validation_set_fraction_size: confloat(ge=0.0, lt=1.0)
    tolerance_epochs_count: PositiveInt


class EvaluationFoldsSplitSettings(BaseModel):
    key_column: Optional[str] = Field(alias='by_key_column')
    folds_number: Optional[int] = Field(alias='by_folds_number')

    @root_validator()
    def _exactly_one_option_is_used(cls, values: dict[str, Any]) -> dict[str, Any]:
        if values['key_column'] is None and values['folds_number'] is None:
            raise ValueError('exactly one of the properties must be set (by_key_column or by_folds_number)')
        if values['key_column'] is not None and values['folds_number'] is not None:
            raise ValueError('only one of the properties must be set, but not both (by_key_column or by_folds_number)')
        return values


class TrainParams(BaseModel):
    model_builder: Callable[[int], keras.Model]
    loss_function: EvaluationFunctions = Field(default=defaults.LOSS_FUNCTION)
    evaluation_metric: EvaluationFunctions = Field(default=defaults.EVALUATION_METRIC)
    optimizer_builder: Callable[[], Optimizer]
    epochs_count: int
    early_stopping: Optional[TrainEarlyStoppingSettings]
    evaluation_folds_split: EvaluationFoldsSplitSettings
    random_seed: Optional[int]
    silent_models_fitting: Optional[bool] = Field(default=True)
    create_dataset_eda_reports: Optional[bool] = Field(default=True)
