from typing import Optional, Callable

from pydantic import validator, confloat, PositiveInt, Field
from keras.optimizers import Optimizer

from bi_nitrogen_fertilization_ml_pipeline.core import defaults
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.base_model import BaseModel
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.evaluation_functions import EvaluationFunctions
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.field_utils import not_empty_str


class TrainEarlyStoppingSettings(BaseModel):
    validation_set_fraction_size: confloat(ge=0.0, lt=1.0)
    tolerance_epochs_count: PositiveInt


class EvaluationFoldsKeySettings(BaseModel):
    column: not_empty_str
    values_mapper: Optional[Callable[[str], str]]

    @validator('values_mapper')
    def _values_mapper_validator(
        cls, values_mapper: Optional[Callable[[str], str]],
    ) -> Optional[Callable[[str], str]]:
        if values_mapper is not None:
            if values_mapper.__code__.co_argcount != 1:
                raise ValueError("values_mapper must take exactly one argument.")
        return values_mapper


class TrainParams(BaseModel):
    loss_function: EvaluationFunctions = Field(default=defaults.LOSS_FUNCTION)
    evaluation_metric: EvaluationFunctions = Field(default=defaults.EVALUATION_METRIC)
    optimizer: Optimizer = Field(default_factory=defaults.ADAM_OPTIMIZER)
    epochs_count: int
    early_stopping: Optional[TrainEarlyStoppingSettings]
    evaluation_folds_key: EvaluationFoldsKeySettings
    random_seed: Optional[int]
