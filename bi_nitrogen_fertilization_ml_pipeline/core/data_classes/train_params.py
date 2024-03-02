from typing import Optional, Callable

from pydantic import validator

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.base_model import BaseModel
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.field_utils import not_empty_str


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
    evaluation_folds_key: EvaluationFoldsKeySettings
    random_seed: Optional[int]
