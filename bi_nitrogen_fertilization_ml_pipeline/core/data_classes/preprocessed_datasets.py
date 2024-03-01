from typing import Optional

import pandas as pd
from pydantic import validator

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.base_model import BaseModel
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.validations import validate_dataframe_has_2_dimensions


class PreprocessedTrainDataset(BaseModel):
    X: pd.DataFrame
    y: pd.Series
    evaluation_folds_key_col: pd.Series

    def total_columns_count(self) -> int:
        return sum(
            1 if prop_val.ndim == 1 else prop_val.shape[1]
            for prop_val in self.dict().values()
        )

    def get_full_dataset(self) -> pd.DataFrame:
        full_dataset_df = self.X
        full_dataset_df['y'] = self.y
        full_dataset_df['evaluation_folds_key'] = self.evaluation_folds_key_col
        return full_dataset_df.copy()

    @validator('X')
    def _validate_X_has_2_dimensions(cls, X: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        validate_dataframe_has_2_dimensions(X)
        return X


class PreprocessedInferenceDataset(BaseModel):
    X: pd.DataFrame

    @validator('X')
    def _validate_X_has_2_dimensions(cls, X: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        validate_dataframe_has_2_dimensions(X)
        return X