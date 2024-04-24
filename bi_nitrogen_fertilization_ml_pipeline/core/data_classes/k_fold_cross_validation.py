from typing import Iterable

import numpy as np
import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.base_model import BaseModel


class DatasetFoldSplit(BaseModel):
    fold_key: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_evaluation: pd.DataFrame
    y_evaluation: pd.Series


class FoldModelEvaluationResults(BaseModel):
    fold_key: str
    evaluation_set_size: int
    train_set_loss: float
    train_set_main_metric: float
    evaluation_set_loss: float
    evaluation_set_main_metric: float
    train_random_guess_on_evaluation_set_loss: float
    train_random_guess_on_evaluation_set_main_metric: float


class FoldsResultsAggregation(BaseModel):
    mean: float
    std: float


class KFoldCrossValidationResults(BaseModel):
    fold_results: list[FoldModelEvaluationResults]

    def folds_count(self) -> int:
        return len(self.fold_results)

    def aggregate_train_set_folds_loss(self) -> FoldsResultsAggregation:
        return self._aggregate_fold_values(
            map(lambda fold_res: fold_res.train_set_loss, self.fold_results)
        )

    def aggregate_train_set_folds_main_metric(self) -> FoldsResultsAggregation:
        return self._aggregate_fold_values(
            map(lambda fold_res: fold_res.train_set_main_metric, self.fold_results)
        )

    def aggregate_evaluation_set_folds_loss(self) -> FoldsResultsAggregation:
        return self._aggregate_fold_values(
            map(lambda fold_res: fold_res.evaluation_set_loss, self.fold_results)
        )

    def aggregate_evaluation_set_folds_main_metric(self) -> FoldsResultsAggregation:
        return self._aggregate_fold_values(
            map(lambda fold_res: fold_res.evaluation_set_main_metric, self.fold_results)
        )

    def aggregate_train_random_guess_on_evaluation_set_loss(self) -> FoldsResultsAggregation:
        return self._aggregate_fold_values(
            map(lambda fold_res: fold_res.train_random_guess_on_evaluation_set_loss, self.fold_results)
        )

    def aggregate_train_random_guess_on_evaluation_set_main_metric(self) -> FoldsResultsAggregation:
        return self._aggregate_fold_values(
            map(lambda fold_res: fold_res.train_random_guess_on_evaluation_set_main_metric, self.fold_results)
        )

    @classmethod
    def _aggregate_fold_values(cls, fold_values: Iterable[float]) -> FoldsResultsAggregation:
        fold_values_array = np.array(list(fold_values))
        return FoldsResultsAggregation(
            mean=fold_values_array.mean(),
            std=fold_values_array.std(),
        )
