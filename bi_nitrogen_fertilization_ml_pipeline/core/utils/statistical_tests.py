from enum import Enum

import numpy as np
import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.base_model import BaseModel


class DistributionBalance(Enum):
    relatively_balanced = 'relatively_balanced'
    moderately_imbalanced = 'moderately_imbalanced'
    highly_imbalanced = 'highly_imbalanced'
    unknown = 'unknown'


class GiniCoefficientTestResult(BaseModel):
    gini_coefficient: float
    distribution_balance: DistributionBalance


def values_distribution_gini_coefficient_test(series: pd.Series) -> GiniCoefficientTestResult:
    array_gini_coefficient = _calc_gini_coefficient(series)
    array_distribution_balance = _determine_distribution_balance(array_gini_coefficient)
    return GiniCoefficientTestResult(
        gini_coefficient=array_gini_coefficient,
        distribution_balance=array_distribution_balance,
    )


def _calc_gini_coefficient(series: pd.Series) -> float:
    category_indices = {category: i for i, category in enumerate(np.unique(series))}
    series_value_indices = np.array([category_indices[category] for category in series])

    mad = np.abs(np.subtract.outer(series_value_indices, series_value_indices)).mean()
    rmad = mad / np.mean(series_value_indices)
    gini_value = 0.5 * rmad
    return gini_value


def _determine_distribution_balance(gini_coefficient: float) -> DistributionBalance:
    if 0.0 <= gini_coefficient < 0.2:
        return DistributionBalance.relatively_balanced
    elif 0.2 <= gini_coefficient < 0.4:
        return DistributionBalance.moderately_imbalanced
    elif 0.4 <= gini_coefficient < 1.0:
        return DistributionBalance.highly_imbalanced
    else:
        return DistributionBalance.unknown
