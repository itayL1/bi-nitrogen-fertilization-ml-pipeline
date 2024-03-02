import numpy as np
import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.evaluation_functions import EvaluationFunctions
from bi_nitrogen_fertilization_ml_pipeline.model_training.utils.train_params_to_keras_api_conversions import \
    eval_func_to_keras_metric


def calculate_evaluation_metric_for_random_guess_predictions(
    y_true: pd.Series,
    eval_func: EvaluationFunctions,
) -> float:
    eval_metric = eval_func_to_keras_metric(eval_func)

    random_guess_predictions = np.full_like(y_true, np.mean(y_true))
    eval_metric.update_state(y_true, random_guess_predictions)
    metric_value = float(eval_metric.result().numpy())

    return metric_value
