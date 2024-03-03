import numpy as np
import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.evaluation_functions import EvaluationFunctions
from bi_nitrogen_fertilization_ml_pipeline.model_training.utils.train_params_to_keras_api_conversions import \
    eval_func_to_keras_metric


def calculate_evaluation_metric_for_random_guess_predictions(
    y_train: pd.Series,
    y_evaluation: pd.Series,
    eval_func: EvaluationFunctions,
) -> float:
    eval_metric = eval_func_to_keras_metric(eval_func)

    random_guess_predictions = np.full_like(y_evaluation, np.mean(y_train))
    eval_metric.update_state(y_evaluation, random_guess_predictions)
    metric_value_on_evaluation_set = float(eval_metric.result().numpy())

    return metric_value_on_evaluation_set
