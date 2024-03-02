from typing import Callable

import keras.losses
import keras.metrics
from keras import backend as K

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.evaluation_functions import EvaluationFunctions


def eval_func_to_keras_loss(eval_func: EvaluationFunctions) -> keras.losses.Loss | Callable:
    if eval_func == EvaluationFunctions.mse:
        return keras.losses.MeanSquaredError()
    elif eval_func == EvaluationFunctions.rmse:
        return _root_mean_squared_error_loss
    elif eval_func == EvaluationFunctions.mae:
        return keras.losses.MeanAbsoluteError()
    elif eval_func == EvaluationFunctions.huber_loss:
        return keras.losses.Huber()
    else:
        raise NotImplementedError(
            f"the conversion of the evaluation function '{eval_func}' to a keras "
            f"loss function is not supported"
        )


def eval_func_to_keras_metric(eval_func: EvaluationFunctions) -> keras.metrics.Metric:
    if eval_func == EvaluationFunctions.mse:
        return keras.metrics.MeanSquaredError()
    elif eval_func == EvaluationFunctions.rmse:
        return keras.metrics.RootMeanSquaredError()
    elif eval_func == EvaluationFunctions.mae:
        return keras.metrics.MeanAbsoluteError()
    else:
        raise NotImplementedError(
            f"the conversion of the evaluation function '{eval_func}' to a keras "
            f"metric function is not supported"
        )


def _root_mean_squared_error_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
