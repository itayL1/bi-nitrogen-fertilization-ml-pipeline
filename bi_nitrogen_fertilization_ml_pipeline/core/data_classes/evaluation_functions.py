from enum import Enum


class EvaluationFunctions(str, Enum):
    mse = 'mse'
    rmse = 'rmse'
    mae = 'mae'
    huber_loss = 'huber_loss'
