from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.evaluation_functions import EvaluationFunctions


def get_evaluation_function_display_name(eval_func: EvaluationFunctions) -> str:
    if eval_func == EvaluationFunctions.mse:
        return 'MSE'
    elif eval_func == EvaluationFunctions.rmse:
        return 'RMSE'
    elif eval_func == EvaluationFunctions.mae:
        return 'MAE'
    elif eval_func == EvaluationFunctions.huber_loss:
        return 'Huber'
    else:
        raise NotImplementedError(f"unknown evaluation function '{eval_func}'")
