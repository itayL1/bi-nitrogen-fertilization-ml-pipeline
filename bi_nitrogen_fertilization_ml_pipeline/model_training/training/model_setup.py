import keras
import keras.optimizers
from keras import Model
from keras.src.optimizers.legacy import optimizer_v2

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams
from bi_nitrogen_fertilization_ml_pipeline.model_training.utils.keras_utils import is_model_compiled
from bi_nitrogen_fertilization_ml_pipeline.model_training.utils.train_params_to_keras_api_conversions import \
    eval_func_to_keras_loss, eval_func_to_keras_metric


def prepare_new_model_for_training(
    train_params: TrainParams,
    input_features_count: int,
) -> Model:
    model = train_params.model_builder(input_features_count)
    assert model is not None, \
        "the provided model builder returned None, this isn't allowed"
    assert isinstance(model, keras.Model), \
        f"the provided model builder returned an instance of '{type(model)}', " \
        f"while only '{keras.Model}' instances are allowed."
    assert not is_model_compiled(model), \
        "the provided model builder a compiled model. the compilation of the model " \
        "must be executed as a part of this pipeline. please adjust the model builder " \
        "so it won't compile the model before retrieving it."

    _compile_model(model, train_params)
    return model


def _compile_model(model: Model, train_params: TrainParams) -> None:
    loss = eval_func_to_keras_loss(train_params.loss_function)
    evaluation_metric = eval_func_to_keras_metric(train_params.evaluation_metric)
    optimizer = train_params.optimizer_builder()
    assert isinstance(optimizer, (keras.optimizers.Optimizer, optimizer_v2.OptimizerV2)), \
        f"the provided optimizer builder returned an instance of '{type(optimizer)}', " \
        f"while only '{keras.optimizers.Optimizer}' instances are allowed."
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[evaluation_metric],
    )
