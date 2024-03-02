from keras.models import Model


def is_model_compiled(model: Model) -> bool:
    return model.optimizer is not None
