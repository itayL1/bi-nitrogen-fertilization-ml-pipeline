from keras.models import Model
from keras.src.callbacks import History


def is_model_compiled(model: Model) -> bool:
    return model.optimizer is not None


def extract_train_epochs_count(train_history: History) -> int:
    return len(next(iter(train_history.history.values())))
