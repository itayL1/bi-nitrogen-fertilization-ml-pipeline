from keras.models import Model
from keras.src.callbacks import History


def is_model_compiled(model: Model) -> bool:
    return model.optimizer is not None


def extract_train_epochs_count(train_history: History) -> int:
    return len(next(iter(train_history.history.values())))


def get_model_architecture_summary(model: Model) -> str:
    model_summary_lines = []
    model.summary(print_fn=lambda x: model_summary_lines.append(x))
    model_summary_text = '\n'.join(model_summary_lines)
    return model_summary_text
