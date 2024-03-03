from pathlib import Path

from keras import Model
from keras.models import load_model


def store_trained_keras_model(
    trained_model: Model,
    output_model_file_path: Path,
) -> None:
    _validate_model_file_path(output_model_file_path)
    trained_model.save(str(output_model_file_path))


def load_trained_keras_model(stored_model_file_path: Path) -> Model:
    _validate_model_file_path(stored_model_file_path)
    loaded_trained_model = load_model(str(stored_model_file_path))
    return loaded_trained_model


def _validate_model_file_path(output_model_file_path):
    assert output_model_file_path.suffix == '.keras', \
        "only paths with the .keras extensions are supported"
