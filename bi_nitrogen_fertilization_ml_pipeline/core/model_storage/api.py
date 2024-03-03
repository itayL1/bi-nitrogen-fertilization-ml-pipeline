import shutil
import tempfile
from pathlib import Path

import keras

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.model_storage.keras_model_storage import store_trained_keras_model, \
    load_trained_keras_model
from bi_nitrogen_fertilization_ml_pipeline.core.model_storage.train_artifcats_storage import store_train_artifacts, \
    load_train_artifacts

KERAS_MODEL_STORAGE_FILE_NAME = 'model.keras'
TRAIN_ARTIFACTS_STORAGE_FILE_NAME = 'train_artifacts.pkl'
ARCHIVE_EXTENSIONS = 'zip'


def store_trained_model(
    trained_model: keras.Model,
    train_artifacts: TrainArtifacts,
    output_file_path: Path,
) -> None:
    _validate_storage_file_path(output_file_path)
    output_file_path.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory() as wip_folder_path:
        wip_folder_path = Path(wip_folder_path) / 'wip'
        wip_folder_path.mkdir(parents=False, exist_ok=False)

        store_trained_keras_model(trained_model, wip_folder_path / KERAS_MODEL_STORAGE_FILE_NAME)
        store_train_artifacts(train_artifacts, wip_folder_path / TRAIN_ARTIFACTS_STORAGE_FILE_NAME)

        output_file_path_without_file_extension = output_file_path.parent / output_file_path.stem
        shutil.make_archive(str(output_file_path_without_file_extension), ARCHIVE_EXTENSIONS, wip_folder_path)


def load_trained_model(
    model_file_path: Path,
) -> tuple[keras.Model, TrainArtifacts]:
    _validate_storage_file_path(model_file_path)

    with tempfile.TemporaryDirectory() as wip_folder_path:
        wip_folder_path = Path(wip_folder_path)
        shutil.unpack_archive(model_file_path, str(wip_folder_path), ARCHIVE_EXTENSIONS)
        loaded_trained_model =\
            load_trained_keras_model(wip_folder_path / KERAS_MODEL_STORAGE_FILE_NAME)
        loaded_train_artifacts =\
            load_train_artifacts(wip_folder_path / TRAIN_ARTIFACTS_STORAGE_FILE_NAME)

    return loaded_trained_model, loaded_train_artifacts


def _validate_storage_file_path(storage_file_path: Path):
    expected_file_extension = f'.{ARCHIVE_EXTENSIONS}'
    assert storage_file_path.suffix == expected_file_extension, \
        f"only storage paths with the {expected_file_extension} extensions are supported"
