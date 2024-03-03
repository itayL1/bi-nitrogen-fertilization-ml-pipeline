import pickle
from pathlib import Path

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import TrainArtifacts


def store_train_artifacts(
    train_artifacts: TrainArtifacts,
    output_artifacts_file_path: Path,
) -> None:
    with open(output_artifacts_file_path, 'wb') as f:
        pickle.dump(train_artifacts, f)


def load_train_artifacts(stored_artifacts_file_path: Path) -> TrainArtifacts:
    with open(stored_artifacts_file_path, 'rb') as file:
        loaded_train_artifacts = pickle.load(file)
    assert isinstance(loaded_train_artifacts, TrainArtifacts), 'unexpected state'
    return loaded_train_artifacts
