from pathlib import Path

import pandas as pd
import ydata_profiling

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import DatasetPreprocessing


def create_preprocessing_datasets_eda_reports(
    dataset_preprocessing: DatasetPreprocessing,
    report_output_root_folder: Path,
) -> None:
    eda_reports_folder = report_output_root_folder / 'EDA_reports'
    assert dataset_preprocessing.raw_dataset_columns_required_for_training is not None,\
        'the field required_columns_for_training is required by this module'

    original_input_dataset = dataset_preprocessing.original_input_dataset
    if original_input_dataset is not None:
        original_input_dataset_training_columns_only = original_input_dataset[
            list(dataset_preprocessing.raw_dataset_columns_required_for_training)
        ]
        _create_dataset_eda_report(
            original_input_dataset_training_columns_only,
            report_file_path=eda_reports_folder / 'original_input_dataset_eda_report.html',
            report_title='Original Input Dataset EDA report',
        )

    preprocessed_input_dataset = dataset_preprocessing.preprocessed_input_dataset
    if preprocessed_input_dataset is not None:
        _create_dataset_eda_report(
            preprocessed_input_dataset,
            report_file_path=eda_reports_folder / 'preprocessed_input_dataset_eda_report.html',
            report_title='Preprocessed Input Dataset EDA report',
        )


def _create_dataset_eda_report(
    dataset_df: pd.DataFrame,
    report_file_path: Path,
    report_title: str,
) -> None:
    assert report_file_path.suffix == '.html', 'only output html files are supported'
    report_file_path.parent.mkdir(parents=True, exist_ok=True)

    eda_report = ydata_profiling.ProfileReport(
        dataset_df, title=report_title, lazy=False)
    eda_report.to_file(report_file_path)
