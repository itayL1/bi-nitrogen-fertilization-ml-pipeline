from pathlib import Path

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import TrainPipelineReportData
from bi_nitrogen_fertilization_ml_pipeline.model_training.train_pipeline_report.eda import \
    create_preprocessing_datasets_eda_reports
from bi_nitrogen_fertilization_ml_pipeline.model_training.train_pipeline_report.train_report import create_train_report


def create_and_save_train_pipeline_report(
    report_data: TrainPipelineReportData,
    train_params: TrainParams,
    report_output_root_folder: Path,
) -> None:
    report_output_root_folder.mkdir(parents=True, exist_ok=True)

    output_dataset_eda_reports_folder = None
    if train_params.create_dataset_eda_reports:
        output_dataset_eda_reports_folder = report_output_root_folder / 'EDA_reports'
        create_preprocessing_datasets_eda_reports(
            report_data.dataset_preprocessing,
            output_eda_reports_folder=output_dataset_eda_reports_folder,
        )

    create_train_report(
        report_data, train_params,
        dataset_eda_reports_folder=output_dataset_eda_reports_folder,
        output_report_html_file_path=report_output_root_folder / 'train_pipeline_report.html',
    )

    report_asserts_folder = report_output_root_folder / 'assets'
    report_asserts_folder.mkdir(parents=False, exist_ok=False)
    report_data.model_training.evaluation_folds.folds_train_figures_root_folder.rename(
        report_asserts_folder / report_data.model_training.evaluation_folds.folds_train_figures_root_folder.name)
    report_data.model_training.final_model.train_figures_folder.rename(
        report_asserts_folder / report_data.model_training.final_model.train_figures_folder.name)
