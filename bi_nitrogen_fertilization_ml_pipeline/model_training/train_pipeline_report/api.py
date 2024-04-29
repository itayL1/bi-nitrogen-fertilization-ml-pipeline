from pathlib import Path

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import TrainPipelineReportData
from bi_nitrogen_fertilization_ml_pipeline.model_training.train_pipeline_report.eda import \
    create_preprocessing_datasets_eda_reports
from bi_nitrogen_fertilization_ml_pipeline.model_training.train_pipeline_report.train_report import create_final_train_pipeline_report


def create_and_save_train_pipeline_report(
    report_data: TrainPipelineReportData,
    train_params: TrainParams,
    output_report_html_file_path: Path,
    wip_outputs_folder_path: Path,
) -> None:
    wip_outputs_folder_path.mkdir(parents=True, exist_ok=True)

    output_dataset_eda_reports_folder = None
    if train_params.create_dataset_eda_reports:
        output_dataset_eda_reports_folder = wip_outputs_folder_path / 'EDA_reports'
        create_preprocessing_datasets_eda_reports(
            report_data.dataset_preprocessing,
            output_eda_reports_folder=output_dataset_eda_reports_folder,
        )

    create_final_train_pipeline_report(
        report_data, train_params,
        dataset_eda_reports_folder=output_dataset_eda_reports_folder,
        output_report_html_file_path=output_report_html_file_path,
    )
