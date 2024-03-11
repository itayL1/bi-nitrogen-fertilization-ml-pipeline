from pathlib import Path

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import TrainPipelineReportData
from bi_nitrogen_fertilization_ml_pipeline.model_training.train_pipeline_report.eda import \
    create_preprocessing_datasets_eda_reports


def create_and_save_train_pipeline_report(
    report_data: TrainPipelineReportData,
    report_output_root_folder: Path
) -> None:
    report_output_root_folder.mkdir(parents=True, exist_ok=True)

    train_pipeline_report_dump = \
        report_data.copy_without_large_members().json(ensure_ascii=False, indent=4)
    with open(report_output_root_folder / 'train_pipeline_report_dump.json', 'w') as f:
        f.write(train_pipeline_report_dump)

    create_preprocessing_datasets_eda_reports(
        report_data.dataset_preprocessing, report_output_root_folder)

    report_asserts_folder = report_output_root_folder / 'assets'
    report_asserts_folder.mkdir(parents=False, exist_ok=False)
    report_data.model_training.evaluation_folds_train_figures_root_folder.rename(
        report_asserts_folder / report_data.model_training.evaluation_folds_train_figures_root_folder.name)
    report_data.model_training.final_model_train_figures_folder.rename(
        report_asserts_folder / report_data.model_training.final_model_train_figures_folder.name)
