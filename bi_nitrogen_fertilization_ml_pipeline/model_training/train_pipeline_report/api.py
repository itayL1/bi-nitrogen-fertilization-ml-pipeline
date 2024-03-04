from pathlib import Path

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import TrainPipelineReportData


def create_and_save_train_pipeline_report(
    report_data: TrainPipelineReportData,
    report_output_root_folder: Path
) -> None:
    report_output_root_folder.mkdir(parents=True, exist_ok=True)

    train_pipeline_report_dump = \
        report_data.copy_without_large_members().json(ensure_ascii=False, indent=4)
    with open(report_output_root_folder / 'train_pipeline_report_dump.json', 'w') as f:
        f.write(train_pipeline_report_dump)
