from pathlib import Path

import humanize
import pandas as pd
import datapane as dp
from matplotlib import pyplot as plt

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.evaluation_functions import EvaluationFunctions
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.k_fold_cross_validation import FoldsResultsAggregation
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import TrainPipelineReportData, \
    PipelineExecutionTime, ReportWarning


def create_train_report(
    report_data: TrainPipelineReportData,
    train_params: TrainParams,
    output_report_html_file_path: Path
) -> None:
    output_report_html_file_path.parent.mkdir(parents=True, exist_ok=True)

    report = dp.Report(
        _build_pipeline_summary_page(report_data, train_params),
        _build_model_evaluation_page(report_data, train_params),
    )
    report.save(str(output_report_html_file_path), open=False)


def _build_pipeline_summary_page(
    report_data: TrainPipelineReportData,
    train_params: TrainParams,
) -> dp.Page:
    training_evaluation_folds_results = report_data.model_training.evaluation_folds_results
    preprocessed_dataset_shape = report_data.dataset_preprocessing.preprocessed_input_dataset.shape

    summary_details = {
        'K fold cross validation model evaluation sets metric': _display_folds_aggregation(
            training_evaluation_folds_results.aggregate_evaluation_set_folds_main_metric()),
        'K fold cross validation model train sets metric': _display_folds_aggregation(
            training_evaluation_folds_results.aggregate_train_set_folds_main_metric()),
        'guess (train set target mean) evaluation sets metric': _display_folds_aggregation(
            training_evaluation_folds_results.aggregate_train_random_guess_on_evaluation_set_main_metric()),
        'preprocessed dataset size':
            f'{preprocessed_dataset_shape[0]:,} rows, {preprocessed_dataset_shape[1]:,} columns',
        'evaluation metric name': _get_evaluation_function_display_name(train_params.evaluation_metric),
        'evaluation folds count': training_evaluation_folds_results.folds_count(),
        'pipeline execution duration': _get_pipeline_execution_duration_display_text(
            report_data.pipeline_execution_time),
        'warnings raised': _warnings_raised_display_text(report_data.warnings),
    }
    summary_details_table_df = pd.DataFrame(
        data=[dict(key=key, value=value) for key, value in summary_details.items()]
    )

    pipeline_summary_page = dp.Page(
        title="Summary",
        blocks=[
            dp.Text('### Summary Details'),
            dp.Table(_style_df_cells_to_align_left(summary_details_table_df)),
        ],
    )
    return pipeline_summary_page


def _build_model_evaluation_page(
    report_data: TrainPipelineReportData,
    train_params: TrainParams,
) -> dp.Page:
    training_evaluation_folds_results = report_data.model_training.evaluation_folds_results

    page_details = {
        'K fold cross validation model evaluation sets metric': _display_folds_aggregation(
            training_evaluation_folds_results.aggregate_evaluation_set_folds_main_metric()),
        'K fold cross validation model train sets metric': _display_folds_aggregation(
            training_evaluation_folds_results.aggregate_train_set_folds_main_metric()),
        'guess (train set target mean) evaluation sets metric': _display_folds_aggregation(
            training_evaluation_folds_results.aggregate_train_random_guess_on_evaluation_set_main_metric()),
        'evaluation metric name': _get_evaluation_function_display_name(train_params.evaluation_metric),
        'loss function name': _get_evaluation_function_display_name(train_params.loss_function),
        'evaluation folds key column':
            f"{train_params.evaluation_folds_key.column}{' (mutated)' if train_params.evaluation_folds_key.values_mapper is not None else ''}",
        'evaluation folds count': training_evaluation_folds_results.folds_count(),
        'folds distribution GINI coefficient':
            f'{report_data.model_training.evaluation_folds_distribution_gini_coefficient:.4f}'
    }
    page_details_table_df = pd.DataFrame(
        data=[dict(key=key, value=value) for key, value in page_details.items()]
    )

    evaluation_folds_key_frequencies_hist_figure =\
        _get_evaluation_folds_key_frequencies_histogram(report_data)

    fold_train_models_page_items = _get_folds_models_train_page_items(report_data, train_params)

    model_evaluation_page = dp.Page(
        title="Model evaluation",
        blocks=[
            dp.Text('## Model evaluation details'),
            dp.Table(_style_df_cells_to_align_left(page_details_table_df)),

            dp.Plot(evaluation_folds_key_frequencies_hist_figure, responsive=False, scale=1.5),

            *fold_train_models_page_items,
        ],
    )

    return model_evaluation_page


def _get_folds_models_train_page_items(
    report_data: TrainPipelineReportData,
    train_params: TrainParams,
):
    loss_function_name = _get_evaluation_function_display_name(train_params.loss_function)
    evaluation_metric_name = _get_evaluation_function_display_name(train_params.evaluation_metric)
    full_dataset_rows_count = report_data.dataset_preprocessing.preprocessed_input_dataset.shape[0]

    fold_key_table_col = f'fold key ({train_params.evaluation_folds_key.column})'
    fold_evaluation_results_table_rows = [
        {
            fold_key_table_col: fold_results.fold_key,
            'train set size': f'{full_dataset_rows_count - fold_results.evaluation_set_size:,}',
            'evaluation set size': f'{fold_results.evaluation_set_size:,}',
            f'train set model loss ({loss_function_name})': fold_results.train_set_loss,
            f'train set model {evaluation_metric_name}': fold_results.train_set_main_metric,
            f'evaluation set model loss ({loss_function_name})': fold_results.evaluation_set_loss,
            f'evaluation set model {evaluation_metric_name}': fold_results.evaluation_set_main_metric,
            f'guess evaluation set loss ({loss_function_name})': fold_results.train_random_guess_on_evaluation_set_loss,
            f'guess evaluation set {evaluation_metric_name}': fold_results.train_random_guess_on_evaluation_set_main_metric,
        }
        for fold_results in report_data.model_training.evaluation_folds_results.fold_results
    ]
    fold_evaluation_results_table_df =\
        pd.DataFrame(fold_evaluation_results_table_rows)\
        .sort_values(by=fold_key_table_col)\
        .round(3)

    folds_train_figures_subfolders = \
        report_data.model_training.evaluation_folds_train_figures_root_folder.iterdir()
    fold_train_figures_page_items = (
        (
            dp.Text(f"### {folder_path.name}"),
            *(
                dp.Media(folder_child_file)
                for folder_child_file in folder_path.iterdir()
                if folder_child_file.suffix in ('.jpeg', 'jpg', '.png')
            )
        )
        for folder_path in folds_train_figures_subfolders
    )
    fold_train_figures_page_items_flatted = (
        page_item
        for fold_page_items in fold_train_figures_page_items
        for page_item in fold_page_items
    )

    return (
        dp.Text('## Fold models evaluation results and train graphs'),
        dp.Table(fold_evaluation_results_table_df),
        *fold_train_figures_page_items_flatted
    )


def _get_evaluation_folds_key_frequencies_histogram(report_data: TrainPipelineReportData) -> plt.Figure:
    evaluation_folds_key_col = \
        report_data.dataset_preprocessing.preprocessed_input_dataset['evaluation_folds_key']
    try:
        # plt.figure(figsize=(4, 3))
        evaluation_folds_key_col.value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Evaluation Fold key values frequency')
        plt.xlabel('Fold key')
        plt.xticks(rotation=45, ha='left')
        plt.ylabel('Frequency')

        return plt.gcf()
    finally:
        plt.close()


def _get_pipeline_execution_duration_display_text(pipeline_execution_time: PipelineExecutionTime) -> str:
    datetime_display_format = '%H:%M:%S %d/%m/%Y'
    pipeline_start_timestamp = pipeline_execution_time.pipeline_start_timestamp
    pipeline_end_timestamp = pipeline_execution_time.pipeline_end_timestamp

    pipeline_start_timestamp_text =\
        pipeline_start_timestamp.strftime(datetime_display_format)\
        if pipeline_start_timestamp is not None else 'unknown'
    pipeline_end_timestamp_text = \
        pipeline_end_timestamp.strftime(datetime_display_format)\
        if pipeline_end_timestamp is not None else 'unknown'

    if pipeline_start_timestamp is not None and pipeline_end_timestamp is not None:
        pipeline_duration_text = humanize.precisedelta(pipeline_start_timestamp - pipeline_end_timestamp)
    else:
        pipeline_duration_text = 'unknown'

    return f"{pipeline_duration_text} (start: {pipeline_start_timestamp_text} | end: {pipeline_end_timestamp_text})"


def _get_evaluation_function_display_name(eval_func: EvaluationFunctions) -> str:
    if eval_func is EvaluationFunctions.mse:
        return 'MSE'
    elif eval_func is EvaluationFunctions.rmse:
        return 'RMSE'
    else:
        raise NotImplementedError(f"unknown evaluation function: '{eval_func}'")


def _warnings_raised_display_text(warnings: list[ReportWarning]) -> str:
    if not any(warnings):
        return 'No'
    else:
        return f'Yes ({len(warnings)} in total)'


def _display_folds_aggregation(result_aggregation: FoldsResultsAggregation) -> str:
    return f'{round(result_aggregation.mean, 3)} (std: {round(result_aggregation.std, 3)})'


def _style_df_cells_to_align_left(df: pd.DataFrame):
    return\
        df.style.set_properties(**{'text-align': 'left'}) \
        .set_table_styles([dict(selector='th', props=[('text-align', 'left')])])


if __name__ == '__main__':
    import keras.optimizers.legacy
    from bi_nitrogen_fertilization_ml_pipeline.assets.baseline_model import init_baseline_model
    from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import \
        EvaluationFoldsKeySettings, TrainEarlyStoppingSettings

    pipeline_file_path_ = '/Users/itaylotan/git/bi-nitrogen-fertilization-ml-pipeline/scratch_359_outputs3/train_pipeline_report/train_pipeline_report_dump.json'

    output_report_html_file_path_ = Path('/Users/itaylotan/git/bi-nitrogen-fertilization-ml-pipeline/bi_nitrogen_fertilization_ml_pipeline/model_training/train_pipeline_report/tmp/dummy_report.html')
    create_train_report(
        report_data=TrainPipelineReportData.parse_file(pipeline_file_path_),
        train_params=TrainParams(
            model_builder=init_baseline_model,
            epochs_count=100,
            # epochs_count=5,
            evaluation_folds_key=EvaluationFoldsKeySettings(
                column='year',
                values_mapper=lambda year_str: str(int(year_str.strip()) % 3),
            ),
            early_stopping=TrainEarlyStoppingSettings(
                validation_set_fraction_size=0.2,
                tolerance_epochs_count=9,
                # tolerance_epochs_count=2,
            ),
            optimizer_builder=keras.optimizers.legacy.Adam,
            random_seed=42,
            silent_models_fitting=True,
        ),
        output_report_html_file_path=output_report_html_file_path_,
    )

    def _open_path_in_browser(file_path: str | Path):
        import subprocess
        subprocess.run(f"open -a 'google chrome' '{str(file_path)}'", shell=True)

    _open_path_in_browser(output_report_html_file_path_)
