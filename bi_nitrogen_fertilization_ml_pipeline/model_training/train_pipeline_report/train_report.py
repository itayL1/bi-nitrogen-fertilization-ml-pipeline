from pathlib import Path
from typing import Iterable, Optional

import datapane as dp
import humanize
import pandas as pd
from matplotlib import pyplot as plt

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.evaluation_functions import EvaluationFunctions
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.k_fold_cross_validation import FoldsResultsAggregation
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_params import TrainParams, \
    EvaluationFoldsSplitSettings
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import TrainPipelineReportData, \
    PipelineExecutionTime, ReportWarning, ImputationFunnel, WarningPipelineModules

IMAGE_FILE_EXTENSIONS = ('.jpeg', 'jpg', '.png')


def create_final_train_pipeline_report(
    report_data: TrainPipelineReportData,
    train_params: TrainParams,
    dataset_eda_reports_folder: Optional[Path],
    output_report_html_file_path: Path,
) -> None:
    output_report_html_file_path.parent.mkdir(parents=True, exist_ok=True)

    report = dp.Report(
        _build_pipeline_summary_page(report_data, train_params),
        _build_dataset_eda_page(dataset_eda_reports_folder),
        _build_dataset_preprocessing_page(report_data),
        _build_model_evaluation_page(report_data, train_params),
        _build_final_model_page(report_data),
        _build_warnings_page(report_data),
    )

    report.save(str(output_report_html_file_path), open=False)


def _build_pipeline_summary_page(
    report_data: TrainPipelineReportData,
    train_params: TrainParams,
) -> dp.Page:
    evaluation_folds = report_data.model_training.evaluation_folds
    preprocessed_dataset_shape = report_data.dataset_preprocessing.preprocessed_input_dataset.shape

    eval_metric_display_name = _get_evaluation_function_display_name(train_params.evaluation_metric)
    summary_details = {
        f'K fold model evaluation sets {eval_metric_display_name}': _display_folds_aggregation(
            evaluation_folds.folds_results.aggregate_evaluation_set_folds_main_metric()),
        f'K fold model train sets {eval_metric_display_name}': _display_folds_aggregation(
            evaluation_folds.folds_results.aggregate_train_set_folds_main_metric()),
        f'guess (train set target mean) evaluation sets {eval_metric_display_name}': _display_folds_aggregation(
            evaluation_folds.folds_results.aggregate_train_random_guess_on_evaluation_set_main_metric()),
        'preprocessed dataset size':
            f'{preprocessed_dataset_shape[0]:,} rows, {preprocessed_dataset_shape[1]:,} columns',
        'evaluation folds count': evaluation_folds.folds_results.folds_count(),
        'pipeline execution duration': _get_pipeline_execution_duration_display_text(
            report_data.pipeline_execution_time),
        'warnings raised': _warnings_raised_display_text(report_data.warnings),
        'global random seed': train_params.random_seed if train_params.random_seed is not None else 'not set'
    }
    summary_details_table_df = pd.DataFrame(
        data=[dict(key=key, value=value) for key, value in summary_details.items()]
    )

    pipeline_summary_page = dp.Page(
        title="Summary",
        blocks=[
            dp.Text('## Summary Details'),
            dp.Table(_style_df_cells_to_align_left(summary_details_table_df)),
        ],
    )
    return pipeline_summary_page


def _build_dataset_eda_page(
    dataset_eda_reports_folder: Optional[Path],
) -> dp.Page:
    if dataset_eda_reports_folder is None:
        page_content_blocks = [
            dp.Text('This section is not available in the current report.'),
        ]
    else:
        page_content_blocks = [
            dp.Text('## Dataset EDA reports'),
            dp.Group(
                *(
                    dp.Attachment(file=eda_report_file)
                    for eda_report_file in dataset_eda_reports_folder.iterdir()
                ),
                columns=2,
            )
        ]

    warnings_page = dp.Page(
        title="Dataset EDA",
        blocks=page_content_blocks,
    )
    return warnings_page


def _build_dataset_preprocessing_page(
    report_data: TrainPipelineReportData,
) -> dp.Page:
    dataset_preprocessing = report_data.dataset_preprocessing
    preprocessed_dataset_shape = dataset_preprocessing.preprocessed_input_dataset.shape

    page_details = {
        'preprocessed dataset size':
            f'{preprocessed_dataset_shape[0]:,} rows, {preprocessed_dataset_shape[1]:,} columns',
        'unused dropped columns count': dataset_preprocessing.unused_dropped_columns_count,
        'imputation funnel':
            _imputation_funnel_display_text(dataset_preprocessing.imputation_funnel),
    }
    page_details_table_df = pd.DataFrame(
        data=[dict(key=key, value=value) for key, value in page_details.items()]
    )

    categorical_features_encoding_page_items = (
        (
            dp.Text(f"## '{feature_name}' feature categorical encoding "
                    f"({categorical_encoding_details.encoding_method.value})"),

            dp.Text(f"### categories distribution"),
            dp.DataTable(pd.DataFrame(data=[
                dict(category=category, frequency=frequency_perc)
                for category, frequency_perc
                in categorical_encoding_details.categories_perc_distribution.items()
            ])),
            dp.Plot(
                _plot_categorical_feature_values_frequency_hist(categorical_encoding_details.categories_frequency),
                responsive=False,
            ),

            *((
                dp.Text(f"### 'other' category aggregated categories distribution"),
                dp.DataTable(pd.DataFrame(data=[
                    dict(category=category, frequency=frequency_perc)
                    for category, frequency_perc
                    in categorical_encoding_details.other_category_aggregation.aggregated_categories_distribution.items()
                ])),
            ) if any(categorical_encoding_details.other_category_aggregation.aggregated_categories_distribution) else [])
        )
        for feature_name, categorical_encoding_details
        in report_data.dataset_preprocessing.categorical_features_encoding_details.items()
    )
    flat_categorical_features_encoding_page_items =\
        _flat_nested_collection_by_one_level(categorical_features_encoding_page_items)

    dataset_preprocessing_page = dp.Page(
        title="Dataset Preprocessing",
        blocks=[
            dp.Text('## Dataset preprocessing details'),
            dp.Table(_style_df_cells_to_align_left(page_details_table_df)),
            *flat_categorical_features_encoding_page_items,
        ],
    )
    return dataset_preprocessing_page


def _plot_categorical_feature_values_frequency_hist(
    categories_frequency: dict[str, int],
) -> plt.Figure:
    try:
        plt.figure(figsize=(8, 6))
        pd.Series(categories_frequency).plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Categories frequency')
        plt.xlabel('Category')
        plt.xticks(rotation=45, ha='left')
        plt.ylabel('Frequency')

        return plt.gcf()
    finally:
        plt.close()


def _build_model_evaluation_page(
    report_data: TrainPipelineReportData,
    train_params: TrainParams,
) -> dp.Page:
    evaluation_folds = report_data.model_training.evaluation_folds

    eval_metric_display_name = _get_evaluation_function_display_name(train_params.evaluation_metric)
    page_details = {
        f'K fold model evaluation sets {eval_metric_display_name}': _display_folds_aggregation(
            evaluation_folds.folds_results.aggregate_evaluation_set_folds_main_metric()),
        f'K fold model train sets {eval_metric_display_name}': _display_folds_aggregation(
            evaluation_folds.folds_results.aggregate_train_set_folds_main_metric()),
        f'guess (train set target mean) evaluation sets {eval_metric_display_name}': _display_folds_aggregation(
            evaluation_folds.folds_results.aggregate_train_random_guess_on_evaluation_set_main_metric()),
        'loss function name': _get_evaluation_function_display_name(train_params.loss_function),
        'evaluation folds split setup':
            _evaluation_folds_split_setup_display_text(train_params.evaluation_folds_split),
        'evaluation folds count': evaluation_folds.folds_results.folds_count(),
        'folds distribution GINI coefficient':
            f'{report_data.model_training.evaluation_folds.folds_distribution_gini_coefficient:.4f}'
    }
    page_details_table_df = pd.DataFrame(
        data=[dict(key=key, value=value) for key, value in page_details.items()]
    )

    evaluation_folds_key_frequencies_hist_figure =\
        _get_evaluation_folds_key_frequencies_histogram(report_data, train_params)

    fold_train_models_page_items = _get_folds_models_train_page_items(report_data, train_params)
    model_evaluation_page = dp.Page(
        title="Model Evaluation",
        blocks=[
            dp.Text('## Model evaluation details'),
            dp.Table(_style_df_cells_to_align_left(page_details_table_df)),

            dp.Plot(evaluation_folds_key_frequencies_hist_figure, responsive=False),

            dp.Text('## Regression errors graph'),
            dp.Text('### Deviations of fold models evaluation set predictions from true values'),
            dp.Group(
                dp.Attachment(
                    file=evaluation_folds.folds_evaluation_set_y_true_and_pred_csv_path,
                    caption='this file contains the data points that were used to produce ' +
                            'the following graph',
                ),
                columns=2,
            ),
            dp.Media(evaluation_folds.folds_eval_set_prediction_deviations_graph_file),

            *fold_train_models_page_items,
        ],
    )
    return model_evaluation_page


def _build_final_model_page(
    report_data: TrainPipelineReportData,
) -> dp.Page:
    final_model = report_data.model_training.final_model
    model_architecture_summary = report_data.model_training.model_architecture_summary

    final_model_page = dp.Page(
        title="Final Model",
        blocks=[
            dp.Text('## Model architecture (Keras summary)'),
            _multi_line_text_as_html_block(model_architecture_summary),

            dp.Text('## Final model feature importance'),
            dp.Media(final_model.feature_importance_summary_figure_path),

            dp.Text('## Final model training'),
            dp.Text(f'### train epochs count: {final_model.train_epochs_count}'),
            *(
                dp.Media(folder_child_file)
                for folder_child_file in final_model.train_figures_folder.iterdir()
                if (
                    folder_child_file.suffix in IMAGE_FILE_EXTENSIONS and
                    folder_child_file != final_model.feature_importance_summary_figure_path
                )
            ),
        ],
    )
    return final_model_page


def _build_warnings_page(report_data: TrainPipelineReportData) -> dp.Page:
    if not any(report_data.warnings):
        page_content_blocks = [
            dp.Text('No warnings were detected during the train pipeline execution'),
        ]
    else:
        warnings_table_df = pd.DataFrame(data=[
            {
                'Pipeline Module': _warning_pipeline_module_display_text(warning.pipeline_module),
                'Description': warning.description,
                'Context': _warning_context_display_text(warning.context) if warning.context is not None else '',
            }
            for warning in report_data.warnings
        ])
        page_content_blocks = [
            dp.Table(_style_df_cells_to_align_left(warnings_table_df))
        ]

    warnings_page = dp.Page(
        title="Warnings",
        blocks=page_content_blocks,
    )
    return warnings_page


def _warning_pipeline_module_display_text(warning_pipeline_module: WarningPipelineModules) -> str:
    if warning_pipeline_module is WarningPipelineModules.dataset_preprocessing:
        return 'Dataset Preprocessing'
    elif warning_pipeline_module is WarningPipelineModules.model_training:
        return 'Model training'
    else:
        raise NotImplementedError(f"unknown warning pipeline module: '{warning_pipeline_module}'")


def _warning_context_display_text(warning_context: dict) -> str:
    return '  '.join(
        f'{key}: {val}'
        for key, val in warning_context.items()
    )


def _get_folds_models_train_page_items(
    report_data: TrainPipelineReportData,
    train_params: TrainParams,
):
    evaluation_folds = report_data.model_training.evaluation_folds
    loss_function_name = _get_evaluation_function_display_name(train_params.loss_function)
    evaluation_metric_name = _get_evaluation_function_display_name(train_params.evaluation_metric)
    full_dataset_rows_count = report_data.dataset_preprocessing.preprocessed_input_dataset.shape[0]

    evaluation_folds_split = train_params.evaluation_folds_split
    if evaluation_folds_split.key_column is not None:
        fold_id_table_col = f'fold key ({evaluation_folds_split.key_column})'
    elif evaluation_folds_split.folds_number is not None:
        fold_id_table_col = 'fold id (random split)'
    else:
        raise AssertionError('unknown evaluation folds split setup')

    fold_evaluation_results_table_rows = [
        {
            fold_id_table_col: fold_results.fold_key,
            'train set size': f'{full_dataset_rows_count - fold_results.evaluation_set_size:,}',
            'evaluation set size': f'{fold_results.evaluation_set_size:,}',
            'train epochs count': fold_results.train_epochs_count,
            f'train set model loss ({loss_function_name})': fold_results.train_set_loss,
            f'train set model {evaluation_metric_name}': fold_results.train_set_main_metric,
            f'evaluation set model loss ({loss_function_name})': fold_results.evaluation_set_loss,
            f'evaluation set model {evaluation_metric_name}': fold_results.evaluation_set_main_metric,
            f'guess evaluation set loss ({loss_function_name})': fold_results.train_random_guess_on_evaluation_set_loss,
            f'guess evaluation set {evaluation_metric_name}': fold_results.train_random_guess_on_evaluation_set_main_metric,
        }
        for fold_results in evaluation_folds.folds_results.fold_results
    ]
    fold_evaluation_results_table_df =\
        pd.DataFrame(fold_evaluation_results_table_rows)\
        .sort_values(by=fold_id_table_col)\
        .round(3)

    folds_train_figures_root_folder = evaluation_folds.folds_train_figures_root_folder
    fold_train_figures_page_items = (
        (
            dp.Text(f"### {folder_path.name}"),
            *(
                dp.Media(folder_child_file)
                for folder_child_file in folder_path.iterdir()
                if folder_child_file.suffix in IMAGE_FILE_EXTENSIONS
            )
        )
        for folder_path in sorted(folds_train_figures_root_folder.iterdir())
    )
    fold_train_figures_page_items_flatted =\
        _flat_nested_collection_by_one_level(fold_train_figures_page_items)

    return (
        dp.Text('## Fold models evaluation results and train graphs'),
        dp.Table(fold_evaluation_results_table_df),
        *fold_train_figures_page_items_flatted
    )


def _imputation_funnel_display_text(imputation_funnel: ImputationFunnel) -> str:
    return (
        f'{imputation_funnel.remaining_rows_percentage} rows remained after the imputation funnel '
        f'(before: {imputation_funnel.rows_count_before_imputation:,} | '
        f'after: {imputation_funnel.rows_count_after_imputation:,})'
    )


def _get_evaluation_folds_key_frequencies_histogram(
    report_data: TrainPipelineReportData,
    train_params: TrainParams,
) -> plt.Figure:
    evaluation_folds_key_col = \
        report_data.dataset_preprocessing.preprocessed_input_dataset['evaluation_folds_key']

    evaluation_folds_split = train_params.evaluation_folds_split
    if evaluation_folds_split.key_column is not None:
        graph_title = 'Evaluation fold key values frequency'
    elif evaluation_folds_split.folds_number is not None:
        graph_title = 'Evaluation fold sizes'
    else:
        raise AssertionError('unknown evaluation folds split setup')

    try:
        evaluation_fold_keys_frequency = evaluation_folds_key_col.value_counts()
        evaluation_fold_keys_frequency.sort_index().plot(
            kind='bar', color='skyblue', edgecolor='black',
        )
        plt.title(graph_title)
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


def _evaluation_folds_split_setup_display_text(evaluation_folds_split: EvaluationFoldsSplitSettings) -> str:
    if evaluation_folds_split.key_column is not None:
        return f"by the key column '{evaluation_folds_split.key_column}'"
    elif evaluation_folds_split.folds_number is not None:
        return 'random split'
    else:
        raise AssertionError('unknown evaluation folds split setup')


def _display_folds_aggregation(result_aggregation: FoldsResultsAggregation) -> str:
    return f'mean: {round(result_aggregation.mean, 3)} (std: {round(result_aggregation.std, 3)})'


def _style_df_cells_to_align_left(df: pd.DataFrame):
    return\
        df.style.set_properties(**{'text-align': 'left'}) \
        .set_table_styles([dict(selector='th', props=[('text-align', 'left')])])


def _flat_nested_collection_by_one_level(nested_collection: Iterable) -> tuple:
    return tuple(
        item
        for child_collection in nested_collection
        for item in child_collection
    )


def _multi_line_text_as_html_block(text: str) -> dp.HTML:
    return dp.HTML(f'<pre>{text}</pre>')
