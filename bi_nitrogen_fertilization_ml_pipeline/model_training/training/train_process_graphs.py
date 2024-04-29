from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.evaluation_functions import EvaluationFunctions
from bi_nitrogen_fertilization_ml_pipeline.core.plot_settings import RECTANGLE_FIGURES_SIZE, SAVE_FIGURES_DPI
from bi_nitrogen_fertilization_ml_pipeline.model_training.utils.display_utils import \
    get_evaluation_function_display_name


def plot_evaluation_value_per_training_epoch_graph(
    group_name_to_ordered_epoch_values: dict[str, list[float]],
    eval_value_func: EvaluationFunctions,
    is_loss_function: bool,
    output_graph_jpeg_file_path: Path,
    graph_sub_title: Optional[str] = None,
    y_axis_min_limit: Optional[float] = None,
    y_axis_max_limit: Optional[float] = None,
):
    fig, ax = plt.subplots(figsize=RECTANGLE_FIGURES_SIZE)
    try:
        fig.set_dpi(SAVE_FIGURES_DPI)
        ordered_line_names = []
        for group_name, ordered_epoch_values in group_name_to_ordered_epoch_values.items():
            ordered_epoch_capped_values = _cap_values_between_limits(
                ordered_epoch_values, y_axis_min_limit, y_axis_max_limit)
            ordered_one_based_epoch_numbers = range(1, len(ordered_epoch_values) + 1)
            ax.plot(ordered_one_based_epoch_numbers, ordered_epoch_capped_values)
            ordered_line_names.append(group_name)

        eval_func_display_name = get_evaluation_function_display_name(eval_value_func)
        graph_title =\
            f'Loss ({eval_func_display_name}) per epoch' if is_loss_function\
            else f'{eval_func_display_name} per epoch'
        ax.set_title(graph_title)
        if graph_sub_title is not None:
            fig.suptitle(graph_sub_title)
        ax.set_ylim(ymin=y_axis_min_limit, ymax=y_axis_max_limit)
        ax.set_ylabel(eval_func_display_name)
        ax.set_xlabel('epoch')
        ax.legend(ordered_line_names)

        fig.savefig(str(output_graph_jpeg_file_path))
    finally:
        plt.close(fig)


def _cap_values_between_limits(
    values: list[float],
    min_value: Optional[float],
    max_value: Optional[float],
) -> list[float]:
    ret_values = values
    if min_value is not None:
        ret_values = [
            max(val, min_value)
            for val in values
        ]
    if max_value is not None:
        ret_values = [
            min(val, max_value)
            for val in values
        ]
    return ret_values
