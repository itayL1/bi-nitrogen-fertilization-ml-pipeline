from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt

FIGURES_SIZE = (12, 5)
SAVE_FIGURES_DPI = 300


def plot_evaluation_value_per_training_epoch_graph(
    group_name_to_ordered_epoch_values: dict[str, list[float]],
    eval_value_display_name: str,
    output_graph_jpeg_file_path: Path,
    graph_sub_title: Optional[str] = None,
    set_y_axis_min_limit_to_0: bool = False,
):
    fig, ax = plt.subplots(figsize=FIGURES_SIZE)
    try:
        fig.set_dpi(SAVE_FIGURES_DPI)
        ordered_line_names = []
        for group_name, ordered_epoch_values in group_name_to_ordered_epoch_values.items():
            ax.plot(ordered_epoch_values)
            ordered_line_names.append(group_name)

        ax.set_title(f'{eval_value_display_name} per epoch'.capitalize())
        if graph_sub_title is not None:
            fig.suptitle(graph_sub_title)
        if set_y_axis_min_limit_to_0:
            ax.set_ylim(ymin=0)
        ax.set_ylabel(eval_value_display_name)
        ax.set_xlabel('epoch')
        ax.legend(ordered_line_names)

        fig.savefig(str(output_graph_jpeg_file_path))
    finally:
        plt.close(fig)
