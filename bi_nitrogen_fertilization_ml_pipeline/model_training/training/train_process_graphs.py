from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt


def plot_evaluation_value_per_training_epoch_graph(
    group_name_to_ordered_epoch_values: dict[str, list[float]],
    metric_display_name: str,
    output_graph_jpeg_file_path: Path,
    graph_sub_title: Optional[str] = None,
    set_y_axis_min_limit_to_0: bool = False,
):
    plt.figure(figsize=(12, 5))
    plt.rcParams['savefig.dpi'] = 300
    try:
        ordered_line_names = []
        for group_name, ordered_epoch_values in group_name_to_ordered_epoch_values.items():
            plt.plot(ordered_epoch_values)
            ordered_line_names.append(ordered_line_names)

        plt.title(f'{metric_display_name} per epoch'.capitalize())
        plt.suptitle(graph_sub_title)
        if set_y_axis_min_limit_to_0:
            plt.ylim(ymin=0)
        plt.ylabel(metric_display_name)
        plt.xlabel('epoch')
        plt.legend(ordered_line_names)

        plt.savefig(str(output_graph_jpeg_file_path))
    finally:
        plt.close()
