from rich.progress import Progress

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_logical_steps import \
    TrainPipelineLogicalSteps


class TrainPipelineMainProgressBarManager:
    PROGRESS_BAR_COLOR_PREFIX = '[red]'

    def __init__(
        self,
        rich_progress: Progress,
        first_pipeline_step: TrainPipelineLogicalSteps,
    ):
        self._used_steps = []
        self._rich_progress = rich_progress
        self._train_pipeline_progress_task_id =\
            self._start_train_pipeline_progress_task(rich_progress, first_pipeline_step)

    def move_to_next_step(self, next_pipeline_step: TrainPipelineLogicalSteps):
        self._validate_step_not_already_used(next_pipeline_step)
        next_step_description = self._progress_bar_description_for_pipeline_step(next_pipeline_step)
        self._rich_progress.update(
            self._train_pipeline_progress_task_id,
            advance=1,
            description=next_step_description
        )

    def _start_train_pipeline_progress_task(
        self,
        rich_progress: Progress,
        first_pipeline_step: TrainPipelineLogicalSteps,
    ):
        self._used_steps.append(first_pipeline_step)
        total_steps_count = len(TrainPipelineLogicalSteps)

        initial_step_description =\
            self._progress_bar_description_for_pipeline_step(first_pipeline_step)
        train_pipeline_progress_task_id = rich_progress.add_task(
            description=initial_step_description, start=True, total=total_steps_count)
        return train_pipeline_progress_task_id

    def _validate_step_not_already_used(self, pipeline_step: TrainPipelineLogicalSteps):
        assert pipeline_step not in self._used_steps,\
            f"the provided step '{pipeline_step}' was already used for this progress " \
            f"bar. using a step more then once is not allowed."
        self._used_steps.append(pipeline_step)

    @classmethod
    def _progress_bar_description_for_pipeline_step(
        cls, pipeline_step: TrainPipelineLogicalSteps,
    ) -> str:
        if pipeline_step == TrainPipelineLogicalSteps.preprocess_train_dataset:
            step_description = 'preprocessing the train dataset'
        elif pipeline_step == TrainPipelineLogicalSteps.model_k_fold_cross_valuation:
            step_description = 'evaluating the model (K-fold)'
        elif pipeline_step == TrainPipelineLogicalSteps.final_model_training:
            step_description = 'training the final model'
        elif pipeline_step == TrainPipelineLogicalSteps.generate_pipeline_report:
            step_description = 'generating the train pipeline report'
        else:
            raise NotImplementedError(f"unknown pipeline step - {pipeline_step}")

        final_description = f'{cls.PROGRESS_BAR_COLOR_PREFIX}Pipeline progress \[{step_description}]'
        return final_description

#
# # @contextmanager
# def train_pipeline_main_progress_bar(
#     rich_progress: Progress,
# ) -> TrainPipelineMainProgressBarManager:
#     train_pipeline_main_progress_task = rich_progress.add_task(
#         '', start=True, total=total_steps_count)

