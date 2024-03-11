from rich.progress import Progress

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_logical_steps import \
    TrainPipelineLogicalSteps


class TrainPipelineMainProgressBarManager:
    PROGRESS_BAR_COLOR_PREFIX = '[red]'

    def __init__(self, rich_progress: Progress):
        self._used_steps = []
        self._rich_progress = rich_progress
        self._train_pipeline_progress_task_id =\
            self._create_train_pipeline_progress_task(rich_progress)
        self._current_step = None

    def move_to_next_step(self, next_pipeline_step: TrainPipelineLogicalSteps):
        assert next_pipeline_step not in self._used_steps, \
            f"the provided step '{next_pipeline_step}' was already used for this progress " \
            f"bar. using a step more then once is not allowed."
        self._used_steps.append(next_pipeline_step)
        previous_step = self._current_step
        self._current_step = next_pipeline_step

        is_first_step = len(self._used_steps) == 1
        if is_first_step:
            self._rich_progress.start_task(self._train_pipeline_progress_task_id)
        else:
            self._rich_progress.update(self._train_pipeline_progress_task_id, advance=1)

        if previous_step is not None:
            print(self._get_step_finished_display_message(previous_step))

        current_step_description = self._progress_bar_description_for_pipeline_step(next_pipeline_step)
        self._rich_progress.update(
            self._train_pipeline_progress_task_id, description=current_step_description)

    @classmethod
    def _create_train_pipeline_progress_task(cls, rich_progress: Progress):
        total_steps_count = len(TrainPipelineLogicalSteps)
        train_pipeline_progress_task_id = rich_progress.add_task(
            description='', start=False, total=total_steps_count)
        return train_pipeline_progress_task_id

    @classmethod
    def _progress_bar_description_for_pipeline_step(
        cls, pipeline_step: TrainPipelineLogicalSteps,
    ) -> str:
        if pipeline_step == TrainPipelineLogicalSteps.model_k_fold_cross_valuation:
            step_description = 'evaluating the model (K-fold)'
        elif pipeline_step == TrainPipelineLogicalSteps.final_model_training:
            step_description = 'training the final model'
        elif pipeline_step == TrainPipelineLogicalSteps.final_model_feature_importance_extraction:
            step_description = 'extraction the feature importance of the final model'
        elif pipeline_step == TrainPipelineLogicalSteps.generate_pipeline_report:
            step_description = 'generating the train pipeline report'
        else:
            raise NotImplementedError(f"unknown pipeline step - {pipeline_step}")

        final_description = f'{cls.PROGRESS_BAR_COLOR_PREFIX}Pipeline progress \[{step_description}]'
        return final_description

    @classmethod
    def _get_step_finished_display_message(
        cls, pipeline_step: TrainPipelineLogicalSteps,
    ) -> str:
        if pipeline_step == TrainPipelineLogicalSteps.model_k_fold_cross_valuation:
            step_finished_message = 'the model evaluation (K-fold) has finished'
        elif pipeline_step == TrainPipelineLogicalSteps.final_model_training:
            step_finished_message = 'the training process of the final model has finished'
        elif pipeline_step == TrainPipelineLogicalSteps.final_model_feature_importance_extraction:
            step_finished_message = 'the extraction of the feature importance for the final model has finished'
        elif pipeline_step == TrainPipelineLogicalSteps.generate_pipeline_report:
            step_finished_message = 'the train pipeline report has been generated'
        else:
            raise NotImplementedError(f"unknown pipeline step - {pipeline_step}")

        final_description = f'** {step_finished_message.capitalize()} **'
        return final_description
