from enum import Enum


class TrainPipelineLogicalSteps(str, Enum):
    preprocess_train_dataset = 'preprocess_train_dataset'
    model_k_fold_cross_valuation = 'model_k_fold_cross_valuation'
    final_model_training = 'final_model_training'
    generate_pipeline_report = 'generate_pipeline_report'
