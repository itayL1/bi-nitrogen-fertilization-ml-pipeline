from enum import Enum


class TrainPipelineLogicalSteps(str, Enum):
    model_k_fold_cross_valuation = 'model_k_fold_cross_valuation'
    final_model_training = 'final_model_training'
    final_model_feature_importance_extraction = 'final_model_feature_importance_extraction'
    generate_pipeline_report = 'generate_pipeline_report'
