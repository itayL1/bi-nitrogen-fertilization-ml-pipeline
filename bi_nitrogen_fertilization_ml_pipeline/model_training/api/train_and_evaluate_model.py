import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import ModelTrainingArtifacts
from bi_nitrogen_fertilization_ml_pipeline.core.dataset_preprocessing import dataset_preprocessing
from bi_nitrogen_fertilization_ml_pipeline.model_training.api.setup_train_session_context import \
    setup_train_session_context
from bi_nitrogen_fertilization_ml_pipeline.model_training.api.user_input import parse_input_features_config, \
    validate_input_train_dataset, parse_input_train_params
from bi_nitrogen_fertilization_ml_pipeline.model_training.evaluation.k_fold_cross_validation import \
    key_based_k_fold_cross_validation


def train_and_evaluate_model(
    raw_train_dataset_df: pd.DataFrame,
    features_config_dict: dict,
    train_params_dict: dict,
):
    features_config = parse_input_features_config(features_config_dict)
    train_params = parse_input_train_params(train_params_dict)
    validate_input_train_dataset(raw_train_dataset_df)

    with setup_train_session_context(features_config, train_params) as session_context:
        preprocessed_train_dataset = dataset_preprocessing.train_dataset_preprocessing(
            raw_train_dataset_df, session_context)

        key_based_k_fold_cross_validation(
            preprocessed_train_dataset,
            session_context)
        # _train_final_model_on_entire_dataset()

        session_context.artifacts.model_training = ModelTrainingArtifacts(
            model_input_order_feature_columns=list(preprocessed_train_dataset.X.columns),
        )
        session_context.artifacts.is_fitted = True

        # todo - add warnings for
        #  * k fold groups not evenly splitted
        #  * too many k fold groups
        #  * feature importance not proportional
        #  * high std in k fold
        #  * final training close to random guess
