from typing import Collection

import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core import defaults, consts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeatureKinds, FeatureSettings
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import OneHotEncodingMetadata, \
    OneHotEncodedFeatures
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import \
    OtherCategoryAggregationDetails, CategoricalFeatureEncodingDetails, CategoricalFeaturesEncodingMethod
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_session_context import TrainSessionContext
from bi_nitrogen_fertilization_ml_pipeline.core.pipeline_report.display_utils import to_displayable_percentage, \
    to_displayable_percentage_distribution
from bi_nitrogen_fertilization_ml_pipeline.core.utils.collection_utils import filter_dict, set_new_dict_entry
from bi_nitrogen_fertilization_ml_pipeline.core.utils.object_utils import nested_getattr


def fit_categorical_features_one_hot_encoding(
    train_dataset_df: pd.DataFrame,
    session_context: TrainSessionContext,
) -> None:
    features_config = session_context.artifacts.features_config

    categorical_features = _filter_features_by_kind(features_config.features, FeatureKinds.categorical)
    for feature_settings in categorical_features:
        _fit_one_hot_encoding_for_feature(train_dataset_df, feature_settings, session_context)


def transform_categorical_features_one_hot_encoding(
    dataset_df: pd.DataFrame,
    one_hot_encoded_features: OneHotEncodedFeatures,
    for_inference: bool,
) -> None:
    for feature_col, one_hot_encoding_metadata in one_hot_encoded_features.items():
        _transform_categorical_feature_one_hot_encoding(
            dataset_df, feature_col, one_hot_encoding_metadata, for_inference,
        )


def _fit_one_hot_encoding_for_feature(
    train_dataset_df: pd.DataFrame,
    feature_settings: FeatureSettings,
    session_context: TrainSessionContext,
) -> None:
    feature_col = _validate_and_extract_input_col(train_dataset_df, feature_settings)

    original_raw_categories = set(feature_col.unique())
    _validate_original_categories(feature_settings, original_raw_categories)

    final_feature_col, insignificant_categories, report_other_category_aggregation_details =\
        _map_insignificant_categories_to_other_category(feature_col, feature_settings)

    final_categories = set(final_feature_col.unique())
    categories_ordered_by_relative_offset = sorted(final_categories)

    final_categories_perc_distribution = _get_categories_perc_distribution(final_feature_col)
    report_encoding_details = CategoricalFeatureEncodingDetails(
        encoding_method=CategoricalFeaturesEncodingMethod.one_hot_encoding,
        categories_distribution=to_displayable_percentage_distribution(final_categories_perc_distribution),
        other_category_aggregation=report_other_category_aggregation_details,
    )
    allow_unknown_categories_during_inference = nested_getattr(
        feature_settings,
        'one_hot_encoding_settings.allow_unknown_categories_during_inference',
        defaults.ALLOW_UNKNOWN_CATEGORIES_DURING_INFERENCE,
    )
    train_artifact_encoding_metadata = OneHotEncodingMetadata(
        original_raw_categories=sorted(original_raw_categories),
        categories_ordered_by_relative_offset=categories_ordered_by_relative_offset,
        categories_mapped_to_other=sorted(insignificant_categories),
        allow_unknown_categories_during_inference=allow_unknown_categories_during_inference,
    )

    set_new_dict_entry(
        session_context.artifacts.dataset_preprocessing.one_hot_encoded_features,
        key=feature_settings.column, val=train_artifact_encoding_metadata,
    )
    set_new_dict_entry(
        session_context.pipeline_report.dataset_preprocessing.categorical_features_encoding_details,
        key=feature_settings.column, val=report_encoding_details,
    )


def _validate_and_extract_input_col(
    train_dataset_df: pd.DataFrame,
    feature_settings: FeatureSettings,
) -> pd.Series:
    assert feature_settings.kind == FeatureKinds.categorical, \
        'only categorical features are supported'
    assert feature_settings.column in train_dataset_df.columns, \
        f"the feature column '{feature_settings.column}' is missing in the train dataset"

    feature_col = train_dataset_df[feature_settings.column]
    _validate_column_contain_only_non_empty_strings(feature_col)
    return feature_col


def _validate_original_categories(
    feature_settings: FeatureSettings,
    original_raw_categories: set[str],
) -> None:
    assert len(original_raw_categories) >= 2, \
        f"at least 2 categories must be present for a category feature, but for the " \
        f"column '{feature_settings.column}' only {original_raw_categories} was found"
    for category in original_raw_categories:
        other_categories_in_lower_case = {
            cat.lower() for cat in
            original_raw_categories - {category}
        }
        assert category not in other_categories_in_lower_case, \
            f"for the column '{feature_settings.column}', the category '{category}' was " \
            f"found in multiple forms of upper/lowe cases, which is invalid"


def _map_insignificant_categories_to_other_category(
    feature_col: pd.Series,
    feature_settings: FeatureSettings,
) -> tuple[pd.Series, set[str], OtherCategoryAggregationDetails]:
    min_significant_category_frequency_percentage = nested_getattr(
        feature_settings,
        'one_hot_encoding_settings.min_significant_category_frequency_percentage',
        defaults.MIN_SIGNIFICANT_CATEGORY_FREQUENCY_PERCENTAGE,
    )
    max_allowed_categories_count = nested_getattr(
        feature_settings,
        'one_hot_encoding_settings.max_allowed_categories_count',
        defaults.MAX_ALLOWED_CATEGORIES_COUNT_PER_FEATURE,
    )

    raw_categories_perc_distribution_dict = _get_categories_perc_distribution(feature_col)
    insignificant_categories_perc_distribution = filter_dict(
        raw_categories_perc_distribution_dict,
        lambda cat, percentage: percentage < min_significant_category_frequency_percentage,
    )

    insignificant_categories = set(insignificant_categories_perc_distribution.keys())
    processed_feature_col = _map_categories_to_the_other_category(feature_col, insignificant_categories)
    processed_categories_count = len(processed_feature_col.unique())
    assert processed_categories_count <= max_allowed_categories_count, \
        f"the maximum allowed number of categories for this categorical feature " \
        f"is {max_allowed_categories_count}, but {processed_categories_count} " \
        f"categories were found in practice after the preprocessing phase"

    report_other_category_aggregation_details = OtherCategoryAggregationDetails(
        total_percentage=to_displayable_percentage(sum(insignificant_categories_perc_distribution.values())),
        min_significant_category=to_displayable_percentage(min_significant_category_frequency_percentage),
        aggregated_categories_distribution=to_displayable_percentage_distribution(
            insignificant_categories_perc_distribution),
    )
    return (
        processed_feature_col,
        insignificant_categories,
        report_other_category_aggregation_details,
    )


def _map_categories_to_the_other_category(
    feature_col: pd.Series,
    categories_to_map: Collection[str],
) -> pd.Series:
    return feature_col.apply(
        lambda cat: consts.ONE_HOT_OTHER_CATEGORY if cat in categories_to_map else cat
    )


def _transform_categorical_feature_one_hot_encoding(
    dataset_df: pd.DataFrame,
    column: str,
    feature_encoding_metadata: OneHotEncodingMetadata,
    for_inference: bool,
):
    assert column in dataset_df, \
        f"the feature column '{column}' is missing in the train dataset"
    feature_col = dataset_df[column]
    _validate_column_contain_only_non_empty_strings(feature_col)

    actual_feature_categories = set(feature_col.unique())
    unknown_feature_col_categories =\
        actual_feature_categories - set(feature_encoding_metadata.get_known_categories())
    if any(unknown_feature_col_categories):
        if for_inference and feature_encoding_metadata.allow_unknown_categories_during_inference:
            feature_col = _map_categories_to_the_other_category(feature_col, unknown_feature_col_categories)
        else:
            raise AssertionError(
                f"in the provided dataset, the feature column '{column}' contains categories that "
                f"were not included during the model training phase. this isn't allowed in the "
                f"current setup of this feature. the unknown categories are - {unknown_feature_col_categories}"
            )

    adjusted_feature_col = _map_categories_to_the_other_category(
        feature_col, feature_encoding_metadata.categories_mapped_to_other)

    categories_ordered_by_relative_offset = feature_encoding_metadata.categories_ordered_by_relative_offset
    for category in (*categories_ordered_by_relative_offset, consts.ONE_HOT_OTHER_CATEGORY):
        category_col_name = f'{column}_OHE__{category}'
        dataset_df[category_col_name] = (adjusted_feature_col == category).astype(int)
    dataset_df.drop(columns=[column], inplace=True)


def _validate_column_contain_only_non_empty_strings(feature_col: pd.Series) -> None:
    assert feature_col.replace('', None).notna().all(), \
        'the category column must not contain empty values'
    assert all(isinstance(val, str) for val in feature_col), \
        f"all the values in the category column '{feature_col.name}' must be strings"


def _get_categories_perc_distribution(cat_feature_col_no_na: pd.Series) -> dict[str, float]:
    return dict(cat_feature_col_no_na.value_counts(normalize=True) * 100)


def _filter_features_by_kind(
    features: Collection[FeatureSettings],
    kind: FeatureKinds,
) -> tuple[FeatureSettings]:
    return tuple(
        feature for feature in features
        if feature.kind == kind
    )
