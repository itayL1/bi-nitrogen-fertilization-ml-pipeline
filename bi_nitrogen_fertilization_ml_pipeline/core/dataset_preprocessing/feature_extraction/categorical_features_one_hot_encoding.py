import math
from typing import Collection

import pandas as pd

from bi_nitrogen_fertilization_ml_pipeline.core import defaults, consts
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.features_config import FeatureKinds, FeatureSettings
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_artifacts import OneHotEncodingMetadata, \
    OneHotEncodedFeatures
from bi_nitrogen_fertilization_ml_pipeline.core.data_classes.train_pipeline_report import \
    OtherCategoryAggregationDetails, CategoricalFeatureEncodingDetails, CategoricalFeaturesEncodingMethod, \
    FinalCategories, WarningPipelineModules, TrainPipelineReportData
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


def _fit_one_hot_encoding_for_feature(
    train_dataset_df: pd.DataFrame,
    feature_settings: FeatureSettings,
    session_context: TrainSessionContext,
) -> None:
    feature_col = _validate_and_extract_input_col(train_dataset_df, feature_settings)

    original_raw_categories = set(feature_col.unique())
    _validate_original_categories(feature_settings, original_raw_categories)
    (
        min_significant_category_percentage_threshold,
        max_allowed_categories_count,
        allow_unknown_categories_during_inference,
    ) = _extract_feature_one_hot_encoding_settings(feature_settings)

    (
        final_feature_col, insignificant_categories, report_other_category_aggregation_details,
    ) = _map_insignificant_categories_to_other_category(
        feature_col, min_significant_category_percentage_threshold, max_allowed_categories_count,
    )

    final_categories = set(final_feature_col.unique())
    categories_ordered_by_relative_offset = sorted(final_categories)

    final_categories_perc_distribution = _get_categories_perc_distribution(final_feature_col)
    report_encoding_details = CategoricalFeatureEncodingDetails(
        encoding_method=CategoricalFeaturesEncodingMethod.one_hot_encoding,
        final_categories=FinalCategories(
            count=len(final_categories),
            values=sorted(final_categories),
        ),
        categories_frequency=dict(final_feature_col.value_counts()),
        categories_perc_distribution=to_displayable_percentage_distribution(final_categories_perc_distribution),
        other_category_aggregation=report_other_category_aggregation_details,
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

    _add_report_warnings_regarding_encoding_if_needed(
        feature_settings, final_categories, final_categories_perc_distribution,
        max_allowed_categories_count, min_significant_category_percentage_threshold,
        report_encoding_details, session_context.pipeline_report,
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


def _extract_feature_one_hot_encoding_settings(
    feature_settings: FeatureSettings,
) -> tuple[float, int, bool]:
    min_significant_category_percentage_threshold = nested_getattr(
        feature_settings,
        'one_hot_encoding_settings.min_significant_category_percentage_threshold',
        None,
    ) or defaults.MIN_SIGNIFICANT_CATEGORY_PERCENTAGE_THRESHOLD
    max_allowed_categories_count = nested_getattr(
        feature_settings,
        'one_hot_encoding_settings.max_allowed_categories_count',
        None,
    ) or defaults.MAX_ALLOWED_CATEGORIES_COUNT_PER_FEATURE
    allow_unknown_categories_during_inference = nested_getattr(
        feature_settings,
        'one_hot_encoding_settings.allow_unknown_categories_during_inference',
        None,
    ) or defaults.ALLOW_UNKNOWN_CATEGORIES_DURING_INFERENCE

    return (
        min_significant_category_percentage_threshold,
        max_allowed_categories_count,
        allow_unknown_categories_during_inference,
    )


def _map_insignificant_categories_to_other_category(
    feature_col: pd.Series,
    min_significant_category_percentage_threshold: float,
    max_allowed_categories_count: int,
) -> tuple[pd.Series, set[str], OtherCategoryAggregationDetails]:
    raw_categories_perc_distribution_dict = _get_categories_perc_distribution(feature_col)
    insignificant_categories_perc_distribution = filter_dict(
        raw_categories_perc_distribution_dict,
        lambda cat, percentage: percentage < min_significant_category_percentage_threshold,
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
        min_significant_category_percentage_threshold=to_displayable_percentage(min_significant_category_percentage_threshold),
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


def transform_categorical_feature_one_hot_encoding(
    feature_col: pd.Series,
    feature_col_name: str,
    feature_encoding_metadata: OneHotEncodingMetadata,
    for_inference: bool,
) -> pd.DataFrame:
    _validate_column_contain_only_non_empty_strings(feature_col)

    actual_feature_categories = set(feature_col.unique())
    unknown_feature_col_categories =\
        actual_feature_categories - set(feature_encoding_metadata.get_known_categories())
    if any(unknown_feature_col_categories):
        if for_inference and feature_encoding_metadata.allow_unknown_categories_during_inference:
            feature_col = _map_categories_to_the_other_category(feature_col, unknown_feature_col_categories)
        else:
            raise AssertionError(
                f"in the provided dataset, the feature column '{feature_col_name}' contains categories that "
                f"were not included during the model training phase. this isn't allowed in the "
                f"current setup of this feature. the unknown categories are - {unknown_feature_col_categories}"
            )

    adjusted_feature_col = _map_categories_to_the_other_category(
        feature_col, feature_encoding_metadata.categories_mapped_to_other)

    categories_ordered_by_relative_offset = feature_encoding_metadata.categories_ordered_by_relative_offset

    category_col_name_to_series = dict()
    for category in (*categories_ordered_by_relative_offset, consts.ONE_HOT_OTHER_CATEGORY):
        category_col_name = f'{feature_col_name}_OHE__{category}'
        category_col_name_to_series[category_col_name] = (adjusted_feature_col == category).astype(int)

    one_hot_encoded_feature_df = pd.DataFrame(data=category_col_name_to_series)
    return one_hot_encoded_feature_df


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


def _add_report_warnings_regarding_encoding_if_needed(
    feature_settings: FeatureSettings,
    final_categories: set[str],
    final_categories_perc_distribution: dict[str, float],
    max_allowed_categories_count: int,
    min_significant_category_percentage_threshold: float,
    report_encoding_details: CategoricalFeatureEncodingDetails,
    pipeline_report: TrainPipelineReportData,
) -> None:
    categories_count_warning_threshold = math.ceil(max_allowed_categories_count * 2 / 3)
    if report_encoding_details.final_categories.count >= categories_count_warning_threshold:
        pipeline_report.add_warning(
            WarningPipelineModules.dataset_preprocessing,
            f"the number of categories in this feature is {report_encoding_details.final_categories.count}, "
            f"which is relatively high and approaches the limit defined for this feature",
            context={
                'feature_column': feature_settings.column,
                'feature.max_allowed_categories_count': max_allowed_categories_count,
            },
        )

    other_category_frequency_percentage = \
        final_categories_perc_distribution.get(consts.ONE_HOT_OTHER_CATEGORY, 0)
    if other_category_frequency_percentage > 10:
        pipeline_report.add_warning(
            WarningPipelineModules.dataset_preprocessing,
            f"the percentage of the 'others' category in this feature is "
            f"{to_displayable_percentage(other_category_frequency_percentage)}, which is relatively high",
            context={
                'feature_column': feature_settings.column,
            },
        )

    categories_without_other = final_categories - {consts.ONE_HOT_OTHER_CATEGORY}
    if len(categories_without_other) < 2:
        pipeline_report.add_warning(
            WarningPipelineModules.dataset_preprocessing,
            f"there is no diversity in the categories of this features, when counting out the 'others' category. "
            f"consider changing the value of min_significant_category_percentage_threshold defined for this feature.",
            context={
                'feature_column': feature_settings.column,
                'feature.min_significant_category_percentage_threshold': min_significant_category_percentage_threshold,
            },
        )
