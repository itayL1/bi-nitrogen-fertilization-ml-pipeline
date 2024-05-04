# An example for feature config with all the supported feature settings
example_features_config = {
    # the name of the target column (a.k.a true value column)
    'target_column': 'target_column_1',

    # the feature columns that will be used by the model (after their preprocessing) and their settings
    'features': {
        # settings of a numeric feature.
        # these features are passed to the model 'as is'.
        {
            'column': 'column_a',
            'kind': 'numeric',
        },

        # settings of a categorical feature.
        # these features are migrated into one hot encoding representation in the dataset preprocessing phase.
        {
            'column': 'column_b',
            'kind': 'categorical',
        },

        # settings of a categorical feature, that overrides the default behavior of the one hot encoding process.
        # these behavior overrides are completely optional.
        {
            'column': 'column_c',
            'kind': 'categorical',
            'one_hot_encoding_settings': {
                # the minimum frequency of a category for it to be considered significant enough to have
                # its own category column. categories with lower frequency then this threshold will be
                # mapped to the default column of the 'others' category. this value is in percentage, with
                # the default value being 3.
                'min_significant_category_percentage_threshold': 1.5,

                # the maximum number of different categories that are allowed for the current feature, after
                # the mapping of satisfactions categories to the 'others' category was done. if more categories
                # than this threshold are found, an error will be raised. the default value of this threshold
                # is 15.
                'max_allowed_categories_count': 20,

                # the flag defines the behavior of the inference module for the current feature, when
                # it encounters a category that didn't appear in the training set of the model. if this
                # flag is turned off, an error will be raised. if this flag is turned on, these unknown
                # categories will be mapped to the 'others' category instead. the default value of this
                # flag is False.
                'allow_unknown_categories_during_inference': True,
            },
        },
    }
}
