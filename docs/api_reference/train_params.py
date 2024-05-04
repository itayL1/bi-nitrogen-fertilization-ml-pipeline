import keras.models
import keras.layers
import keras.optimizers.legacy


def model_builder_example(input_layer_size: int) -> keras.models.Model:
    return keras.models.Sequential([
        keras.layers.Input(shape=(input_layer_size,)),
        keras.layers.Normalization(axis=-1),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1_024, activation='relu'),
        keras.layers.Dense(1),
    ])


# An example for train params with all the supported options
example_train_params = dict(
    # A function that receives the size of the input layer of the model as a parameter,
    # and returns a new instance of a Keras model. This function is used several times
    # during the training flow - in order to initialize a new model for each evaluation fold,
    # and in order to initialize the final model. Currently only Keras models are supported.
    model_builder=model_builder_example,

    # The loss function that will be used for the model training process.
    # Currently, the supported loss functions are 'mse', 'rmse', 'mae' and 'huber_loss', with
    # the default being mse.
    loss_function='mse',

    # The evaluation function that will be used to evaluate the model in multiple stages of
    # the training process.
    # Currently, the supported loss functions are 'mse', 'rmse', 'mae' and , with the default
    # being rmse.
    evaluation_metric='rmse',

    # The number of training epochs that will be set for the model training. If early stopping
    # is toggled on, this value will constitute the maximum number of training epochs.
    epochs_count=25,

    # Defines how the training set will be split into folds during the model evaluation phase.
    # One of these settings need to be used, but not both: by_key_column, by_folds_number.
    evaluation_folds_split=dict(
        # Split the evaluation folds by a key column from the original dataset.
        # Note that this column doesn't have to also be used as a feature.
        by_key_column='year',

        # If this setting would have been used instead, the train dataset would have been split
        # randomly into 5 evaluation folds.
        # by_folds_number=5,
    ),

    # Enables an early stopping strategy for the model training process. Notice that this setting
    # is shared for the training of all the models in the training flow, including the evaluation
    # folds models and the final model.
    # This setting is optional. When it's not used, early stopping is disabled.
    early_stopping=dict(
        # The fraction of the train dataset that will be used a validation set for the
        # early stopping internal evaluation. this value must be between 0 and 1.
        validation_set_fraction_size=0.2,

        # The number of consecutive training epochs that must introduce an improvement in the model
        # performance, otherwise the training of the model will be stopped.
        tolerance_epochs_count=9,
    ),

    # A function that returns a Keras optimizer instance to use for the training process.
    # Notice that this setting is also intended to be used to adjust the optimizer hyper-parameters
    # when needed. For example: lambda: keras.optimizers.legacy.Adam(learning_rate=0.005).
    optimizer_builder=keras.optimizers.legacy.Adam,

    # Use a random seed to ensure the reproducibility of the training process and results. Please be
    # careful when using this value, its considered a bad practice to use the same random seed over
    # time. This setting is completely optional.
    random_seed=42,

    # Disable Keras's default model fitting terminal progress bar, to keep the training pipeline
    # terminal progress reporting clean. The default value is True.
    silent_models_fitting=True,

    # Enables the inclusion of the EDA reports as part of the final model training report. The
    # generation of these 2 reports usually takes ~10 seconds, but it might take longer depending
    # on the input dataset. The default value is True.
    create_dataset_eda_reports=True,
)
