from keras.models import Model
from keras.layers import Input, Dense, Dropout, ReLU, Normalization
from keras.initializers import GlorotUniform, Zeros


def init_baseline_model(input_layer_size: int) -> Model:
    input_layer = Input(shape=(input_layer_size,))

    # Input normalization layer
    input_normalization_layer = Normalization(axis=-1)(input_layer)

    # Hidden layer 1
    dense_layer_1 = Dense(units=512, activation='linear', use_bias=True,
                          kernel_initializer=GlorotUniform(seed=None),
                          bias_initializer=Zeros())(input_normalization_layer)
    relu_layer_1 = ReLU()(dense_layer_1)

    # Hidden layer 2
    dense_layer_2 = Dense(units=1024, activation='linear', use_bias=True,
                          kernel_initializer=GlorotUniform(seed=None),
                          bias_initializer=Zeros())(relu_layer_1)
    relu_layer_2 = ReLU()(dense_layer_2)

    # Hidden layer 3
    dense_layer_3 = Dense(units=1024, activation='linear', use_bias=True,
                          kernel_initializer=GlorotUniform(seed=None),
                          bias_initializer=Zeros())(relu_layer_2)
    relu_layer_3 = ReLU()(dense_layer_3)

    # Dropout layer
    dropout_layer = Dropout(rate=0.5)(relu_layer_3)

    # Output layer
    output_layer = Dense(units=1, activation='linear', use_bias=True,
                         kernel_initializer=GlorotUniform(seed=None),
                         bias_initializer=Zeros())(dropout_layer)

    model = Model(
        inputs=input_layer,
        outputs=output_layer,
    )
    return model
