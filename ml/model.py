from keras import Input, Model
from keras.layers import Flatten, Dense, Dropout, Conv2D, AveragePooling2D, MaxPooling2D


def build_model(arguments: dict) -> Model:
    """

    :param arguments
    :param flatten_opt:
    :param units:
    :return:
    """
    # extract int for the number of units (as docopt will only give you strings
    units = int(arguments["--units"])

    x = Input(shape=(19, 4, 1))
    if arguments["--flatten"] == "conv":
        flatten = Conv2D(filters=units, kernel_size=(1, 4), strides=1, padding="valid", activation="relu")(x)
        flatten = Flatten()(flatten)
        flatten = Dense(units=19, activation="relu")(flatten)
    elif arguments["--flatten"] == "avg":
        flatten = AveragePooling2D(pool_size=(1, 4), padding="valid")(x)
    elif arguments["--flatten"] == "max":
        flatten = MaxPooling2D(pool_size=(1, 4), padding="valid")(x)

    cnn = Dense(units, activation="relu")(flatten)
    cnn = Dropout(rate=.25)(cnn)
    cnn = Dense(units, activation="relu")(cnn)

    y = Dense(9, activation="sigmoid")(cnn)

    model = Model(inputs=x, outputs=y)

    return model


def compile_model(model: Model, optimizer: str = "adam", loss: str = "binary_crossentropy"):
    """

    :param model:
    :param optimizer:
    :param loss:
    :return:
    """
    model.compile(optimizer=optimizer, loss=loss)


def fit_model(model: Model, genarator, epochs: int = 1):
    """

    :param model:
    :param genarator:
    :param epochs:
    :return:
    """
    model.fit_generator(generator=genarator, epochs=epochs)
