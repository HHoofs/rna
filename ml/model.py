from keras import Input, Model
from keras.layers import Flatten, Dense, Dropout, Conv2D, AveragePooling2D, MaxPooling2D


def build_model(arguments: dict, n_classes: int) -> Model:
    """

    :param arguments
    :param flatten_opt:
    :param units:
    :param n_classes
    :return:
    """
    # extract int for the number of units (as docopt will only give you strings
    units = int(arguments["--units"])

    x = Input(shape=(4, 19, 1))
    if arguments["--flatten"] == "conv":
        flatten = Conv2D(filters=units//2, kernel_size=(4, 1), strides=1, padding="valid", activation="relu")(x)
        flatten = Flatten()(flatten)
        flatten = Dense(units=19, activation="relu")(flatten)
    elif arguments["--flatten"] == "avg":
        flatten = AveragePooling2D(pool_size=(4, 1), padding="valid")(x)
        flatten = Flatten()(flatten)
    elif arguments["--flatten"] == "max":
        flatten = MaxPooling2D(pool_size=(4, 1), padding="valid")(x)
        flatten = Flatten()(flatten)

    cnn = Dense(units, activation="relu")(flatten)
    cnn = Dropout(rate=.25)(cnn)
    cnn = Dense(units, activation="relu")(cnn)

    y = Dense(n_classes, activation="sigmoid")(cnn)

    model = Model(inputs=x, outputs=y)

    return model


def compile_model(model: Model, optimizer: str = "adam", loss: str = "binary_crossentropy"):
    """

    :param model:
    :param optimizer:
    :param loss:
    :return:
    """
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
