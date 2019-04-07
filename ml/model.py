from keras import Input, Model
from keras.layers import Flatten, Dense, Dropout, Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization, K
from sklearn.metrics import accuracy_score
import tensorflow as tf



def build_model(arguments: dict, n_classes: int) -> Model:
    """

    :param arguments
    :param flatten_opt:
    :param units:
    :param n_classes
    :return:
    """
    # extract int for the number of units (as docopt will only give you strings
    drop = 0.
    units = int(arguments["--units"])

    x = Input(shape=(19, 1))
    cnn_input = Flatten()(x)
    cnn = Dense(units, activation="sigmoid")(cnn_input)
    cnn = Dropout(rate=drop)(cnn)
    cnn = Dense(units, activation="sigmoid")(cnn)
    cnn = Dropout(rate=drop)(cnn)
    cnn = Dense(units, activation="sigmoid")(cnn)

    y = Dense(n_classes, activation="sigmoid")(cnn)

    model = Model(inputs=x, outputs=y)

    return model


def compile_model(model: Model, optimizer: str = "sgd", loss: str = "binary_crossentropy"):
    """

    :param model:
    :param optimizer:
    :param loss:
    :return:
    """
    model.compile(optimizer=optimizer, loss=loss)
