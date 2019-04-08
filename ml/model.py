import os
import time

import tensorflow as tf
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.layers import Flatten, Dense, Dropout
from tensorflow import Tensor


def accuracy_em(y_true: Tensor, y_logits: Tensor, threshold: float = .5) -> float:
    """
    Custom keras metric that mirrors the sklearn.metrics.accuracy_score, that is only samples that have the correct
    labels for each class are scored as 1. If not the sample is scored as 0.

    From: https://stackoverflow.com/questions/46799261/how-to-create-an-exact-match-eval-metric-op-for-tensorflow

    :param y_true: Tensor with the the true labels
    :param y_logits: Tensor with the predicted labels
    :param threshold: Threshold  used to classify a prediction as 1/0
    :return: float that represents the accuracy
    """
    # check if prediction are above threshold
    predictions = tf.to_float(tf.greater_equal(y_logits, threshold))
    # check if predictions match ground truth
    pred_match = tf.equal(predictions, tf.round(y_true))
    # reduce to mean
    exact_match = tf.reduce_min(tf.to_float(pred_match), axis=1)

    return exact_match


def build_model(units: int, n_classes: int, n_features: int) -> Model:
    """
    Builds deep learning model

    :param units: (relative) number of units
    :param n_classes number of classes
    :param n_features: number of features
    :return: a keras model
    """
    # set drop out
    drop = 0.05

    # inout shape
    x = Input(shape=(n_features, 1))
    # flatten input shape (i.e. remove the ,1)
    cnn_input = Flatten()(x)
    # first dense (hidden) layer
    cnn = Dense(units//4, activation="sigmoid")(cnn_input)
    # dropout
    cnn = Dropout(rate=drop)(cnn)
    # second dense (hidden) layer
    cnn = Dense(units, activation="sigmoid")(cnn)

    # output layer (corresponding to the number of classes)
    y = Dense(n_classes, activation="sigmoid")(cnn)

    # define inputs and outputs of the model
    model = Model(inputs=x, outputs=y)

    return model


def compile_model(model: Model, optimizer: str = "adam", loss: str = "binary_crossentropy") -> None:
    """
    compile a keras model using an optimizer and a loss function

    :param model: a keras model
    :param optimizer: a string or optimizer class that is supported by keras
    :param loss: a string or loss class that is supported by keras
    """
    model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy_em])


def create_callbacks(batch_size: int) -> list:
    """
    create callbacks to use in model.fit

    :param batch_size: batch size used for training the model
    :return: a list of callbacks
    """
    # create path for log dir
    log_dir = os.path.join('./logs', str(time.time()).replace('.', ''))
    # create log dir path
    os.makedirs(log_dir)
    # create callbacks
    callbacks = [#ReduceLROnPlateau(monitor='loss'),
                 TensorBoard(log_dir=log_dir, batch_size=batch_size)]

    return callbacks
