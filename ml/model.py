import os
import time
import numpy as np

import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import Tensor
from tensorflow.contrib.labeled_tensor.python.ops.core import Scalar
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import TensorBoard, Callback
from tensorflow.python.keras.layers import Flatten, Dense, Dropout

from ml.generator import EvalGenerator


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
    x = Input(shape=(n_features, ))
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
    model.compile(optimizer=optimizer, loss=loss, metrics=[_accuracy_em])


def create_callbacks(batch_size: int, generator: EvalGenerator) -> list:
    """
    create callbacks to use in model.fit

    :param batch_size: batch size used for training the model
    :param generator: data generator
    :return: a list of callbacks
    """
    # create path for log dir
    log_dir = os.path.join('./logs', str(time.time()).replace('.', ''))
    # create log dir path
    os.makedirs(log_dir)
    # create callbacks
    callbacks = [TensorBoard(log_dir=log_dir, batch_size=batch_size),
                 MetricsPerType(generator)]

    return callbacks


def _accuracy_exact_match(y_true: Tensor, y_pred: Tensor, threshold: float = .5) -> Scalar:
    """
    Custom keras metric that mirrors the sklearn.metrics.accuracy_score, that is only samples that have the correct
    labels for each class are scored as 1. If not the sample is scored as 0.

    From: https://stackoverflow.com/questions/46799261/how-to-create-an-exact-match-eval-metric-op-for-tensorflow

    :param y_true: Tensor with the the true labels
    :param y_pred: Tensor with the predicted labels
    :param threshold: Threshold  used to classify a prediction as 1/0
    :return: float that represents the accuracy
    """
    # check if prediction are above threshold
    predictions = tf.to_float(tf.greater_equal(y_pred, threshold))
    # check if predictions match ground truth
    pred_match = tf.equal(predictions, tf.round(y_true))
    # reduce to mean
    exact_match = tf.reduce_min(tf.to_float(pred_match), axis=1)

    return exact_match


def _accuracy_em(*args):
    """
    wrapper for _accuracy_exact_match
    :param args: *
    :return: float that represents the accuracy
    """
    return tf.reduce_mean(_accuracy_exact_match(*args))


class MetricsPerType(Callback):
    def __init__(self, eval_generator: EvalGenerator, threshold: float = .5):
        object.__init__(self)
        self.eval_generator = eval_generator
        self.threshold = threshold
        self.y_pred = {}
        self.y_true = {}

    def on_train_end(self, logs={}):
        for sample_group, sample_types, index, in self.eval_generator.indexes:
            samples = self.eval_generator.__getattribute__(sample_group)[sample_types][index]

            fin_sample = np.mean(samples, 0) if len(samples.shape) == 2 else samples

            y = np.zeros(self.eval_generator.n_classes)
            # Store class
            if sample_types:
                if sample_group == 'single':
                    sample_types = [sample_types]
                else:
                    sample_types = sample_types.split("+")
                for sample_type_idx in self.eval_generator.encoder.transform(sample_types):
                    y[sample_type_idx] = 1

            y_pred = self.model.predict(np.expand_dims(fin_sample, 0))

            if sample_group not in self.y_pred:
                self.y_pred[sample_group] = [y_pred]
                self.y_true[sample_group] = [y]
            else:
                self.y_pred[sample_group].append(y_pred)
                self.y_true[sample_group].append(y)

        for i, sample_group in enumerate(self.y_true.keys()):
            y_true_mat = np.array(self.y_true[sample_group])
            y_pred_mat = np.squeeze(np.array(self.y_pred[sample_group]) >= self.threshold)
            acc = accuracy_score(y_true_mat, y_pred_mat)
            f1 = f1_score(y_true_mat, y_pred_mat, average='samples')
            print(f'accuracy {sample_group}: {acc}')
            print(f'f1-score {sample_group}: {f1}')

            if i == 0:
                y_true_tot, y_pred_tot = y_true_mat, y_pred_mat
            else:
                y_true_tot = np.append(y_true_tot, y_true_mat, 0)
                y_pred_tot = np.append(y_pred_tot, y_pred_mat, 0)

        acc_tot = accuracy_score(y_true_tot, y_pred_tot)
        f1_tot = f1_score(y_true_tot, y_pred_tot, average='samples')
        print(f'accuracy total: {acc_tot}')
        print(f'f1-score total: {f1_tot}')
