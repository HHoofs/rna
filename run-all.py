"""
rna

Usage:
  run-all.py [--blanks] [--mixture] [--units <number>] [--epochs <number>] [--batch <size>]

Options:
  -h --help           Show this screen.
  --flatten <mode>    Mode that is used to flatten te multiple scans
  --blanks            Include blanks in the data
  --mixture           If provided, the mixture data is included
  --units <number>    Number of units for each conv/dense layer [default: 500]
  --epochs <number>   Number of epochs used for training [default: 1000]
  --batch <size>      Size of each batch during training [default: 32]
"""
from docopt import docopt
from keras import Model
import numpy as np
from keras.callbacks import ReduceLROnPlateau, Callback
from sklearn.metrics import accuracy_score


from ml.generator import generate_data, DataGenerator, EvalGenerator
from ml.model import build_model, compile_model


def create_model(arguments: dict, n_classes: int) -> Model:
    """

    :param arguments: arguments as parsed by docopt
    :return: A compiled keras model
    """
    model = build_model(arguments, n_classes)
    compile_model(model)
    return model


def create_generators(arguments: dict) -> (DataGenerator, EvalGenerator):
    """

    :param arguments: arguments as parsed by docopt
    :return: two DataGenerators, the first containing the train data, the second containing the test data
    """
    x_train, y_train, x_test, y_test, label_encoder = generate_data(include_blanks=arguments["--blanks"],
                                                                    include_mixtures=arguments["--mixture"])
    # ex
    batch_size = int(arguments["--batch"])
    # return DataGenerator(x_train, y_train, encoder=label_encoder,
    #                      batch_size=batch_size, batches_per_epoch=250), \
    #        EvalGenerator(x_train, y_test, encoder=label_encoder)
    return EvalGenerator(x_train, y_train, encoder=label_encoder, shuffle=True,
                         batch_size=batch_size), \
           EvalGenerator(x_train, y_test, encoder=label_encoder)


def main(arguments: dict) -> None:
    """

    :param arguments:
    """
    train_gen, validation_gen = create_generators(arguments)
    model = create_model(arguments, n_classes=train_gen.n_classes)
    print(model.summary())
    model.fit_generator(train_gen, epochs=int(arguments["--epochs"]), validation_data=validation_gen,
                        callbacks=[CustomMetric(validation_gen)])
    preds = model.predict_generator(validation_gen)
    preds_t = model.predict_generator(train_gen)
    preds, preds_t


class CustomMetric(Callback):
    def __init__(self, generator: EvalGenerator):
        self.gen = generator
        self.steps = len(generator)
        self.threshold = .5

    def on_train_begin(self, logs={}):
        self.acc = []
        self.y_true = np.zeros((len(self.gen.indexes), self.gen.n_classes))
        y_true = [self.gen.encoder.transform(class_name[1].split("+")) for class_name in self.gen.indexes]
        for i, y in enumerate(y_true):
            for idx_y in y:
                self.y_true[i, idx_y] = 1


    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_generator(self.gen)
        print(accuracy_score(self.y_true, y_pred > self.threshold))
        # print(y_pred)

if __name__ == '__main__':
    arguments = docopt(__doc__, version='rna 0.0')
    main(arguments)
