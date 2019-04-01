"""
rna

Usage:
  run-all.py --flatten <mode> [--units <number>] [--epochs <number>]

Options:
  -h --help           Show this screen.
  --flatten <mode>    Mode that is used to flatten te multiple scans
  --units <number>    Number of units for each conv/dense layer [default: 16]
  --epochs <number>   Number of epochs used for training [default: 1]
"""
from docopt import docopt
from keras import Model

from ml.generator import generate_data, DataGenerator
from ml.model import build_model, compile_model


def create_model(arguments: dict) -> Model:
    """

    :param arguments: arguments as parsed by docopt
    :return: A compiled keras model
    """
    model = build_model(arguments)
    compile_model(model)
    return model


def create_generators(arguments: dict) -> (DataGenerator, DataGenerator):
    """

    :param arguments: arguments as parsed by docopt
    :return: two DataGenerators, the first containing the train data, the second containing the test data
    """
    x_train, y_train, x_test, y_test, label_encoder = generate_data()
    print(label_encoder.classes_)
    return DataGenerator(x_train, y_train, batch_size=8), DataGenerator(x_train, y_test, batch_size=1)


def main(arguments: dict):
    """

    :param arguments:
    """
    model = create_model(arguments)
    print(model.summary())
    train_gen, validation_gen = create_generators(arguments)
    model.fit_generator(train_gen, epochs=int(arguments["--epochs"]), validation_data=validation_gen)


if __name__ == '__main__':
    arguments = docopt(__doc__, version='rna 0.0')
    main(arguments)
