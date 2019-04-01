"""
rna

Usage:
  run-all.py --flatten <mode> [--blanks] [--mixture] [--units <number>] [--epochs <number>] [--batch <size>]

Options:
  -h --help           Show this screen.
  --flatten <mode>    Mode that is used to flatten te multiple scans
  --blanks            If provided, the blanks are included in the data (and as category)
  --mixture           If provided, the mixture data is included
  --units <number>    Number of units for each conv/dense layer [default: 16]
  --epochs <number>   Number of epochs used for training [default: 1]
  --batch <size>      Size of each batch during training [default: 8]
"""
from docopt import docopt
from keras import Model
from sklearn.preprocessing import LabelEncoder

from ml.generator import generate_data, DataGenerator
from ml.model import build_model, compile_model


def create_model(arguments: dict, n_classes: int) -> Model:
    """

    :param arguments: arguments as parsed by docopt
    :return: A compiled keras model
    """
    model = build_model(arguments, n_classes)
    compile_model(model)
    return model


def create_generators(arguments: dict) -> (DataGenerator, DataGenerator, LabelEncoder):
    """

    :param arguments: arguments as parsed by docopt
    :return: two DataGenerators, the first containing the train data, the second containing the test data
    """
    x_train, y_train, x_test, y_test, label_encoder = generate_data(include_blanks=arguments["--blanks"],
                                                                    include_mixtures=arguments["--mixture"])
    # ex
    batch_size = int(arguments["--batch"])
    n_classes = len(label_encoder.classes_)
    return DataGenerator(x_train, y_train, batch_size=batch_size, n_classes=n_classes), \
           DataGenerator(x_train, y_test, batch_size=1, n_classes=n_classes), \
           label_encoder


def main(arguments: dict):
    """

    :param arguments:
    """
    train_gen, validation_gen, encoder = create_generators(arguments)
    model = create_model(arguments, n_classes=len(encoder.classes_))
    print(model.summary())
    model.fit_generator(train_gen, epochs=int(arguments["--epochs"]), validation_data=validation_gen)


if __name__ == '__main__':
    arguments = docopt(__doc__, version='rna 0.0')
    main(arguments)
