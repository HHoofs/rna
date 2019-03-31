"""
rna

Usage:
  run-all.py --flatten <mode> [--units <number>] [--epochs <number>]

Options:
  -h --help           Show this screen.
  --flatten <mode>    Mode that is used to flatten te multiple scans
  --units <number>    Number of units for each conv/dense layer [default: 8]
  --epochs <number>   Number of epochs used for training [default: 1]
"""
from docopt import docopt
from keras import Model

from ml.model import build_model, compile_model


def create_model(arguments: dict) -> Model:
    """

    :param arguments:
    :return:
    """
    model = build_model(arguments)
    compile_model(model)
    # TODO fit_model(model)
    return model


if __name__ == '__main__':
    arguments = docopt(__doc__, version='rna 0.0')
    model = create_model(arguments)
    print(model.summary())
