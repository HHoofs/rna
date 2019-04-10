"""
rna

Usage:
  run-all.py [--blanks] [--mixture] [--augment] [--features <n>] [--units <n>] [--epochs <n>] [--batch <s>]

Options:
  -h --help            Show this screen.
  --flatten <mode>     Mode that is used to flatten te multiple scans
  --blanks             Include blanks in the data
  --mixture            If provided, the mixture data is included
  --features <n>       Number of features used for each sample [default: 19]
  --units <n>          Number of units for each conv/dense layer [default: 100]
  --epochs <n>         Number of epochs used for training [default: 50]
  --batch <s>          Size of each batch during training [default: 16]
"""
from typing import Tuple

from docopt import docopt
from tensorflow.python.keras import Model

from ml.generator import generate_data, DataGenerator, EvalGenerator
from ml.model import build_model, compile_model, create_callbacks


def create_model(arguments: dict, n_classes: int) -> Model:
    """
    Create keras/tf model based on the number of classes, features and the the number of units in the model

    :param arguments: arguments as parsed by docopt (including `--units` and `--features`)
    :param n_classes: number of classes in the output layer
    :return: A compiled keras model
    """
    # build model
    model = build_model(int(arguments['--units']), n_classes, int(arguments['--features']))
    # compile model
    compile_model(model)

    return model


def create_generators(arguments: dict) -> Tuple[DataGenerator, EvalGenerator]:
    """
    Read in data and create two generators (one for training and one for evaluation/testing)

    :param arguments: arguments as parsed by docopt
    :return: two DataGenerators, the first containing the train data, the second containing the test data
    """
    # generate data and split into train and test, and return the label encoder for the purpose of
    # converting the output (y) from string to a numeric value
    x_train, y_train, x_test, y_test, label_encoder = generate_data(include_blanks=arguments["--blanks"],
                                                                    include_mixtures=arguments["--mixture"])

    # init train generator
    sampling = {"single": 1,
                "mixture": 1 if arguments["--mixture"] else 0,
                "augment": 1 if arguments["--augment"] else 0}

    print(sampling)

    train_generator = DataGenerator(x_train, y_train, encoder=label_encoder,
                                    n_features=int(arguments["--features"]), sampling=sampling,
                                    batch_size=int(arguments["--batch"]), batches_per_epoch=len(x_train))

    # init eval generator
    augmented_samples = len(x_test)//2 if arguments["--augment"] else None

    eval_generator = EvalGenerator(x_test, y_test, encoder=label_encoder,
                                   augmented_samples=augmented_samples, n_features=int(arguments["--features"]))

    return train_generator, eval_generator


def main(arguments: dict) -> None:
    """
    main, compiling and running the model

    :param arguments: arguments as parsed by docopt
    """
    # create train and validation genrators
    train_gen, validation_gen = create_generators(arguments)
    # create model
    model = create_model(arguments=arguments, n_classes=train_gen.n_classes)
    # print model
    print(model.summary())
    # create callbacks
    callbacks = create_callbacks(int(arguments['--batch']), validation_gen)
    # fit model
    model.fit_generator(train_gen, epochs=int(arguments["--epochs"]), validation_data=validation_gen,
                        callbacks=callbacks, verbose=1, shuffle=False)


if __name__ == '__main__':
    arguments = docopt(__doc__, version='rna 0.1')
    main(arguments)
