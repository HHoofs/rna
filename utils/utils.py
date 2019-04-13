import time
from os import makedirs
from os.path import join

import yaml
from confidence import Configuration


def store_configuration(config: Configuration, path: str) -> None:
    with open(join(path, 'configuration.yml'), 'w') as outfile:
        yaml.dump(config._source, outfile, default_flow_style=False)


def store_arguments(arguments: dict, path: str) -> None:
    with open(join(path, 'arguments.yml'), 'w') as outfile:
        yaml.dump(arguments, outfile, default_flow_style=False)


def create_logdir():
    # create path for log dir
    log_dir = join('./logs', str(time.time()).replace('.', ''))
    # create log dir path
    makedirs(log_dir)

    return log_dir