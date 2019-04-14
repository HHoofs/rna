import logging
import time
from os import makedirs
from os.path import join


def create_logdir() -> str:
    """
    creates directory for logging and returns path as string
    :return: string of the logdir
    """
    # create path for log dir
    log_dir = join('./logs', str(time.time()).replace('.', ''))
    # create log dir path
    makedirs(log_dir)

    return log_dir


def create_logging() -> str:
    """
    setup logging and
    :return: string of the logdir
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('main')
    logger.propagate = False

    logdir = create_logdir()

    logger.addHandler(logging.FileHandler(join(logdir, 'log.log'))), \

    return logdir
