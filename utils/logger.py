# coding: utf-8
# @email: enoche.chow@gmail.com

"""
###############################
"""
import logging

import coloredlogs


def init_logger(config):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
    """

    log_file = config['log_file_name']

    file_fmt = "%(asctime)-15s %(levelname)s %(message)s"
    file_date_fmt = "%a %d %b %Y %H:%M:%S"
    file_formatter = logging.Formatter(file_fmt, file_date_fmt)

    sfmt = u"%(asctime)-15s %(levelname)s %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = logging.Formatter(sfmt, sdatefmt)
    if config['state'] is None or config['state'].lower() == 'info':
        level = logging.INFO
    elif config['state'].lower() == 'debug':
        level = logging.DEBUG
    elif config['state'].lower() == 'error':
        level = logging.ERROR
    elif config['state'].lower() == 'warning':
        level = logging.WARNING
    elif config['state'].lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    # comment following 3 lines and handlers = [sh, fh] to cancel file dump.
    fh = logging.FileHandler(log_file, 'w', 'utf-8')
    fh.setLevel(level)
    fh.setFormatter(file_formatter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(
        level=level,
        # handlers=[sh]
        handlers=[sh, fh]
    )

    coloredlogs.install(level=level, fmt=sfmt, datefmt=sdatefmt)
