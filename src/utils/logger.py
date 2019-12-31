import datetime
import logging

import colors

class Color:
    DEFAULT = '\033[0m'  # white (normal)
    W = '\033[30m'  # white
    R = '\033[31m'  # red
    GREE = '\033[32m'  # green
    O = '\033[33m'  # orange
    B = '\033[34m'  # blue
    P = '\033[35m'  # purple
    C = '\033[36m'  # cyan
    GREY = '\033[90m'  # grey

class CustomFormatter(logging.Formatter):
    """
    Logging Formatter to add colors
    """

    def __init__(self, verbose):
        super().__init__()
        self._verbose = verbose

    BASE_FORMAT = "{0}[" + datetime.datetime.now().strftime("%a %H:%M:%S") + "] " + \
                  Color.DEFAULT + "{1} {0}: %(message)s" + Color.DEFAULT
    VERBOSE_FORMAT = "{0} [%(asctime)s] " + Color.DEFAULT + "{1} {0}[%(module)s]: %(message)s" + Color.DEFAULT
    LEVEL_SPACES = 10

    def _compute_format(self, verbose):
        base_format = CustomFormatter.VERBOSE_FORMAT if verbose else CustomFormatter.BASE_FORMAT
        return {
        logging.DEBUG: base_format.format(
            Color.GREY, colors.color("DEBUG".center(CustomFormatter.LEVEL_SPACES), 'black', bg='grey')
        ),
        logging.INFO: base_format.format(
            Color.B, colors.color("INFO".center(CustomFormatter.LEVEL_SPACES), 'black', bg='blue')
        ),
        logging.WARNING: base_format.format(
            Color.O, colors.color("WARNING".center(CustomFormatter.LEVEL_SPACES), 'black', bg='yellow')
        ),
        logging.ERROR: base_format.format(
            Color.R, colors.color("ERROR".center(CustomFormatter.LEVEL_SPACES), 'black', bg='red')
        ),
        logging.CRITICAL: base_format.format(
            Color.R, colors.color("CRITICAL".center(CustomFormatter.LEVEL_SPACES), 'black', bg='red')
        )
    }

    def format(self, record):
        if record.levelno == logging.DEBUG and record.msg == '':
            return ""
        log_fmt = self._compute_format(self._verbose).get(record.levelno)
        formatter = logging.Formatter(log_fmt)

        return formatter.format(record)



def init(debug, verbose=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.getLogger().setLevel(level)

    handler = logging.StreamHandler()

    formatter = CustomFormatter(verbose)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

def skip_line():
    logging.getLogger().debug("")

def get():
    return logging.getLogger()

