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
        self._previous_carriage_returned = False

    BASE_FORMAT = "{0}[{2}] " + Color.DEFAULT + "{1} {0}: %(message)s" + Color.DEFAULT
    VERBOSE_FORMAT = "{0} [%(asctime)s] " + Color.DEFAULT + "{1} {0}[%(module)s]: %(message)s" + Color.DEFAULT
    LEVEL_SPACES = 10

    def _compute_format(self, verbose):
        base_format = CustomFormatter.VERBOSE_FORMAT if verbose else CustomFormatter.BASE_FORMAT
        current_time = datetime.datetime.now().strftime("%a %H:%M:%S")

        return {
        logging.DEBUG: base_format.format(
            Color.GREY, colors.color("DEBUG".center(CustomFormatter.LEVEL_SPACES), 'white', bg='grey'), current_time
        ),
        logging.INFO: base_format.format(
            Color.B, colors.color("INFO".center(CustomFormatter.LEVEL_SPACES), 'white', bg='blue'), current_time
        ),
        logging.WARNING: base_format.format(
            Color.O, colors.color("WARNING".center(CustomFormatter.LEVEL_SPACES), 'white', bg='yellow'), current_time
        ),
        logging.ERROR: base_format.format(
            Color.R, colors.color("ERROR".center(CustomFormatter.LEVEL_SPACES), 'white', bg='red'), current_time
        ),
        logging.CRITICAL: base_format.format(
            Color.R, colors.color("CRITICAL".center(CustomFormatter.LEVEL_SPACES), 'white', bg='red'), current_time
        )
    }

    def format(self, record):
        prefix = "\n"
        if self._previous_carriage_returned:
            prefix = "\r"

        if record.levelno == logging.DEBUG and record.msg == "":
            message = ""
        else:
            if record.msg[-1] == '\r':
                self._previous_carriage_returned = True
                record.msg = record.msg[:-1]
            else:
                self._previous_carriage_returned = False

            log_fmt = self._compute_format(self._verbose).get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            message = formatter.format(record)

        # Suffix to write over the previous line when doing inplace writing
        suffix = " " * 10
        return prefix + message + suffix


def init(debug, verbose=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.getLogger().setLevel(level)

    handler = logging.StreamHandler()
    handler.terminator = ''

    formatter = CustomFormatter(verbose)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)


def set_debug_min_level(debug):
    level = logging.DEBUG if debug else logging.INFO
    logging.getLogger().setLevel(level)


def skip_line():
    get().debug("")

def get():
    return logging.getLogger()

