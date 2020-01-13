import datetime
import logging

# for i in range(255):
#     print('\33[38;5;' + str(i) + "m" + str(i) + "\33[0m", end=" ")
#     print('\33[48;5;' + str(i) + "m" + str(i) + "\33[0m", end=" ")

class Color:
    W = '0'  # white
    R = '124'  # red
    DARK_RED = '88'
    GREE = '22'  # green
    O = '130'  # orange
    B = '19'  # blue
    P = '98'  # purple
    C = '4'  # cyan
    GREY = '7'  # grey
    Y = '3' # yellow

class CustomFormatter(logging.Formatter):
    """
    Logging Formatter to add colors
    """

    def __init__(self, verbose):
        super().__init__()
        self._verbose = verbose
        self._previous_carriage_returned = False

    RESET_COLOR = "\033[0m"
    BASE_FORMAT = "{0}[{1}] " + RESET_COLOR + "{3} {2}" + RESET_COLOR + " {0}: %(message)s" + RESET_COLOR
    VERBOSE_FORMAT = "{0} [%(asctime)s] " + RESET_COLOR + "{3} {2}" + RESET_COLOR + "{0}[%(module)s]: %(message)s" + RESET_COLOR
    LEVEL_SPACES = 10

    def _fg(self, color):
        return "\33[38;5;" + color + "m"

    def _bg(self, color):
        return "\33[48;5;" + color + "m"

    def _compute_format(self, verbose):
        base_format = CustomFormatter.VERBOSE_FORMAT if verbose else CustomFormatter.BASE_FORMAT
        current_time = datetime.datetime.now().strftime("%a %H:%M:%S")

        return {
        logging.DEBUG: base_format.format(
            self._fg(Color.GREY),
            current_time,
            "DEBUG".center(CustomFormatter.LEVEL_SPACES),
            self._bg(Color.GREY) + self._fg(Color.W)
        ),
        logging.INFO: base_format.format(
            self._fg(Color.C),
            current_time,
            "INFO".center(CustomFormatter.LEVEL_SPACES),
            self._bg(Color.C) + self._fg(Color.W)
        ),
        logging.WARNING: base_format.format(
            self._fg(Color.Y),
            current_time,
            "WARNING".center(CustomFormatter.LEVEL_SPACES),
            self._bg(Color.Y) + self._fg(Color.W)
        ),
        logging.ERROR: base_format.format(
            self._fg(Color.R),
            current_time,
            "ERROR".center(CustomFormatter.LEVEL_SPACES),
            self._bg(Color.R) + self._fg(Color.W)
        ),
        logging.CRITICAL: base_format.format(
            self._fg(Color.DARK_RED),
            current_time,
            "CRITICAL".center(CustomFormatter.LEVEL_SPACES),
            self._bg(Color.DARK_RED) + self._fg(Color.W)
        )
    }

    def format(self, record):
        prefix = "\n"
        suffix = ""
        if self._previous_carriage_returned:
            prefix = "\r"
            # Suffix to write over the previous line when doing inplace writing
            suffix = " " * 10

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

