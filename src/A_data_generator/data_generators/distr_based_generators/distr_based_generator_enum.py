import time
from concurrent.futures import thread
from enum import Enum
from threading import Timer

from utils import logger


class Distribution(Enum):
    UNIFORM = "uniform"
    GEOMETRIC = "geometric"
    POISSON = "poisson"
    BINOMIAL = "binomial"
    HYPERGEOMETRIC = "hypergeometric"
    NORMAL = "normal"



