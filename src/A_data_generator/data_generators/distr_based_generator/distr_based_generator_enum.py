from enum import Enum


class Distribution(Enum):
    UNIFORM = "uniform"
    GEOMETRIC = "geometric"
    POISSON = "poisson"
    BINOMIAL = "binomial"
    HYPERGEOMETRIC = "hypergeometric"
    NORMAL = "normal"