from enum import Enum


class Pooling(Enum):
    GLOBAL_ADD = "global_add"
    GLOBAL_MEAN = "global_mean"
    GLOBAL_MAX = "global_max"
