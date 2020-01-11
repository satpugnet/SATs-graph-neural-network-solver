import torch

from configs import exp_configs, other_configs


class MultiGpu:
    @staticmethod
    def is_enabled():
        return torch.cuda.device_count() > 1 and \
               other_configs["multi_gpu"] and \
               other_configs["device"].type == "cuda"