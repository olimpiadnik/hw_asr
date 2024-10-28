import torch_audiomentations
from torch import Tensor, nn


class Gain(nn.Module):
    def __init__(self, p=0.2, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.Gain(p=0.2, *args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)