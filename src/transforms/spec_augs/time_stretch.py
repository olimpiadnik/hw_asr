import numpy as np
import torchaudio.transforms
from torch import Tensor, nn


class TimeStretch(nn.Module):
    def __init__(
        self, p=0.2, min_stretch=0.95, max_stretch=1.05, n_freq=128, *args, **kwargs
    ):
        super().__init__()
        self.p = p
        self.min_stretch = min_stretch
        self.max_stretch = max_stretch

        self._aug = torchaudio.transforms.TimeStretch(n_freq=n_freq, *args, **kwargs)

    def __call__(self, data: Tensor):
        x = data  # .unsqueeze(1)
        device = x.device

        apply = np.random.choice(a=[False, True], size=(1), p=[1 - self.p, self.p])
        if apply[0] is False:
            return x.squeeze(1)

        stretch = (
            np.random.random() * (self.max_stretch - self.min_stretch) + self.min_stretch
        )
        return self._aug(x, stretch).to(device)  # .squeeze(1)