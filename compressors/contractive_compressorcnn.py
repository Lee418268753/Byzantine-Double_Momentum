import torch
import numpy as np

from .base_class import Compressor


class TopKCompressorcnn(Compressor):
    def __init__(self, ratio=0.1):
        self.ratio = ratio
        assert 0 < ratio <= 1, f"{ratio} out of boundaries"
        super().__init__(1 / ratio)

    def compress(self, x):
        global residuals

        if not isinstance(x, torch.Tensor):
            return x

        device = x.device
        y = x.flatten()
        d = y.numel()

        k = max(int(d * self.ratio), 1)
        values, indices = torch.topk(y.abs(), k=k)

        y_compressed = torch.zeros_like(y)
        y_compressed[indices] = y[indices]

        return y_compressed.view_as(x)


