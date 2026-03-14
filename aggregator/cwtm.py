from .base import _BaseAggregator
import torch
import math


class Cwtm(_BaseAggregator):
    def __init__(self, b):
        self.b = b
        super(Cwtm, self).__init__()

    def __call__(self, inputs):
        '''coordinate-wise trimmed mean'''
        if self.b == 0:  
            return torch.mean(torch.stack(inputs), dim=0)
        inputs = torch.stack(inputs, dim=0)  # (n, d)

        n,d = inputs.shape

        num_points_to_discard = min(n // 2, math.ceil(n * self.b))

        if num_points_to_discard == 0:
            return torch.mean(inputs, dim=0)

        sorted_inputs, _ = torch.sort(inputs, dim=0)

        trimmed_inputs = sorted_inputs[num_points_to_discard: -num_points_to_discard]

        aggregated_update = torch.mean(trimmed_inputs, dim=0)

        return aggregated_update

    def __str__(self):
        return "Coordinatewise Trimmed Mean (b={})".format(self.b)
