from typing import List
import torch
import itertools
class TorchServer(object):
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.aggregated_gradient = None 
        self.gradient = []
        self.update = None
    def apply_gradient(self) -> None:
        self.optimizer.step()
    def clean(self) -> None:
        self.aggregated_gradient = None
    def store(self,gradients: List[torch.Tensor]) -> None:
        self.gradient.append(torch.cat(gradients))
    def update_gradient(self, diffs: List[torch.Tensor]) -> None:
        if self.aggregated_gradient is None:
            self.aggregated_gradient = torch.zeros_like(diffs[0])
        if len(self.gradient) ==0:
            return diffs
        else:
            updated_diffs = []
            extended_gradients = itertools.chain(self.gradient, itertools.repeat(self.gradient[-1]))
            for diff,gradient in zip(diffs,extended_gradients):
                combined = gradient + diff
                updated_diffs.append(combined)
            self.gradient =[]
            return updated_diffs

    def set_gradient(self, gradient: torch.Tensor) -> None:
        beg = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                end = beg + len(p.grad.view(-1))
                x = gradient[beg:end].reshape_as(p.grad.data)
                p.grad.data = x.clone().detach()
                beg = end