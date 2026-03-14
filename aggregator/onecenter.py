import torch
import numpy as np
from .base import _BaseAggregator

in_eta = 0.8


class OneCenterAggregator(_BaseAggregator):
    def __call__(self, updates):
        eta = in_eta
        n = len(updates)
        k = int(np.floor(eta * n))
        weights = [1] * n
        radii = [None] * n

        for i in range(n):
            _, radii[i] = k_closest(updates, updates[i], k)
            weights[i] = 1

        radii_tensor = torch.tensor(radii, device="cuda:0")
        min_r_index = torch.argmin(radii_tensor)
        k_data_idx, _ = k_closest(updates, updates[min_r_index], k)
        k_data = [updates[i] for i in k_data_idx]
        k_data_tensors = torch.stack(k_data)
        k_data_weights = [weights[i] for i in k_data_idx]
        k_data_weights_tensor = torch.tensor(k_data_weights, dtype=k_data_tensors.dtype, device=k_data_tensors.device)

        inner_center = torch.mean(k_data_tensors * k_data_weights_tensor.unsqueeze(1), dim=0)
        return inner_center

    def __str__(self):
        return "OneCenter Aggregator"
def k_closest(data, x, k):
    n = len(data)
    data_tensor = torch.stack(data).to(x.device)
    x_tensor = x.unsqueeze(0).repeat(len(data), 1)

    distances = torch.norm(data_tensor - x_tensor, dim=1)
    sorted_distances, sorted_indices = torch.sort(distances)

    key_dist = sorted_distances[k - 1]
    k_data_idx = sorted_indices[distances <= key_dist]

    return k_data_idx, key_dist