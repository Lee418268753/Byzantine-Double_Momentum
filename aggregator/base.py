"""
Aggregators which takes in weights and gradients.
"""
import torch

from utils.logger import Logger
from typing import List, Union
from worker import TorchWorker


class _BaseAggregator(object):
    """Base class of aggregators.
    Args:
        dist_communicator (object): A link object which can broadcast / gather, etc.
    """

    def __init__(self):
        Logger.get().info("Init aggregator: " + self.__str__())
        # log_dict({"Aggregator": self.__str__(), "Type": "Setup"})
    
    def _get_updates(
        self, inputs: Union[List[TorchWorker], List[torch.Tensor], torch.Tensor]
    ):
        if all(isinstance(element, TorchWorker) for element in inputs):
            updates = torch.stack(list(map(lambda w: w.get_update(), inputs)))
        elif isinstance(inputs, List) and all(
            isinstance(element, torch.Tensor) for element in inputs
        ):
            updates = torch.stack(inputs, dim=0)
        else:
            updates = inputs
        return updates

    def __call__(self, inputs):
        """Aggregate the inputs and update in-place.
        Args:
            inputs (list): A list of tensors to be aggregated.
        Raises:
            NotImplementedError:
        """
        raise NotImplementedError


class _BaseAsyncAggregator(object):
    """AsyncAggregator base object"""

    def __init__(self):
        Logger.get().info("Init aggregator: " + self.__str__())
        # log_dict({"Aggregator": self.__str__(), "Type": "Setup"})

    def __call__(self, inputs):
        """Aggregate the inputs and update in-place.
        Args:
            inputs (list): A list of tensors to be aggregated.
        Raises:
            NotImplementedError:
        """
        raise NotImplementedError


class Mean(_BaseAggregator):
    def __call__(self, inputs):
        values = torch.stack(inputs, dim=0).mean(dim=0)
        return values

    def __str__(self):
        return "Mean"
