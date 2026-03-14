import torch
import sys
from collections import defaultdict
from copy import deepcopy
from typing import Optional, Union, Callable, Any, Tuple
from opts import get_args
from utils.random_generator import RandomNumber
from compressors import Identity, Compressor
from server import TorchServer

class TorchWorker(object):
    """A worker for distributed training.
    Compute gradients locally and store the gradient.
    """

    def __init__(
            self,
            server: TorchServer,
            data_loader: torch.utils.data.DataLoader,
            model: torch.nn.Module,
            model_snap: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            optimizer_snap: torch.optim.Optimizer,
            loss_func: torch.nn.modules.loss._Loss,
            device: Union[torch.device, str],
            compression: Compressor,
    ):
        self.server = server
        self.data_loader = data_loader
        # self.scalar = len(data_loader.dataset) / (
        #         data_loader.sampler.num_replicas * len(data_loader.sampler))
        self.scalar = 1.
        self.model = model
        self.model_snap = model_snap
        self.optimizer = optimizer
        self.optimizer_snap = optimizer_snap
        self.loss_func = loss_func
        self.device = device
        if compression is None:
            self.compression = Identity()
        else:
            self.compression = compression

        self.running = {}
        self.metrics = {}
        self.state = defaultdict(dict)

    def add_metric(
            self,
            name: str,
            callback: Callable[[torch.Tensor, torch.Tensor], float],
    ):
        """
        The `callback` function takes predicted and groundtruth value
        and returns its metric.
        """
        if name in self.metrics or name in ["loss", "length"]:
            raise KeyError(f"Metrics ({name}) already added.")

        self.metrics[name] = callback

    def add_metrics(self, metrics: dict):
        for name in metrics:
            self.add_metric(name, metrics[name])

    
    def get_data_size(self) -> int:
        return len(self.data_loader.dataset)

    def is_train(self):
        return True    
    def __str__(self) -> str:
        return "TorchWorker"

    def train_epoch_start(self) -> None:
        self.running["train_loader_iterator"] = iter(self.data_loader)
        self.model.train()

    def compute_gradient(self):
        results = {}

        data, target = next(self.running["train_loader_iterator"])
        data, target = data.to(self.device), target.to(self.device) 
        self.optimizer.zero_grad() 
        output = self.model(data) 
        loss = torch.clamp(self.loss_func(output, target,self.model), 0, 1e6)
        loss.backward()  

        self.running["data"] = data  
        self.running["target"] = target  

        self._save_grad()  

        self.model_snap.load_state_dict(deepcopy(self.model.state_dict()))  

        results["loss"] = loss.item()
        results["batch_size"] = len(target)
        results["metrics"] = {}
        for name, metric in self.metrics.items():
            results["metrics"][name] = metric(output, target)

        return results


    def get_gradient(self) -> torch.Tensor:  
        return self.scalar * self._get_saved_grad()

    def apply_gradient(self) -> None:
        self.optimizer.step()

    def set_gradient(self, gradient: torch.Tensor) -> None:  
        beg = 0 
        for p in self.model.parameters():
            end = beg + len(p.grad.view(-1)) 
            x = gradient[beg:end].reshape_as(p.grad.data) 
            p.grad.data = x.clone().detach()  
            beg = end 

    def _save_grad(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if "saved_grad" not in param_state:
                    param_state["momentum_buffer"] = (1 - self.momentum) * p.grad.data.detach().clone()
                    param_state["top_buffer"] = p.grad.data.detach().clone()*(1-self.momentum)
                    param_state["grads_buffer"] = self.compression(param_state["top_buffer"].detach().clone())
                    param_state["saved_grad"] = torch.zeros_like(p.grad.data.detach().clone())
                else:
                    param_state["momentum_buffer"] = (self.momentum) * param_state["momentum_buffer"] + (
                                    1 - self.momentum) * p.grad.data.detach().clone()
                    param_state["top_buffer"].mul_(self.momentum).add_(p.grad.data.detach().clone(), alpha=1 - self.momentum)
                    diff = self.compression(
                    param_state["top_buffer"].detach().clone() - param_state["grads_buffer"].detach().clone()
                        )
                    param_state["grads_buffer"].add_(diff)
                    param_state["saved_grad"] = diff.detach().clone()


    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                layer_gradients.append(
                    param_state["saved_grad"].data.view(-1))
        return torch.cat(layer_gradients)


class MomentumWorker(TorchWorker):
    def __init__(self, server,momentum, *args, **kwargs):
        super().__init__(server,*args, **kwargs)
        self.momentum = momentum
        self.server = server

    def _save_grad(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = p.grad.data.detach().clone()
                else:
                    param_state["momentum_buffer"].mul_(self.momentum).add_(p.grad)

    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                layer_gradients.append(
                    self.compression(param_state["momentum_buffer"].data.view(-1)))
        return torch.cat(layer_gradients)


    def __str__(self) -> str:
        return "MomentumWorker"

class DoubleMomentumWorker(TorchWorker):
    def __init__(self, momentum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum


    def _compute_previous_grad_top(self) -> None:  
        self.model_snap.train()
        self.optimizer_snap.zero_grad()
        data, target  = self.running["data"], self.running["target"]
        output = self.model_snap(data)
        loss = self.loss_func(output, target,self.model_snap)
        loss.backward()



    def _save_grad(self) -> None:
        self._compute_previous_grad_top()
        worker_gradients=[]
        for group, group_snap in zip(self.optimizer.param_groups, self.optimizer_snap.param_groups):
            for p, p_snap in zip(group["params"], group_snap["params"]):
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if "top_buffer" not in param_state:
                    param_state["top_buffer"] =  p.grad.data.detach().clone()
                    param_state["second_buffer"] = p.grad.data.detach().clone()
                    param_state["grads_buffer"] = p.grad.data.detach().clone()
                    param_state["diff"] = torch.zeros_like(p.grad.data.detach().clone())
                    param_state["first_buffer"]=torch.zeros_like(p.grad.data.detach().clone())
                    worker_gradients.append(param_state["first_buffer"].data.view(-1))
                else:
                    param_state["top_buffer"]=param_state["top_buffer"]*(self.momentum)+p.grad.data.detach().clone()*(1-self.momentum) +(self.momentum)*(p.grad.data.detach().clone()-p_snap.grad.data.detach().clone())
                    param_state["second_buffer"]=param_state["second_buffer"].detach().clone()*(self.momentum)+param_state["top_buffer"].detach().clone()*(1-self.momentum)
                    diff = self.compression(
                    param_state["second_buffer"].detach().clone() -
                    param_state["grads_buffer"].detach().clone()
                    )
                    param_state["first_buffer"] = param_state["grads_buffer"].detach().clone()
                    worker_gradients.append(param_state["first_buffer"].data.view(-1))
                    param_state["grads_buffer"].add_(diff)
                    param_state["diff"] = diff.detach().clone()
        self.server.store(worker_gradients)

    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                layer_gradients.append(param_state["diff"].data.view(-1))
        return torch.cat(layer_gradients)

    def __str__(self) -> str:
        return "DoubleMomentumWorker"
    

class VRDoubleMomentumWorker(TorchWorker):
    def __init__(self, momentum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum


    def _compute_previous_grad_top(self) -> None:  
        self.model_snap.train()
        self.optimizer_snap.zero_grad()
        data, target  = self.running["data"], self.running["target"]
        output = self.model_snap(data)
        loss = self.loss_func(output, target,self.model_snap)
        loss.backward()



    def _save_grad(self) -> None:
        self._compute_previous_grad_top()
        worker_gradients=[]
        for group, group_snap in zip(self.optimizer.param_groups, self.optimizer_snap.param_groups):
            for p, p_snap in zip(group["params"], group_snap["params"]):
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if "top_buffer" not in param_state:
                    param_state["top_buffer"] =  p.grad.data.detach().clone()*(1-self.momentum)
                    param_state["second_buffer"] = (1-self.momentum)*param_state["top_buffer"].detach().clone()
                    param_state["grads_buffer"] = self.compression(param_state["second_buffer"].detach().clone())
                    param_state["diff"] = param_state["grads_buffer"].detach().clone()
                    param_state["first_buffer"]=torch.zeros_like(p.grad.data.detach().clone())
                    worker_gradients.append(param_state["first_buffer"].data.view(-1))
                else:
                    param_state["top_buffer"]=param_state["top_buffer"]*(self.momentum)+p.grad.data.detach().clone()*(1-self.momentum) +(self.momentum)*(p.grad.data.detach().clone()-p_snap.grad.data.detach().clone())
                    param_state["second_buffer"]=param_state["second_buffer"].detach().clone()*(self.momentum)+param_state["top_buffer"].detach().clone()*(1-self.momentum)
                    diff = self.compression(
                    param_state["second_buffer"].detach().clone() -
                    param_state["grads_buffer"].detach().clone()
                    )
                    param_state["first_buffer"] = param_state["grads_buffer"].detach().clone()
                    worker_gradients.append(param_state["first_buffer"].data.view(-1))
                    param_state["grads_buffer"].add_(diff)
                    param_state["diff"] = diff.detach().clone()
        self.server.store(worker_gradients)

    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                layer_gradients.append(param_state["diff"].data.view(-1))
        return torch.cat(layer_gradients)

    def __str__(self) -> str:
        return "VRDoubleMomentumWorker"
    


class TopMomentumWorker(TorchWorker):
    def __init__(self, momentum,server, *args, **kwargs):
        super().__init__(server,*args, **kwargs)
        self.momentum = momentum
        self.server = server

    def _save_grad(self) -> None:
        worker_gradients = []
        for group, group_snap in zip(self.optimizer.param_groups, self.optimizer_snap.param_groups):
            
            for p, p_snap in zip(group["params"], group_snap["params"]):
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if "top_buffer" not in param_state:
                    param_state["top_buffer"] = p.grad.data.detach().clone()*(1-self.momentum)
                    param_state["grads_buffer"] = self.compression(param_state["top_buffer"].detach().clone())
                    param_state["diff"] = torch.zeros_like(p.grad.data.detach().clone())
                    param_state["momentum_buffer"] = torch.zeros_like(p.grad.data.detach().clone())
                    param_state["first_buffer"]=torch.zeros_like(p.grad.data.detach().clone())
                    worker_gradients.append(param_state["first_buffer"].data.view(-1))

                else:
                    param_state["top_buffer"].mul_(self.momentum).add_(p.grad.data.detach().clone(), alpha=1 - self.momentum)
                    diff = self.compression(
                        param_state["top_buffer"].detach().clone() - param_state["grads_buffer"].detach().clone()
                    )
                    param_state["first_buffer"] = param_state["grads_buffer"].detach().clone()
                    worker_gradients.append(param_state["first_buffer"].data.view(-1))
                    param_state["grads_buffer"].add(diff)
                    param_state["diff"] = diff.detach().clone()
        self.server.store(worker_gradients)

    def _get_saved_grad(self) -> torch.Tensor:
        diffs = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                diffs.append(param_state["diff"].data.view(-1))
        return torch.cat(diffs)

    def __str__(self) -> str:
        return "TopMomentumWorker"


class DianaWorker(TorchWorker):
    def __init__(self,server, *args, **kwargs):
        super().__init__(server,*args, **kwargs)
        self.server = server

    def _save_grad(self) -> None:
        worker_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if "shift_buffer" not in param_state:
                    param_state["shift_buffer"] = torch.zeros_like(p.grad.data.detach().clone())
                    param_state["grad_buffer"] = self.compression(p.grad.data.detach().clone())
                    param_state["diff"] = torch.zeros_like(p.grad.data.detach().clone())
                    param_state["first_buffer"]=torch.zeros_like(p.grad.data.detach().clone())
                    worker_gradients.append(param_state["first_buffer"].data.view(-1))
                else:
                    diff = self.compression(
                        p.grad.data.detach().clone() - param_state["shift_buffer"])
                    param_state["first_buffer"]=param_state["shift_buffer"].detach().clone()
                    worker_gradients.append(param_state["first_buffer"].data.view(-1))
                    param_state["grad_buffer"] = param_state["shift_buffer"] + diff
                    param_state["diff"] = diff.detach().clone()
                    param_state["shift_buffer"].add_(diff, alpha=1 / self.compression.w)
        self.server.store(worker_gradients)

    def _get_saved_grad(self) -> torch.Tensor:
        diffs = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                diffs.append(param_state["diff"].data.view(-1))
        return torch.cat(diffs)

    def __str__(self) -> str:
        return "DianaWorker"


class EF21Worker(TorchWorker):
    def __init__(self, server,*args, **kwargs):
        super().__init__(server,*args, **kwargs)
        self.server = server
    def _compute_full_grad(self) -> None:
        self.optimizer.zero_grad()
        loss = 0.
        n_points = 0
        for data, target in self.data_loader:
            batch_size = data.shape[0]
            n_points += batch_size
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss += self.loss_func(output, target, self.model) * batch_size
        loss /= n_points
        loss.backward()


    def _save_grad(self) -> None:
        self._compute_full_grad()
        worker_gradients = []
        for group, group_snap in zip(self.optimizer.param_groups, self.optimizer_snap.param_groups):
            for p, p_snap in zip(group["params"], group_snap["params"]):
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if "grads_buffer" not in param_state:
                    param_state["grads_buffer"] = p.grad.data.detach().clone()
                    param_state["diff"] = torch.zeros_like(p.grad.data.detach().clone())
                    
                    param_state["first_buffer"]=torch.zeros_like(p.grad.data.detach().clone())
                    worker_gradients.append(param_state["first_buffer"].data.view(-1))
                else:
                    diff = self.compression(
                        p.grad.data.detach().clone() - param_state["grads_buffer"].detach().clone()
                    )
                    param_state["first_buffer"] = param_state["grads_buffer"].detach().clone()

                    worker_gradients.append(param_state["first_buffer"].data.view(-1))
                    param_state["grads_buffer"].add_(diff)
                    param_state["diff"] = diff.detach().clone()
        self.server.store(worker_gradients)
    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                layer_gradients.append(param_state["diff"].data.view(-1))
        return torch.cat(layer_gradients)
    

    def __str__(self) -> str:
        return "EF21Worker"


class DashaWorker(TorchWorker):
    def __init__(self, server,*args, **kwargs):
        super().__init__(server,*args, **kwargs)
        self.server = server
    def _compute_full_grad(self) -> None:
        self.optimizer.zero_grad()
        loss = 0.
        n_points = 0
        for data, target in self.data_loader:
            batch_size = data.shape[0]
            n_points += batch_size
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss += self.loss_func(output, target, self.model) * batch_size
        loss /= n_points
        loss.backward()


    def _compute_previous_grad(self) -> None:
        self.model_snap.train()
        self.optimizer_snap.zero_grad()
        data, target  = self.running["data"], self.running["target"]
        output = self.model_snap(data)
        loss = self.loss_func(output, target, self.model_snap)
        loss.backward()

    def _save_grad(self) -> None:
        if RandomNumber.full_grad:
            self._compute_full_grad()
        else:
            self._compute_previous_grad()
        worker_gradients = []
        for group, group_snap in zip(self.optimizer.param_groups, self.optimizer_snap.param_groups):
            for p, p_snap in zip(group["params"], group_snap["params"]):
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if RandomNumber.full_grad:
                    param_state["h1_buffer"] = p.grad.data.detach().clone()
                    param_state["dasha_buffer"] = p.grad.data.detach().clone()
                    param_state["first_buffer"]=torch.zeros_like(param_state["h1_buffer"].detach().clone())
                    worker_gradients.append(param_state["first_buffer"].data.view(-1))
                    param_state["diff"] = torch.zeros_like(p.grad.data.detach().clone())
                    param_state["h_buffer"] = param_state["h1_buffer"]
                else:
                    param_state["h1_buffer"] = param_state["h_buffer"]+p.grad.data.detach().clone() - p_snap.grad.data.detach().clone()
                    diff = self.compression(
                        param_state["h1_buffer"]-param_state["h_buffer"]-(1 / (2*self.compression.w+1))*(param_state["dasha_buffer"] - param_state["h_buffer"])
                    )
                    param_state["first_buffer"]=param_state["dasha_buffer"].detach().clone()
                    worker_gradients.append(param_state["first_buffer"].data.view(-1))
                    param_state["dasha_buffer"].add_(diff)
                    param_state["h_buffer"] = param_state["h1_buffer"]
        self.server.store(worker_gradients)
    def _get_saved_grad(self) -> torch.Tensor:
        diffs = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                diffs.append(param_state["first_buffer"].data.view(-1))
        return torch.cat(diffs)

    def __str__(self) -> str:
        return "DashaaWorker"


class MarinaWorker(TorchWorker):
    def __init__(self, server,*args, **kwargs):
        super().__init__(server,*args, **kwargs)
        self.server = server
    def _compute_full_grad(self) -> None:
        self.optimizer.zero_grad()
        loss = 0.
        n_points = 0
        for data, target in self.data_loader:
            batch_size = data.shape[0]
            n_points += batch_size
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss += self.loss_func(output, target, self.model) * batch_size
        loss /= n_points
        loss.backward()


    def _compute_previous_grad(self) -> None:
        self.model_snap.train()
        self.optimizer_snap.zero_grad()
        data, target  = self.running["data"], self.running["target"]
        output = self.model_snap(data)
        loss = self.loss_func(output, target, self.model_snap)
        loss.backward()

    def _save_grad(self) -> None:
        if RandomNumber.full_grad:
            self._compute_full_grad()
        else:
            self._compute_previous_grad()
        worker_gradients = []
        for group, group_snap in zip(self.optimizer.param_groups, self.optimizer_snap.param_groups):
            for p, p_snap in zip(group["params"], group_snap["params"]):
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if RandomNumber.full_grad:
                    param_state["marina_buffer"] = p.grad.data.detach().clone()
                    param_state["first_buffer"]=torch.zeros_like(param_state["marina_buffer"].detach().clone())
                    worker_gradients.append(param_state["first_buffer"].data.view(-1))
                    param_state["diff"] = torch.zeros_like(p.grad.data.detach().clone())
                else:
                    diff = self.compression(
                        p.grad.data.detach().clone() - p_snap.grad.data.detach().clone()
                    )
                    param_state["first_buffer"]=param_state["marina_buffer"].detach().clone()
                    worker_gradients.append(param_state["first_buffer"].data.view(-1))
                    param_state["marina_buffer"].add_(diff)
                    param_state["diff"] = diff.detach().clone()
        self.server.store(worker_gradients)
    def _get_saved_grad(self) -> torch.Tensor:
        diffs = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                diffs.append(param_state["diff"].data.view(-1))
        return torch.cat(diffs)

    def __str__(self) -> str:
        return "MarinaWorker"


class ByzantineWorker(TorchWorker):
    def configure(self,momentum,simulator):
        # call configure after defining DistribtuedSimulator
        self.simulator = simulator
        self.momentum = momentum
        simulator.register_omniscient_callback(self.omniscient_callback)

    def get_gradient(self):
        return self._gradient
    
    def omniscient_callback(self):
        # Loop over good workers and accumulate their gradients
        gradients = []
        for w in self.simulator.workers:
            if not isinstance(w, ByzantineWorker):
                gradients.append(w.get_gradient())

        stacked_gradients = torch.stack(gradients, 1)
        self._gradient = torch.mean(stacked_gradients, 1)

    def compute_gradient(self) -> Tuple[float, int]:
        return super().compute_gradient()


    def __str__(self) -> str:
        return "ByzantineWorker"