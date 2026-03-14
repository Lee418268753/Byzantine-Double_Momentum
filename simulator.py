import copy
import logging
import time
from typing import Union, Callable, Any

import numpy as np
from utils.random_generator import RandomNumber

import torch

from worker import TorchWorker
from server import TorchServer

class BaseSimulator(object):
    def __init__(
            self,
            server,
            use_cuda,

    ):
        self.use_cuda = use_cuda
        self.server = server

        self.random_states = {}
        self.json_logger = logging.getLogger("stats")
        self.debug_logger = logging.getLogger("debug")

    def cache_random_state(self) -> None:
        if self.use_cuda:
            self.random_states["torch_cuda"] = torch.cuda.get_rng_state()
        self.random_states["torch"] = torch.get_rng_state()
        self.random_states["numpy"] = np.random.get_state()

    def restore_random_state(self) -> None:
        if self.use_cuda:
            torch.cuda.set_rng_state(self.random_states["torch_cuda"])
        torch.set_rng_state(self.random_states["torch"])
        np.random.set_state(self.random_states["numpy"])

    def __str__(self):
        return f"BaseSimulator"

class TrainSimulator(BaseSimulator):
    def __init__(
            self,
            metrics,
            max_batches_per_epoch,
            log_interval,
            aggregator: Callable[[list], torch.Tensor],
            *args,
            **kwargs):
        self.metrics = metrics

        self.workers = []
        self.train_workers = []
        # NOTE: omniscient_callbacks are called before aggregation or gossip
        self.omniscient_callbacks = []
        self.max_batches_per_epoch = max_batches_per_epoch
        self.log_interval = log_interval
        self.aggregator = aggregator
        self.aggregated_gradient = None
        super(TrainSimulator, self).__init__(*args, **kwargs)

    def add_worker(self, worker):
        worker.add_metrics(self.metrics)
        self.debug_logger.info(f"=> Add worker {worker}")
        self.workers.append(worker)
        if worker.is_train():
            self.train_workers.append(worker)

    def register_omniscient_callback(self, callback):
        self.omniscient_callbacks.append(callback)

    def parallel_call(self, f: Callable[[TorchWorker], None]) -> None:
        for w in self.workers:
            self.cache_random_state()
            f(w)
            self.restore_random_state()

    def parallel_get(self, f: Callable[[TorchWorker], Any]) -> list:
        results = []
        for w in self.workers:
            self.cache_random_state()
            results.append(f(w))
            self.restore_random_state()
        return results


    def aggregation_and_update(self):
        start = time.time()
        for omniscient_attacker_callback in self.omniscient_callbacks:
            omniscient_attacker_callback()



        diffs = self.parallel_get(lambda w: w.get_gradient())
        updated_diffs = self.server.update_gradient(diffs)
        
        aggregated = self.aggregator(updated_diffs)
        

        self.server.set_gradient(aggregated)
        self.server.apply_gradient()
        

        end = time.time()
        return end - start

    def train(self, epoch):
        self.debug_logger.info(f"Train epoch {epoch}")
        
        self.parallel_call(lambda worker: worker.train_epoch_start())
        
        agg_time = 0.0
        start = time.time()
        progress = 0
        for batch_idx in range(self.max_batches_per_epoch):
            try:
                
                results = self.parallel_get(lambda w: w.compute_gradient())
                
                
                # self.debug_logger.info(f"results:{results}")
                agg_time_per_batch = self.aggregation_and_update()
                agg_time += agg_time_per_batch
                progress += sum(res["batch_size"] for res in results)
                if batch_idx % self.log_interval == 0:
                    self.log_train(progress, batch_idx, epoch, results)
                start1 = time.time()
                RandomNumber.sample()
                end1 = time.time()
                aggtime = end1 - start1
            except StopIteration:
                end = time.time()
                total_time = end - start
                cs_time = agg_time
                node_time = total_time - cs_time
                
                
                self.debug_logger.info(
                    f"total time: {total_time:.4f} cs time: {cs_time:.4f} node time: {node_time:.4f} start1 time: {agg_time:.4f}")
                break

    def set_trusted_clients(self, ids) -> None:
        r"""Set a list of input as trusted. This is usable for trusted-based
        algorithms that assume some clients are known as not Byzantine.

        :param ids: a list of client ids that are trusted
        :type ids: list
        """
        for worker in self.train_workers:
            if int(worker.id()) in ids:
                worker.trust()

    def log_train(self, progress, batch_idx, epoch, results):
        length = sum(res["batch_size"] for res in results)

        r = {
            "_meta": {"type": "train"},
            "E": epoch,
            "B": batch_idx,
            "Length": length,
            "Loss": sum(res["loss"] * res["batch_size"] for res in results) / length,
        }

        for metric_name in self.metrics:
            r[metric_name] = (
                    sum(res["metrics"][metric_name] * res["batch_size"] for res in results)
                    / length
            )

        # Output to console
        total = sum(self.parallel_get(lambda w: w.get_data_size()))
        pct = 100 * progress / total
        self.debug_logger.info(
            f"[E{r['E']:2}B{r['B']:<3}| {progress:6}/{total} ({pct:3.0f}%) ] Loss: {r['Loss']:.4f} "
            + " ".join(name + "=" + "{:>8.4f}".format(r[name]) for name in self.metrics)
        )

        # Output to file
        self.json_logger.info(r)

    def __str__(self):
        return f"TrainSimulator"


class EvalSimulator(BaseSimulator):
    def __init__(
            self,
            model: torch.nn.Module,
            data_loader: torch.utils.data.DataLoader,
            loss_func: torch.nn.modules.loss._Loss,
            device: Union[torch.device, str],
            metrics: dict,
            *args, **kwargs):
        self.model = model
        self.data_loader = data_loader
        self.loss_func = loss_func
        self.device = device
        self.metrics = metrics
        self.json_logger = logging.getLogger("stats")
        self.debug_logger = logging.getLogger("debug")
        super(EvalSimulator, self).__init__(*args, **kwargs)

    def test(self, epoch):
        # model = self.server.get_model()
        self.model.eval()
        r = {
            "_meta": {"type": "validation"},
            "E": epoch,
            "Length": 0,
            "Loss": 0,
        }
        for name in self.metrics:
            r[name] = 0

        with torch.no_grad():
            for _, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                r["Loss"] += self.loss_func(output, target,self.model).item() * len(target)
                r["Length"] += len(target)

                for name, metric in self.metrics.items():
                    r[name] += metric(output, target) * len(target)

        for name in self.metrics:
            r[name] /= r["Length"]
        r["Loss"] /= r["Length"]

        # Output to file
        self.json_logger.info(r)
        self.debug_logger.info(
            f"\n=> Eval Loss={r['Loss']:.4f} "
            + " ".join(name + "=" + "{:>8.4f}".format(r[name]) for name in self.metrics)
            + "\n"
        )
        return r["top1"]
