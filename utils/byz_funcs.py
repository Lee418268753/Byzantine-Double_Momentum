import numpy as np
import torch
# IID vs Non-IID
from data_funcs.sampler import (
    DistributedSampler,
)

# Aggregators
from aggregator import *


def _get_aggregator(args):
    if args.agg == "avg":
        return Mean()

    if args.agg == "cm":
        return CM()
    if args.agg == "cwtm":

        return Cwtm(b=args.f / args.n)
    if args.agg == "onecenter":
        return OneCenterAggregator()

    if args.agg == "cp":
        if args.clip_scaling is None:
            tau = args.clip_tau
        elif args.clip_scaling == "linear":
            tau = args.clip_tau / (1 - args.momentum)
        elif args.clip_scaling == "sqrt":
            tau = args.clip_tau / np.sqrt(1 - args.momentum)
        else:
            raise NotImplementedError(args.clip_scaling)
        return Clipping(tau=tau, n_iter=3)

    if args.agg == "rfa":
        return RFA(T=8)
    if args.agg == "fltrust":
        return Fltrust()
    if args.agg == "tm":
        return TM(b=args.f)
    if args.agg == "dnc":
        return Dnc(num_byzantine=args.f)
    # if args.agg == "bulyan":
    #     return Bulyan()

    if args.agg == "byzantinesgd":
        return ByzantineSGD
    if args.agg == "bulyan":
        return Bulyan(n_byzantine=args.f)

    if args.agg == "krum":
        T = int(np.ceil(args.n / args.bucketing)) if args.bucketing > 0 else args.n
        return Krum(n=T, f=args.f, m=1)

    raise NotImplementedError(args.agg)


def average_nearest_neighbors(vectors, f, pivot):
    vector_scores = list()

    for i in range(len(vectors)):
        # compute distance to pivot
        distance = vectors[i].sub(pivot).norm().item()
        vector_scores.append((i, distance))

    # sort vector_scores by increasing distance to pivot
    vector_scores.sort(key=lambda x: x[1])

    # Return the average of the n-f closest vectors to pivot
    closest_vectors = [vectors[vector_scores[j][0]] for j in range(len(vectors) - f)]
    return torch.stack(closest_vectors).mean(dim=0)

def NNM(args, aggregator, numb_iter=1):
    """
    Key functionality.
    """
    print("Using nearest neighbor mixing.")
    def aggr(inputs):
        for _ in range(numb_iter):
            mixed_vectors = list()
            for vector in inputs:
                mixed_vectors.append(average_nearest_neighbors(inputs, args.f, vector))
            inputs = mixed_vectors
        return aggregator(inputs)

    return aggr
def bucketing_wrapper(args, aggregator, s):
    """
    Key functionality.
    """
    print("Using bucketing wrapper.")

    def aggr(inputs):
        indices = list(range(len(inputs)))
        np.random.shuffle(indices)

        T = int(np.ceil(args.n / s))

        reshuffled_inputs = []
        for t in range(T):
            indices_slice = indices[t * s : (t + 1) * s]
            g_bar = sum(inputs[i] for i in indices_slice) / len(indices_slice)
            reshuffled_inputs.append(g_bar)
        return aggregator(reshuffled_inputs)

    return aggr


def get_aggregator(args):
    aggr = _get_aggregator(args)
    if args.bucketing != 0:
        return bucketing_wrapper(args, aggr, args.bucketing)
    if args.nnm:
        return NNM(args,aggr,1)
    return aggr


def get_sampler_callback(args, rank):
    """
    Get sampler based on the rank of a worker.
    The first `n-f` workers are good, and the rest are Byzantine
    """
    n_good = args.n - args.f
    if rank >= n_good:
        # Byzantine workers
        return lambda x: DistributedSampler(
            num_replicas=n_good,
            rank=rank % n_good,
            shuffle=True,
            dataset=x,
            full_dataset=args.full_dataset,
            shuffle_iter=True,
        )


    return lambda x: DistributedSampler(
        num_replicas=n_good,
        rank=rank,
        shuffle=True,
        dataset=x,
        full_dataset=args.full_dataset,
        shuffle_iter=True,
    )


def get_test_sampler_callback(args):
    return lambda x: DistributedSampler(
        num_replicas=1,
        rank=0,
        shuffle=False,
        dataset=x,
        shuffle_iter=False,
    )
