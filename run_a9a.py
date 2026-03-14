import os
import sys
import json
import numpy as np
from copy import deepcopy
import torch
import time
from collections import OrderedDict
from torch import nn
from opts import get_args
# Utility functions
from data_funcs.libsvm import LibSVM
from tasks.libsvm import LogisticRegression, libsvm
from tasks.loss import Loss
from utils.utils import top1_accuracy, grad_norm, \
    create_model_dir, init_metrics_meter, extend_metrics_dict, metric_to_dict,initialize_logger
from utils.logger import Logger
from utils.random_generator import RandomNumber
from compressors import get_compression
from utils.model_funcs import accuracy,calculate_accuracy
# Attacks
from attacks import *
from torch.nn.modules.loss import CrossEntropyLoss
from worker import MomentumWorker, TopMomentumWorker,MarinaWorker, DianaWorker,SIGNSGDWorker,EF21Worker,DashaWorker
from server import TorchServer
from tasks import cifar10, tiny_imagenet, fmnist, mnist,femnist
from utils.byz_funcs import get_sampler_callback, get_aggregator, get_test_sampler_callback
from tasks.mnist1 import Net
from simulator import TrainSimulator, EvalSimulator, EvalSimulator
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import csv
# Fixed HPs
# BATCH_SIZE = 32
# TEST_BATCH_SIZE = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = 'femnist'
EXTRA_ID = ''
N_FEATURES = 123
PENALTY = 1e-2
sampler_callback=True



def initial_outfolder(args):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"
    DATA_DIR = ROOT_DIR + f"datasets/{args.datasets}/"
    EXP_DIR = ROOT_DIR + f"outputs/{args.datasets}/"

    if not args.noniid:
        LOG_DIR = (
                EXP_DIR
                + f"f{args.f}_w{args.n}_{args.attack}_{args.agg}_m{args.momentum}_iid_seed{args.seed}_fa{args.factor}"
        )
    else:
        LOG_DIR = (
                EXP_DIR
                + f"f{args.f}_w{args.n}_{args.attack}_{args.agg}_m{args.momentum}_non-iid_s{args.alpha}_seed{args.seed}_fa{args.factor}"
        )
    return DATA_DIR, LOG_DIR
def initialize_worker(
        args,
        trainer,
        
        worker_rank,
        model,
        server,
        train_loader,
        model_snap,
        optimizer,
        optimizer_snap,
        loss_func,
        device,
        kwargs,
):
    compression = get_compression(args.compression)
    attack_compression = get_compression(args.compression)


    if worker_rank < args.n - args.f:
        if args.model == 'marina':
            return MarinaWorker(
                server=server,
                compression=compression,
                data_loader=train_loader,
                model=model,
                model_snap=model_snap,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                optimizer_snap=optimizer_snap,
                **kwargs,
            )
        elif args.model == 'diana':
            return DianaWorker(
                server=server,
                compression=compression,
                data_loader=train_loader,
                model=model,
                model_snap=model_snap,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                optimizer_snap=optimizer_snap,
                **kwargs,
            )
        elif args.model == 'dasha':
            return DashaWorker(
                server=server,
                compression=compression,
                data_loader=train_loader,
                model=model,
                model_snap=model_snap,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                optimizer_snap=optimizer_snap,
                **kwargs,
            )
        elif args.model == 'signsgd':
            return SIGNSGDWorker(
                server=server,
                compression=compression,
                data_loader=train_loader,
                model=model,
                model_snap=model_snap,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                optimizer_snap=optimizer_snap,
                **kwargs,
            )
        elif args.model == 'mom_sgd':
            return MomentumWorker(
                server=server,
                momentum=0.9,
                compression=compression,
                data_loader=train_loader,
                model=model,
                model_snap=model_snap,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                optimizer_snap=optimizer_snap,
                **kwargs,
            )
        elif args.model == 'top_sgd':
            return TopMomentumWorker(
                server=server,
                momentum=0.9,
                compression=compression,
                data_loader=train_loader,
                # batch_size=args.batch_size,
                model=model,
                model_snap=model_snap,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                optimizer_snap=optimizer_snap,
                **kwargs,
            )
        elif args.model == 'ef21':
            return EF21Worker(
                server=server,
                compression=compression,
                data_loader=train_loader,
                model=model,
                model_snap=model_snap,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                optimizer_snap=optimizer_snap,
                **kwargs,
            )
        elif args.model == 'sgd':
            return MomentumWorker(
                server=server,
                momentum=0.,
                compression=compression,
                data_loader=train_loader,
                model=model,
                model_snap=model_snap,
                loss_func=loss_func,
                device=device,
                optimizer=optimizer,
                optimizer_snap=optimizer_snap,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown model: {args.model}")

    if args.attack == "SF":
        attacker = SignFlippingWorker(
            server=server,
            compression=attack_compression,
            data_loader=train_loader,
            model=model,
            model_snap=model_snap,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            optimizer_snap=optimizer_snap,
            **kwargs,
        )
        attacker.configure(0,trainer)
        return attacker

    if args.attack == "LF":
        attacker =  LableFlippingWorker(
            server=server,
            revertible_label_transformer=lambda label: 9 - label,
            compression=attack_compression,
            data_loader=train_loader,
            model=model,
            model_snap=model_snap,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            optimizer_snap=optimizer_snap,
            **kwargs,
        )
        attacker.configure(0.9,trainer)
        return attacker

    if args.attack == "MinMax":
        attacker = MinMaxWorker(
            # id=str(worker_rank),
            model=model,
            data_loader=train_loader,
            # local_round=args.local_round,
            compression=attack_compression,
            model_snap=model_snap,
            optimizer_snap=optimizer_snap,
            num_byzantine=args.f,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            **kwargs,
        )
        attacker.configure(0,trainer)
        return attacker

    
    if args.attack == "RA":
        attacker = RandomWorker(

            model=model,
            data_loader=train_loader,
            # local_round=args.local_round,
            compression=attack_compression,
            model_snap=model_snap,
            optimizer_snap=optimizer_snap,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            **kwargs,
        )
        attacker.configure(0,trainer)
        return attacker    
    if args.attack == "NO":
        attacker = NoiseWorker(
            compression=attack_compression,
            model=model,
            data_loader=train_loader,
            model_snap=model_snap,
            optimizer_snap=optimizer_snap,
            # local_round=args.local_round,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            **kwargs,
        )
        attacker.configure(0,trainer)
        return attacker

    if args.attack == "mimic":
        attacker = MimicVariantAttacker(
            compression=attack_compression,
            warmup=args.mimic_warmup,
            data_loader=train_loader,
            model=model,
            model_snap=model_snap,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            optimizer_snap=optimizer_snap,
            **kwargs,
        )
        attacker.configure(0,trainer)
        return attacker

    if args.attack == "IPM":
        attacker = IPMAttack(
            server=server,
            compression=attack_compression,
            epsilon=0.1,
            data_loader=train_loader,
            model=model,
            model_snap=model_snap,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            optimizer_snap=optimizer_snap,
            **kwargs,
        )
        attacker.configure(0,trainer)
        return attacker

    if args.attack == "ALIE":
        attacker = ALittleIsEnoughAttack(
            server=server,
            n=args.n,
            m=args.f,
            # z=1.5,
            compression=attack_compression,
            data_loader=train_loader,
            model=model,
            model_snap=model_snap,
            loss_func=loss_func,
            device=device,
            optimizer=optimizer,
            optimizer_snap=optimizer_snap,
            **kwargs,
        )
        attacker.configure(0,trainer)
        return attacker

    raise NotImplementedError(f"No such attack {args.attack}")


def main(args):
    Logger.setup_logging(args.loglevel, logfile=args.logfile)
    Logger()
    DATA_DIR, LOG_DIR = initial_outfolder(args)
    initialize_logger(LOG_DIR)


    if args.use_cuda and not torch.cuda.is_available():
        print("=> There is no cuda device!!!!")
        device = "cpu"
    else:
        device = torch.device("cuda" if args.use_cuda else "cpu")
    kwargs = {"pin_memory": True, "num_workers": args.num_workers} if args.use_cuda else {}

    torch.manual_seed(args.seed) 
    np.random.seed(args.seed)


    model = None
    train_loader = None
    test_loader = None

    if args.datasets == "cifar10":
        model = cifar10.get_cifar10_model(use_cuda=args.use_cuda).to(device)
        train_loader = cifar10.get_train_loader(root_dir=DATA_DIR, n_workers=args.n,
                                                alpha=args.alpha, batch_size=args.batch_size,
                                                noniid=args.noniid)
        test_loader = cifar10.get_test_loader(root_dir=DATA_DIR,
                                              batch_size=args.test_batch_size)
        max_batches_per_epoch= 50
    elif args.datasets == "fmnist":
        model = fmnist.get_fmnist_model().to(device)
        train_loader = fmnist.get_train_loader(root_dir=DATA_DIR, n_workers=args.n,
                                               alpha=args.alpha, batch_size=args.batch_size,
                                               noniid=args.noniid)
        test_loader = fmnist.get_test_loader(root_dir=DATA_DIR,
                                             batch_size=args.test_batch_size)
    elif args.datasets == "mnist":
        model = mnist.get_mnist_model().to(device)
        train_loader = mnist.get_train_loader(root_dir=DATA_DIR, n_workers=args.n,
                                              alpha=args.alpha, batch_size=args.batch_size,
                                              noniid=args.noniid)
        test_loader = mnist.get_test_loader(root_dir=DATA_DIR,
                                            batch_size=args.test_batch_size)
    elif args.datasets == "tiny-imagenet":
        model = tiny_imagenet.get_tinyimg_model(use_cuda=args.use_cuda).to(device)
        train_loader = tiny_imagenet.get_train_loader(root_dir=DATA_DIR, n_workers=args.n,
                                                      alpha=args.alpha, batch_size=args.batch_size,
                                                      noniid=args.noniid)
        test_loader = tiny_imagenet.get_test_loader(root_dir=DATA_DIR,
                                                    batch_size=args.test_batch_size)

    elif args.datasets == "femnist":
        model = femnist.get_femnist_model().to(device)
        train_loader = femnist.get_train_loader(root_dir=DATA_DIR, n_workers=args.n,
                                              alpha=args.alpha, batch_size=args.batch_size,
                                              noniid=args.noniid)
        test_loader = femnist.get_test_loader(root_dir=DATA_DIR,
                                            batch_size=args.test_batch_size)
        max_batches_per_epoch= 60


    model_snap_s = [deepcopy(model) for _ in range(args.n)]


    optimizers = [torch.optim.SGD(model.parameters(), lr=args.lr) for _ in range(args.n)] 
    optimizers_snap = [torch.optim.SGD(model_snap_s[i].parameters(), lr=args.lr) for i in range(args.n)] 
    server_opt = torch.optim.SGD(model.parameters(),lr=args.lr) 

    Loss(nn.CrossEntropyLoss(), False, PENALTY)  
    loss_func = Loss.compute_loss

    metrics = {"top1": top1_accuracy}


    if args.batch_size > args.test_batch_size:
        args.test_batch_size = args.batch_size

    server = TorchServer(optimizer=server_opt)

    trainer = TrainSimulator(
        server=server,
        metrics=metrics,
        use_cuda=args.use_cuda,
        max_batches_per_epoch=max_batches_per_epoch,
        log_interval=10,
        aggregator=get_aggregator(args)
    )
    

    evaluator = EvalSimulator(
        model=model,
        server=server,
        data_loader=test_loader,
        loss_func=loss_func,
        device=device,
        metrics=metrics,
        use_cuda=args.use_cuda
    )



    rate=args.f/args.n
    if args.attack == "NA":
        args.n -= args.f
        args.f = 0

    for worker_rank in range(args.n):
        worker = initialize_worker(
            args,
            trainer,
            worker_rank,
            model=model,
            server=server,
            train_loader=train_loader[worker_rank],
            model_snap=model_snap_s[worker_rank],
            optimizer=optimizers[worker_rank],
            optimizer_snap=optimizers_snap[worker_rank],
            loss_func=loss_func,
            device=device,
            kwargs={},
        )
        trainer.add_worker(worker)

    # RandomNumber.full_grad_prob = 1 / len(trainer.workers[0].data_loader)
    RandomNumber.full_grad_prob = 0.
    if not args.dry_run:
        acc_test=[]
        iterations=[]
        model_dir = create_model_dir(args,rate)
        if os.path.exists(os.path.join(
                model_dir, 'acc.json')):
            Logger.get().info(f"{model_dir} already exists.")
            Logger.get().info("Skipping this setup.")
            return
        start=time.time()
        for epoch in range(1, args.epochs + 1):
            RandomNumber.full_grad = True
            start=time.time()
            trainer.train(epoch)
            end = time.time()
            time1 = end -start
            print(time1)
            acc=evaluator.test(epoch)
            
            acc_test.append(acc)
            iterations.append(epoch)
            end = time.time()
        lr=args.lr
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        data = {
        "round": iterations,
        "acc": acc_test,
        }

        with open(model_dir + '/acc.json', 'w') as f:
            json.dump(data, f, indent=4)






if __name__ == "__main__":
    args = get_args(sys.argv)
    main(args)
    torch.cuda.empty_cache()
