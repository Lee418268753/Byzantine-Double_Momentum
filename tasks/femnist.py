import json
import os
from collections import defaultdict
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
from torch import nn
import torch.nn.functional as F
# from utils.language_utils import word_to_indices, letter_to_vec
from dataloader.dataloader import iidLoader,dirichletLoader
from utils.utils import read_data,read_dir
class CNNFemnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 7, padding=3)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.out = nn.Linear(64 * 7 * 7, 62)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.flatten(1)
        # return self.dense2(self.act(self.dense1(x)))
        return self.out(x)

def get_femnist_model():
    return CNNFemnist()


# Dataset
def get_femnist_data(root_dir,noniid=False, train=True):
    femnist_stats = {
        "mean": (0.1307,),
        "std": (0.3081,),
    }
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(femnist_stats['mean'], femnist_stats['std']),
        ]
    )
    dataset = FEMNIST(              noniid=noniid,
                                    train=train,
                                    # download=True,
                                    transform=transform
                                    )
    return dataset


# Dataloader
def get_train_loader(root_dir, n_workers, alpha=1.0, batch_size=32, noniid=False):
    dataset = get_femnist_data(root_dir=root_dir,noniid=noniid, train=True)
    if not noniid:
        loader = iidLoader(size=n_workers, dataset=dataset, bsz=batch_size)
    else:
        # loader = dirichletLoader(size=n_workers, dataset=dataset, alpha=alpha, bsz=batch_size)
        loader = iidLoader(size=n_workers, dataset=dataset, bsz=batch_size)
    return loader


def get_test_loader(root_dir, batch_size, noniid=False):
    dataset = get_femnist_data(root_dir=root_dir, noniid=noniid,train=False)
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

class FEMNIST(Dataset):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """

    def __init__(self, noniid,train=True, transform=None, target_transform=None):
        super(FEMNIST, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train



        if noniid:
            train_clients, train_groups, train_data_temp, test_data_temp = read_data("../datasets/femnist/train-niid",
                                                                                 "../datasets/femnist/test-niid")
        else:
            train_clients, train_groups, train_data_temp, test_data_temp = read_data("../datasets/femnist/train-iid",
                                                                                "../datasets/femnist/test-iid")
        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i in range(len(train_clients)):
                # if i == 100:
                #     break
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.targets = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(train_clients)):
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    test_data_y.append(cur_y[j])
            self.data = test_data_x
            self.targets = test_data_y

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = np.array([img])
        return torch.from_numpy((0.5-img)/0.5).float(), target

    def __len__(self):
        return len(self.data)

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")


