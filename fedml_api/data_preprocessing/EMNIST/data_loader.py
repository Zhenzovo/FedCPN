import json
import logging
import os
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from .datasets import EMNIST_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def load_partition_data_emnist_custom(dataset, data_dir, partition_method, partition_alpha, client_num_in_total,
                                     client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)

    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)

    data_local_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}

    for client_idx in range(client_number):
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size, dataidxs)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local

    logging.info("finished the customized partial data")
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

def partition_data(dataset, datadir, partition, n_nets, alpha):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_EMNIST_data(datadir)
    
    # # 处理EMNIST图像转置问题
    # X_train = X_train.permute(0, 2, 1)  # 调整通道位置
    # X_test = X_test.permute(0, 2, 1)
    
    n_train = X_train.shape[0]

    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero":
        min_size = 0
        K = 62
        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

def _data_transforms_emnist():
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(3),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        Cutout(16)
    ])

    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(3),
        transforms.Pad(4),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return train_transform, valid_transform

def load_EMNIST_data(datadir):
    train_transform, test_transform = _data_transforms_emnist()
    train_dataset = EMNIST_truncated(datadir, train=True, download=True, transform=train_transform)
    test_dataset = EMNIST_truncated(datadir, train=False, download=True, transform=test_transform)

    X_train, y_train = train_dataset.data, train_dataset.target
    X_test, y_test = test_dataset.data, test_dataset.target

    return (X_train, y_train, X_test, y_test)

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts

def get_dataloader_EMNIST(datadir, train_bs, test_bs, dataidxs=None):
    transform_train, transform_test = _data_transforms_emnist()
    train_dataset = EMNIST_truncated(datadir, train=True, dataidxs=dataidxs, transform=transform_train, download=True)
    test_dataset = EMNIST_truncated(datadir, train=False, transform=transform_test, download=True)

    train_dl = data.DataLoader(dataset=train_dataset, batch_size=train_bs, shuffle=True,
                               num_workers=8, pin_memory=True, drop_last=False)
    test_dl = data.DataLoader(dataset=test_dataset, batch_size=test_bs, shuffle=False,
                              num_workers=8, pin_memory=True, drop_last=False)

    return train_dl, test_dl

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    return get_dataloader_EMNIST(datadir, train_bs, test_bs, dataidxs)