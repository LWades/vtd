import logging
import sys

import torch
import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from rich.console import Console
import h5py

logger = logging.getLogger(__name__)

console = Console()
# key_syndrome = 'syndromes'
key_syndrome = 'image_syndromes'
key_logical_error = 'logical_errors'
pwd_trndt = '/root/Surface_code_and_Toric_code/sur_pe/'


def log(info):
    console.print(info)


class SurDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['image_syndromes'])

    def __getitem__(self, idx):
        return self.data['image_syndromes'][idx], self.data['logical_errors'][idx]


def sur_preprocess(data):
    # 先不进行归一化试一下
    syndromes = data['image_syndromes']
    log("shape of syndromes before expand_dims: {}".format(syndromes.shape))
    syndromes = np.expand_dims(syndromes, axis=1)
    log("shape of syndromes after expand_dims: {}".format(syndromes.shape))
    # 处理分类类别
    logical_errors = data['logical_errors']
    logical_output = torch.empty(logical_errors.shape[0], dtype=torch.long)
    log("logical_errors preprocess(00-11 -> 0-3) start...")
    for i, logical_error in enumerate(logical_errors):
        if logical_error[0] == 0 and logical_error[1] == 0:
            logical_output[i] = 0
        if logical_error[0] == 0 and logical_error[1] == 1:
            logical_output[i] = 1
        if logical_error[0] == 1 and logical_error[1] == 0:
            logical_output[i] = 2
        if logical_error[0] == 1 and logical_error[1] == 1:
            logical_output[i] = 3
    log("logical_errors preprocess(00-11 -> 0-3) end.")
    res = {'image_syndromes': syndromes, 'logical_errors': logical_output}
    return res


# 准备数据等
def get_loader(args):
    log("args: {}".format(args))
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "sur":
        # --------------numpy-----------------
        # traindata = np.load("./data/{}_d{}_p{}_trnsz{}_imgsdr.npz".format(args.c_type, args.d, format(args.p, '.3f'), args.trnsz))
        # traindata = sur_preprocess(traindata)
        # trainset = SurDataset(traindata)
        # testdata = np.load("./data/{}_d{}_p{}_trnsz{}_imgsdr_eval.npz".format(args.c_type, args.d, format(args.p, '.3f'), 10000))
        # testdata = sur_preprocess(testdata)
        # testset = SurDataset(testdata)
        # --------------h5py-----------------
        filename_train_data = pwd_trndt + '{}_d{}_p{}_trnsz{}_imgsdr_seed{}.hdf5'.format(args.c_type, args.d, format(args.p, '.3f'), args.trnsz, args.train_seed)
        log("train_data: {}".format(filename_train_data))
        with h5py.File(filename_train_data, 'r') as f:
            train_syndrome = f[key_syndrome][()]
            log("type train_syndrome: {}".format(type(train_syndrome)))
            train_syndrome_post = np.expand_dims(train_syndrome, axis=1)
            train_logical_error = f[key_logical_error][()]
            log("train_logical_error.shape: {}".format(train_logical_error.shape))
            log("train_logical_error[0]: {}".format(train_logical_error[0]))
            log("train_logical_error[0] type: {}".format(type(train_logical_error[0])))
            trainset = SurDataset({key_syndrome: train_syndrome_post, key_logical_error: train_logical_error})
        filename_test_data = pwd_trndt + '{}_d{}_p{}_trnsz{}_imgsdr_eval_seed{}.hdf5'.format(args.c_type, args.d, format(args.p, '.3f'), 10000, args.eval_seed)
        log("test_data: {}".format(filename_test_data))
        with h5py.File(filename_test_data, 'r') as f:
            test_syndrome = f[key_syndrome][()]
            test_syndrome_post = np.expand_dims(test_syndrome, axis=1)
            test_logical_error = f[key_logical_error][()]
            testset = SurDataset({key_syndrome: test_syndrome_post, key_logical_error: test_logical_error})
    else:
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=2,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=2,
                             pin_memory=True) if testset is not None else None

    log("train_loader.dataset type: {}".format(type(train_loader.dataset)))
    log("test_loader.dataset type: {}".format(type(test_loader.dataset)))
    log("train_loader.dataset: {}".format(train_loader.dataset))
    log("test_loader.dataset: {}".format(test_loader.dataset))

    return train_loader, test_loader
