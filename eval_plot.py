# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import random

import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader, log
from utils.dist_util import get_world_size

import logging
import sys

import torch
import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from rich.console import Console
import h5py
from args import args
from train import valids, setup, set_seed
import numpy as np


logger = logging.getLogger(__name__)

console = Console()
# key_syndrome = 'syndromes'
key_syndrome = 'image_syndromes'
key_logical_error = 'logical_errors'
pwd_trndt = '/root/Surface_code_and_Toric_code/sur_pe/'
pwd_model = '/root/ViT-pytorch/output/'



class SurDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['image_syndromes'])

    def __getitem__(self, idx):
        return self.data['image_syndromes'][idx], self.data['logical_errors'][idx]


ps = torch.linspace(0.01, 0.20, 20)

# Setup CUDA, GPU & distributed training
if args.local_rank == -1:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         timeout=timedelta(minutes=60))
    args.n_gpu = 1
args.device = device

# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
               (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

# Set seed
set_seed(args)

args, model = setup(args)

if args.fp16:
    model = amp.initialize(models=model, opt_level=args.fp16_opt_level)
    amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20


# Distributed training
if args.local_rank != -1:
    model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

accs = []

log("Eval...")

for p in ps:
    # filename_test_data = pwd_trndt + '{}_d{}_p{}_trnsz{}_imgsdr_eval_seed{}.hdf5'.format(args.c_type, args.d, format(args.p, '.3f'), 10000, args.eval_seed)
    filename_test_data = pwd_trndt + '{}_d{}_p{}_trnsz{}_imgsdr_eval_seed{}.hdf5'.format(args.c_type, args.d, format(p, '.3f'), 10000, args.eval_seed)
    log("test_data: {}".format(filename_test_data))
    with h5py.File(filename_test_data, 'r') as f:
        test_syndrome = f[key_syndrome][()]
        test_syndrome_post = np.expand_dims(test_syndrome, axis=1)
        test_logical_error = f[key_logical_error][()]
        testset = SurDataset({key_syndrome: test_syndrome_post, key_logical_error: test_logical_error})

    test_sampler = SequentialSampler(testset)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=2,
                             pin_memory=True) if testset is not None else None

    log("test_loader.dataset type: {}".format(type(test_loader.dataset)))
    log("test_loader.dataset: {}".format(test_loader.dataset))

    model_name = 'sur-{}-{}-1e7_checkpoint.bin'.format(args.d, format(0.10, '.2f'))
    log("model: {}".format(model_name))
    model.load_state_dict(torch.load(pwd_model + model_name))
    model.eval()

    acc = valids(args, model, test_loader)
    accs.append(accs)
    log("p {} acc: {}".format(format(p, '.3f'), acc))
log("accs: \n{}".format(accs))
log("Eval... Done.")
# python3 eval_plot.py --c_type sur --d 11 --name sur-11-0.10-1e7
# python3 eval_plot.py --name sur-11-0.10-1e7 --dataset sur --model_type Sur_11 --d 11 --p 0.10 --img_size 21 --eval_seed 1 --fp16 --fp16_opt_level O2
