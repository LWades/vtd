# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
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

import wandb

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    log(f"config: {config}")
    num_classes = 4  # 分类结果是 4 种逻辑算符 I X Z Y
    # num_classes = 10 if args.dataset == "cifar10" else 100

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    # 不进行预训练
    # model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        x = x.to(torch.float16)
        y = y.to(torch.long)
        y = torch.squeeze(y)
        with torch.no_grad():
            logits = model(x)[0]
            # log(f"logits shape: {logits.shape}")
            # log(f"y shape: {y.shape}")

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)
    wandb.log({'valid_loss': eval_losses.avg, 'valid_accuracy': accuracy})

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy


def train(args, model):
    # 假设 `model` 是您的模型实例
    if torch.cuda.is_available():
        model.to('cuda:{}'.format(args.gpu))
    else:
        raise RuntimeError("CUDA is not available. AMP requires a CUDA device.")

    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = (args.trnsz // args.train_batch_size + 1) * args.epoch
    # t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # 这里看看数据的形式和 shape，输出一些东西
            # log("\nbatch01.1.len: {}".format(len(batch)))
            batch = tuple(t.to(args.device) for t in batch)
            # log("\nbatch01.2.len {}".format(len(batch)))
            x, y = batch
            # log("x.shape: {}".format(x.shape))
            # log("x[0]: {}".format(x[0]))
            # log("y.shape: {}".format(y.shape))
            # log("y[0]: {}".format(y[0]))
            # log("y[0] type: {}".format(type(y[0])))
            # sys.exit(0)
            x = x.to(torch.float16)
            y = y.to(torch.long)
            loss = model(x, y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                wandb.log({'train_loss': losses.val})
                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy = valid(args, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                        logger.info("Best Accuracy until Now: \t%f" % best_acc)
                        wandb.log({'Best Accuracy until Now': best_acc})
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "sur"], default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16",
                                                 "Sur_3", "Sur_5", "Sur_7"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--d', type=int, default=3,
                        help='the distance of the original code, one of the labels of code')
    parser.add_argument('--c_type', type=str, default='sur',
                        help='the code type of the original code, one of the labels of code, default: %(default)s')
    parser.add_argument('--p', type=float, default=0.1,
                        help='deplorazed model error rate')
    parser.add_argument('--trnsz', type=int, default=10000000,
                        help='the size of training set')
    parser.add_argument('--epoch', type=int, default=1,
                        help='epoch')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu id')

    args = parser.parse_args()

    wandb.init(
        project="ViT for Surface code",
        name='d{}_p{}_ptc{}_ep{}'.format(args.d, args.p, CONFIGS['Sur_{}'.format(args.d)].patches['size'][0],
                                         args.epoch),
        config={
            'd': args.d,
            'physical error rate': args.p,
            'code type': args.c_type,
            'train size': args.trnsz,
            'epoch': args.epoch
        }
    )

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

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)

    wandb.finish()


if __name__ == "__main__":
    main()

# python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --fp16 --fp16_opt_level O2
# python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/ViT-B_16.npz --fp16 --fp16_opt_level O2
# nohup python3 train.py --name sur-3-0.01-1e7 --dataset sur --model_type Sur_3 --d 3 --p 0.01 --img_size 5 --fp16 --fp16_opt_level O2 > train_3_0.01.log &
# nohup python3 train.py --name sur-3-0.02-1e7 --dataset sur --model_type Sur_3 --d 3 --p 0.02 --img_size 5 --fp16 --fp16_opt_level O2 > train_3_0.02.log &
# nohup python3 train.py --name sur-3-0.03-1e7 --dataset sur --model_type Sur_3 --d 3 --p 0.03 --img_size 5 --fp16 --fp16_opt_level O2 > train_3_0.03_02.log &
# nohup python3 train.py --name sur-3-0.03-1e7 --dataset sur --model_type Sur_3 --d 3 --p 0.03 --img_size 5 --learning_rate 1e-6 --fp16 --fp16_opt_level O2 > train_3_0.03.log &
# nohup python3 train.py --name sur-3-0.04-1e7 --dataset sur --model_type Sur_3 --d 3 --p 0.04 --img_size 5 --fp16 --fp16_opt_level O2 > train_log/train_3_0.04.log &
# nohup python3 train.py --name sur-3-0.05-1e7 --dataset sur --model_type Sur_3 --d 3 --p 0.05 --img_size 5 --fp16 --fp16_opt_level O2 > train_log/train_3_0.05.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-3-0.09-1e7 --dataset sur --model_type Sur_3 --d 3 --p 0.09 --img_size 5 --fp16 --fp16_opt_level O2 > train_log/train_3_0.09.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-3-0.11-1e7 --dataset sur --model_type Sur_3 --d 3 --p 0.11 --img_size 5 --fp16 --fp16_opt_level O2 > train_log/train_3_0.11.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-3-0.12-1e7 --dataset sur --model_type Sur_3 --d 3 --p 0.12 --img_size 5 --fp16 --fp16_opt_level O2 > train_log/train_3_0.12.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-3-0.13-1e7 --dataset sur --model_type Sur_3 --d 3 --p 0.13 --img_size 5 --fp16 --fp16_opt_level O2 > train_log/train_3_0.13.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-3-0.14-1e7 --dataset sur --model_type Sur_3 --d 3 --p 0.14 --img_size 5 --fp16 --fp16_opt_level O2 > train_log/train_3_0.14.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-3-0.15-1e7 --dataset sur --model_type Sur_3 --d 3 --p 0.15 --img_size 5 --fp16 --fp16_opt_level O2 > train_log/train_3_0.15.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-3-0.16-1e7 --dataset sur --model_type Sur_3 --d 3 --p 0.16 --img_size 5 --fp16 --fp16_opt_level O2 > train_log/train_3_0.16.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-3-0.17-1e7 --dataset sur --model_type Sur_3 --d 3 --p 0.17 --img_size 5 --fp16 --fp16_opt_level O2 > train_log/train_3_0.17.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-3-0.18-1e7 --dataset sur --model_type Sur_3 --d 3 --p 0.18 --img_size 5 --fp16 --fp16_opt_level O2 > train_log/train_3_0.18.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-3-0.19-1e7 --dataset sur --model_type Sur_3 --d 3 --p 0.19 --img_size 5 --fp16 --fp16_opt_level O2 > train_log/train_3_0.19.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-3-0.20-1e7 --dataset sur --model_type Sur_3 --d 3 --p 0.20 --img_size 5 --fp16 --fp16_opt_level O2 > train_log/train_3_0.20.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-3-0.20-1e7 --dataset sur --model_type Sur_3 --d 3 --p 0.20 --img_size 5 --fp16 --fp16_opt_level O2 > train_log/train_3_0.20.log &

# nohup python3 train.py --name sur-5-0.01-1e7 --dataset sur --model_type Sur_5 --d 3 --p 0.010 --img_size 9 --fp16 --fp16_opt_level O2 > train_log/train_5_0.01.log &
# python3 train.py --name sur-5-0.01-1e7 --dataset sur --model_type Sur_5 --d 5 --p 0.010 --img_size 9 --fp16 --fp16_opt_level O2 > train_log/train_5_0.01.log
# nohup python3 train.py --name sur-5-0.01-1e7 --dataset sur --model_type Sur_5 --d 5 --p 0.010 --img_size 9 --fp16 --fp16_opt_level O2 > train_log/train_5_0.01.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-5-0.01-1e7 --dataset sur --model_type Sur_5 --d 5 --p 0.010 --img_size 9 --fp16 --fp16_opt_level O2 > train_log/train_5_0.01_1x1.log &
# nohup python3 train.py --name sur-5-0.02-1e7 --dataset sur --model_type Sur_5 --d 5 --p 0.020 --img_size 9 --fp16 --fp16_opt_level O2 > train_log/train_5_0.02.log &
# nohup python3 train.py --name sur-5-0.03-1e7 --dataset sur --model_type Sur_5 --d 5 --p 0.030 --img_size 9 --fp16 --fp16_opt_level O2 > train_log/train_5_0.03.log &
# nohup python3 train.py --name sur-5-0.04-1e7 --dataset sur --model_type Sur_5 --d 5 --p 0.040 --img_size 9 --fp16 --fp16_opt_level O2 > train_log/train_5_0.04.log &
# nohup python3 train.py --name sur-5-0.05-1e7 --dataset sur --model_type Sur_5 --d 5 --p 0.050 --img_size 9 --fp16 --fp16_opt_level O2 > train_log/train_5_0.05.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-5-0.06-1e7 --dataset sur --model_type Sur_5 --d 5 --p 0.060 --img_size 9 --fp16 --fp16_opt_level O2 > train_log/train_5_0.06.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-5-0.07-1e7 --dataset sur --model_type Sur_5 --d 5 --p 0.070 --img_size 9 --fp16 --fp16_opt_level O2 > train_log/train_5_0.07.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-5-0.08-1e7 --dataset sur --model_type Sur_5 --d 5 --p 0.080 --img_size 9 --fp16 --fp16_opt_level O2 > train_log/train_5_0.08.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-5-0.09-1e7 --dataset sur --model_type Sur_5 --d 5 --p 0.090 --img_size 9 --fp16 --fp16_opt_level O2 > train_log/train_5_0.09.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-5-0.10-1e7 --dataset sur --model_type Sur_5 --d 5 --p 0.010 --img_size 9 --fp16 --fp16_opt_level O2 > train_log/train_5_0.10.log &
# nohup python3 train.py --name sur-5-0.11-1e7 --dataset sur --model_type Sur_5 --d 5 --p 0.060 --img_size 9 --fp16 --fp16_opt_level O2 > train_log/train_5_0.06.log &
# nohup python3 train.py --name sur-7-0.01-1e7 --dataset sur --model_type Sur_7 --d 7 --p 0.010 --img_size 13 --fp16 --fp16_opt_level O2 > train_log/train_7_0.01.log &
# nohup python3 train.py --name sur-7-0.01-1x1-1e7 --dataset sur --model_type Sur_7 --d 7 --p 0.010 --img_size 13 --fp16 --fp16_opt_level O2 > train_log/train_7_0.01_1x1.log &
# nohup python3 train.py --name sur-5-0.01-1x1-1e7 --dataset sur --model_type Sur_5 --epoch 5 --d 5 --p 0.010 --img_size 9 --fp16 --fp16_opt_level O2 > train_log/train_5_0.01_1x1_ep5.log &
# nohup python3 train.py --name sur-7-0.01-1x1-1e7 --dataset sur --model_type Sur_7 --epoch 6 --d 7 --p 0.010 --img_size 13 --fp16 --fp16_opt_level O2 --gpu 1 > train_log/train_7_0.01_1x1_ep5.log &
# nohup python3 train.py --name sur-7-0.05-1x1-1e7 --dataset sur --model_type Sur_7 --epoch 7 --d 7 --p 0.050 --img_size 13 --fp16 --fp16_opt_level O2 --gpu 1 > train_log/train_7_0.05_1x1_ep7.log &
# nohup python3 train.py --name sur-7-0.11-1x1-1e7 --dataset sur --model_type Sur_7 --epoch 9 --d 7 --p 0.110 --img_size 13 --fp16 --fp16_opt_level O2 --gpu 1 > train_log/train_7_0.11_1x1_ep9.log &
# CUDA_VISIBLE_DEVICES=1 nohup python3 train.py --name sur-5-0.11-1x1-1e7 --dataset sur --model_type Sur_5 --epoch 5 --d 5 --p 0.11 --img_size 9 --fp16 --fp16_opt_level O2 > train_log/train_5_0.11_1x1_ep5.log &
# nohup python3 train.py --name sur-5-0.05-1x1-1e7 --dataset sur --model_type Sur_5 --epoch 5 --d 5 --p 0.050 --img_size 9 --fp16 --fp16_opt_level O2 > train_log/train_5_0.05_1x1_ep5.log &
# nohup python3 train.py --name sur-5-0.08-1x1-1e7 --dataset sur --model_type Sur_5 --epoch 5 --d 5 --p 0.080 --img_size 9 --fp16 --fp16_opt_level O2 --gpu 0 > train_log/train_5_0.08_1x1_ep5.log &
# nohup python3 train.py --name sur-5-0.17-1x1-1e7-ep8 --dataset sur --model_type Sur_5 --epoch 8 --d 5 --p 0.170 --img_size 9 --fp16 --fp16_opt_level O2 > train_log/train_5_0.17_1x1_ep8.log &
