from __future__ import print_function
import os, time, sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import torch.optim as optim
from torch.optim import lr_scheduler

from datasets.Cifar100LT import get_cifar100

from models.resnet import *

from train.train import train_base
from train.validate import valid_base

from utils.config import *
from utils.common import hms_string

from utils.logger import logger
import copy

args = parse_args()
reproducibility(args.seed)
args.logger = logger(args)

def load_model(args, model, testloader, N_SAMPLES_PER_CLASS):
    if args.pretrained_pth is not None:
        pth = args.pretrained_pth
    else:
        pth = f'pretrained/cifar100/IR={args.imb_ratio}/best_model_{args.cur_stage}.pt'

    state_dict  = torch.load(pth)
    model.load_state_dict(state_dict)

    # model = torch.load(pth)
    test_criterion = nn.CrossEntropyLoss()  # For test, validation
    test_loss, test_acc, test_cls = valid_base(testloader, model, test_criterion, N_SAMPLES_PER_CLASS,
                                             num_class=args.num_class, mode='test Valid')

    args.logger(f'Loaded performance...', level=1)
    args.logger(f'[Test ]\t Acc:\t{test_acc:.4f}', level=2)
    args.logger(f'[Stats]\tMany:\t{test_cls[0]:.4f}\tMedium:\t{test_cls[1]:.4f}\tFew:\t{test_cls[2]:.4f}', level=2)

    return model

def main():
    print(f'==> Preparing imbalanced CIFAR-100')

    trainset, testset = get_cifar100(os.path.join(args.data_dir, 'cifar100/'), args)
    N_SAMPLES_PER_CLASS = trainset.img_num_list

    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  drop_last=False, pin_memory=True, sampler=None)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                 pin_memory=True)

    # Model
    print("==> creating {}".format(args.network))
    model = resnet34(num_classes=100, pool_size=4).cuda()

    model = load_model(args, model, testloader, N_SAMPLES_PER_CLASS)

if __name__ == '__main__':
    main()