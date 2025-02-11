from __future__ import print_function
import os, time

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

best_acc = 0  # best test accuracy
many_best, med_best, few_best = 0, 0, 0
best_model = None

def train_stage1(args, model, trainloader, testloader, N_SAMPLES_PER_CLASS):
    global best_acc, many_best, med_best, few_best, best_model

    train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)
    test_criterion = nn.CrossEntropyLoss()  # For test, validation
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=args.nesterov)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min = 0)

    best_model = None
    test_accs = []
    start_time = time.time()
    for epoch in range(args.epochs):

        train_loss, train_acc = train_base(trainloader, model, optimizer, train_criterion)
        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        test_loss, test_acc, test_cls = valid_base(testloader, model, test_criterion, N_SAMPLES_PER_CLASS,
                                                 num_class=args.num_class, mode='test Valid')

        if best_acc <= test_acc:
            best_acc = test_acc
            many_best = test_cls[0]
            med_best = test_cls[1]
            few_best = test_cls[2]

            best_model = copy.deepcopy(model)
        test_accs.append(test_acc)

        args.logger(f'Epoch: [{epoch + 1} | {args.epochs}]', level=1)

        args.logger(f'[Train]\tLoss:\t{train_loss:.4f}\tAcc:\t{train_acc:}', level=2)
        args.logger(f'[Test ]\tLoss:\t{test_loss:.4f}\tAcc:\t{test_acc:.4f}', level=2)
        args.logger(f'[Stats]\tMany:\t{test_cls[0]:.4f}\tMedium:\t{test_cls[1]:.4f}\tFew:\t{test_cls[2]:.4f}', level=2)
        args.logger(
            f'[Best ]\tAcc:\t{np.max(test_accs):.4f}\tMany:\t{100 * many_best:.4f}\tMedium:\t{100 * med_best:.4f}\tFew:\t{100 * few_best:.4f}',
            level=2)
        args.logger(f'[Param]\tLR:\t{lr:.8f}', level=2)

    end_time = time.time()

    file_name = os.path.join(args.out, 'best_model_stage1.pth')
    torch.save(best_model, file_name)

    # Print the final results
    args.logger(f'Finish Training Stage 1...', level=1)
    args.logger(f'Final performance...', level=1)
    args.logger(f'best bAcc (test):\t{np.max(test_accs)}', level=2)
    args.logger(f'best statistics:\tMany:\t{many_best}\tMed:\t{med_best}\tFew:\t{few_best}', level=2)
    args.logger(f'Training Time: {hms_string(end_time - start_time)}', level=1)


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

    train_stage1(args, model, trainloader, testloader, N_SAMPLES_PER_CLASS)

if __name__ == '__main__':
    main()