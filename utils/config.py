import argparse, torch, os, random
import numpy as np
from datetime import datetime


def parse_args(run_type='terminal'):
    parser = argparse.ArgumentParser(description='Python Training')

    # Dataset options
    parser.add_argument('--data_dir', default='data/')
    parser.add_argument('--dataset', default='cifar100', help='Dataset: cifar100')
    parser.add_argument('--num_class', type=int, default=100, help='class number')
    parser.add_argument('--imb_ratio', type=int, default=100, help='Imbalance ratio for data')

    # Optimization options
    parser.add_argument('--network', default='resnet34', help='Network: resnet34')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='train batchsize')

    parser.add_argument('--cur_stage', default='stage1', help='stage1 feature learning, stage2 classifier learning')
    # feature extractor learning parameters
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--wd', default=5e-3, type=float, help='weight decay factor for optimizer')
    parser.add_argument('--nesterov', action='store_true', help="Utilizing Nesterov")

    # classifier learning parameters
    parser.add_argument('--finetune_epoch', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--finetune_lr', default=0.0005, type=float, help='learnign rate decay')
    parser.add_argument('--finetune_wd', default=0, type=float, help='weight decay factor for optimizer')
    parser.add_argument('--label_smooth', default=0, type=float, help='label smoothing')

    # Pretrained model path
    parser.add_argument('--pretrained_pth', default=None, type=str, help='The pretrained model path')

    # Checkpoints save dir
    parser.add_argument('--out', default='output', help='Directory to output the result')

    # Miscs
    parser.add_argument('--workers', type=int, default=16, help='# workers')
    parser.add_argument('--seed', type=str, default='None', help='manual seed')
    parser.add_argument('--gpu', default=None, type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    now = datetime.now()
    time = f'{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}'

    args.out = f'{args.out}/{args.dataset}_IR={args.imb_ratio}_{args.cur_stage}/{time}/'

    if args.gpu:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args


def reproducibility(seed):
    if seed == 'None':
        return
    else:
        seed = int(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
