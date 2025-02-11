from __future__ import print_function

import time
import numpy as np
import torch
import torch.nn as nn

from utils.accuracy import AverageMeter, accuracy
from progress.bar import Bar
import copy, time


def train_base(trainloader, model, optimizer, criterion):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()

    bar = Bar('Training', max=len(trainloader))

    for batch_idx, data_tuple in enumerate(trainloader):
        inputs = data_tuple[0]
        targets = data_tuple[1]

        data_time.update(time.time() - end)

        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))

        # measure accuracy and record loss
        prec1,prec5 = accuracy(outputs, targets, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))

        # record
        batch_time.update(time.time() - end)
        end = time.time()

        # plot
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                     'Loss: {loss:.4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
        )
        bar.next()
    bar.finish()

    return losses.avg, top1.avg