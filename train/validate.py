from utils.accuracy import AverageMeter, accuracy
from progress.bar import Bar
import torch
import numpy as np
import time

def horizontal_flip_aug(model):
    def aug_model(inputs):
        logits = model(inputs)
        h_logits =  model(inputs.flip(3))
        return (logits + h_logits) / 2

    return aug_model

def valid_base(valloader, model, criterion, per_class_num, num_class=100, mode='Test Stats', test_aug=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if test_aug:
        model = horizontal_flip_aug(model)

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    classwise_correct = torch.zeros(num_class)
    classwise_num = torch.zeros(num_class)
    section_acc = torch.zeros(3)

    all_preds = np.zeros(len(valloader.dataset))
    with torch.no_grad():
        for batch_idx, data_tuple in enumerate(valloader):
            inputs = data_tuple[0].cuda(non_blocking=True)
            targets = data_tuple[1].cuda(non_blocking=True)
            indexs = data_tuple[2]

            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            pred_label = outputs.max(1)[1]

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))


            # measure accuracy and record loss

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # classwise prediction

            all_preds[indexs] = pred_label.cpu().numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                         'Loss: {loss:.4f}'.format(
                batch=batch_idx + 1,
                size=len(valloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
            )
            bar.next()
        bar.finish()
        # Major, Neutral, Minor

        all_targets = np.array(valloader.dataset.targets)
        pred_mask = (all_targets == all_preds).astype(np.float)
        for i in range(num_class):
            class_mask = np.where(all_targets == i)[0].reshape(-1)
            classwise_correct[i] += pred_mask[class_mask].sum()
            classwise_num[i] += len(class_mask)

        classwise_acc = (classwise_correct / classwise_num)

        per_class_num = torch.tensor(per_class_num)
        many_pos = torch.where(per_class_num > 100)
        med_pos = torch.where((per_class_num <= 100) & (per_class_num >= 20))
        few_pos = torch.where(per_class_num < 20)
        section_acc[0] = classwise_acc[many_pos].mean()
        section_acc[1] = classwise_acc[med_pos].mean()
        section_acc[2] = classwise_acc[few_pos].mean()

    return losses.avg, top1.avg, section_acc