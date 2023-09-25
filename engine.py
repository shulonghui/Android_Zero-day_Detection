
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import argparse

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score




def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    args: argparse.ArgumentParser.parse_args = None):
    #losses = []
    #accuracies = []
    # TODO fix this for finetuning
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1 if args.debug else 300
    if args.update_temperature:
        model.module.update_temperature()

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)



        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
    
        if not math.isfinite(loss_value):
            logging.error("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        #添加
        #losses.append(loss.item())
        #accuracy = (outputs.argmax(1) == targets).sum().item() / targets.size(0)
        #accuracies.append(accuracy)


        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    '''plt.figure()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    loss_curve_file = f"loss_curve_epoch{epoch}.png"
    plt.savefig(loss_curve_file)

    # 绘制和保存准确率曲线
    plt.figure()
    plt.plot(accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    accuracy_curve_file = f"accuracy_curve_epoch{epoch}.png"
    plt.savefig(accuracy_curve_file)
    '''

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

#global_counter = 0

@torch.no_grad()
def evaluate(data_loader, model, device, header = 'Test:'):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()

    #true_labels = []
    #predicted_labels = []

    for images, target in metric_logger.log_every(data_loader, 200, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
