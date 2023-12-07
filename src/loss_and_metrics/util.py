import torch 
import numpy as np
from src.loss_and_metrics.f_measure_metrics import intersection_over_union, dice, precision, recall, area_under_curve


def get_batch_metrics(predictions, labels, treshold, device):
    '''
    Calculates metrics for a batch
    Metrics calculated are: IoU, dice score, precision, recall, AuC
    Returns these metrics as pytorch tensor
    Inputs:
        -predictions (pytorch tensor): tensor representing image segmentation with shape (image_heigh, image_width)
            each value in tensor represents the possibility that that pixel is a cancer cell
        -labels(pytorch tensor): tensor representing true segmentations with shape (image_heigh, image_width)
            cancer cells have pixel values of 1 while background (not cancer) cells have a pixel value of 0
        -treshold (float): defines a treshold used to find cancer cells: all pixels in tensor predictions which have 
            a higher value than treshold are declared cancer cells
    '''

    tp, fp, fn, tn = calculate_tp_fp_fn_tn(predictions, labels, treshold)
    metrics = calculate_metrics(tp, fp, fn, tn)

    metrics = metrics.sum(1)
    return metrics
    

def calculate_tp_fp_fn_tn(predictions, labels, treshold):

    predicted_true = predictions > treshold
    predicted_false = torch.logical_not(predicted_true)
    labeled_true = labels
    labeled_false = torch.logical_not(labeled_true)

    tp = (predicted_true * labeled_true).sum(axis=(1, 2, 3))
    fp = (predicted_true * labeled_false).sum(axis=(1, 2, 3))
    fn = (predicted_false * labeled_true).sum(axis=(1, 2, 3))
    tn = (predicted_false * labeled_false).sum(axis=(1, 2, 3))

    return tp, fp, fn, tn


def calculate_metrics(tp, fp, fn, tn):
    '''
    Calculates IoU, dice score, precision, recall, AuC
    Returns them in a tensor
    '''

    iou = intersection_over_union(tp, fp, fn)
    dice_score = dice(tp, fp, fn)
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    auc = area_under_curve(tp, fp, fn, tn)
    
    return torch.stack([iou, dice_score, prec, rec, auc])

