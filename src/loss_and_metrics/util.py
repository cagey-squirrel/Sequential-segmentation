import torch 
import numpy as np
from src.loss_and_metrics.hausdorff_distance import averaged_hausdorff_distance
from src.loss_and_metrics.f_measure_metrics import intersection_over_union, dice, precision, recall, area_under_curve


def get_batch_tp_fp_fn_tn(predictions, labels, treshold, device):
    '''
    Calculates tp, fp, fn and tn metrics for a batch
    Inputs:
        -predictions (pytorch tensor): tensor representing image segmentation with shape (image_heigh, image_width)
            each value in tensor represents the possibility that that pixel is a cancer cell
        -labels(pytorch tensor): tensor representing true segmentations with shape (image_heigh, image_width)
            cancer cells have pixel values of 1 while background (not cancer) cells have a pixel value of 0
        -treshold (float): defines a treshold used to find cancer cells: all pixels in tensor predictions which have 
            a higher value than treshold are declared cancer cells
    '''
    predictions_probabilities = predictions.flatten()
    true_labels = labels.flatten()

    predicted_true = predictions_probabilities > treshold
    predicted_false = torch.logical_not(predicted_true)
    labeled_true = true_labels
    labeled_false = torch.logical_not(labeled_true)
    
    tp = (predicted_true * labeled_true).sum()
    fp = (predicted_true * labeled_false).sum()
    fn = (predicted_false * labeled_true).sum()
    tn = (predicted_false * labeled_false).sum()

    return torch.tensor([tp, fp, fn, tn], device=device)


def calculate_metrics(tp_fp_fn_tn):
    '''
    Calculates IoU, dice score, precision, recall, AuC
    '''
    tp, fp, fn, tn = tp_fp_fn_tn

    iou = intersection_over_union(tp, fp, fn)
    dice_score = dice(tp, fp, fn)
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    auc = area_under_curve(tp, fp, fn, tn)
    
    return iou, dice_score, prec, rec, auc

