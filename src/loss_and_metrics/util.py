import torch 
import numpy as np
from loss_and_metrics.hausdorff_distance import averaged_hausdorff_distance
from loss_and_metrics.f_measure_metrics import intersection_over_union, dice, sensitivity, specificity, aoc


def calculate_metrics_for_batch(predictions_batch, labels_batch, treshold=0.5):
    '''
    Wrapper for whole batch arounfd calculate_metrics method
    '''
    # First we calculate the sum of all metrics
    sum_metrics = [[0, 0] for _ in range(6)]

    for predictions, labels in zip(predictions_batch, labels_batch):
        metrics = calculate_metrics(predictions, labels, treshold)    
        add_new_metrics(sum_metrics, metrics)

    return sum_metrics

def calculate_metrics(predictions, labels, treshold):
    '''
    Calculates IoU, dice score, sensitivity, specificity, aoc and hausdorff loss
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

    IoU_score = intersection_over_union(tp, fp, fn)
    dice_score = dice(tp, fp, fn)
    sens_score = sensitivity(tp, fn)
    spec_score = specificity(tn, fp)                                                                 
    aoc_score = aoc(tp, fp, fn, tn)

    predicted_true = (predictions > 0.5).squeeze()
    labeled_true = (labels > 0.5).squeeze()

    predictions_cpu = predicted_true.cpu()
    labeled_cpu = labeled_true.detach().cpu().numpy()
    hausdorff_loss = averaged_hausdorff_distance(predictions_cpu, labeled_cpu)

    return IoU_score, dice_score, sens_score, spec_score, aoc_score, hausdorff_loss

def add_new_metrics(metrics_list, new_metrics):
    '''
    This function adds new metrics to the overall sums in the metrics list
    We keep count of sum of each metric and the count of pictures used for that sum
    (since different metrics sometimes use different number of pictures)

    Input:
        - metrics_list (list): list containing sums of metrics and number of pictures used for that sum
          [(IoU_sum, IoU_cnt), (dice_sum, dice_cnt), (sens_sum, sens_cnt), (spec_sum, spec_cnt), (aoc_sum, aoc_cnt), (haus_sum, haus_cnt)]
    '''
        
    for i in range(len(new_metrics)):
        metrics_list[i][0] += new_metrics[i][0]
        metrics_list[i][1] += new_metrics[i][1]
    
def average_metrics(metrics_list):
    '''
    Calculates average for all metrics based on metric sum and count

    Input:
        - metrics_list (list): list containing sums of metrics and number of pictures used for that sum
          [(IoU_sum, IoU_cnt), (dice_sum, dice_cnt), (sens_sum, sens_cnt), (spec_sum, spec_cnt), (aoc_sum, aoc_cnt), (haus_sum, haus_cnt)]
    Returns:
        - averages (list): list containing average value for each metric based on its sum and count
    '''
    averages = []

    for metric_sum, metric_cnt in metrics_list:
        metric_average = metric_sum / (metric_cnt + 1)
        averages.append(metric_average)
    
    return averages