'''
This script is used to calculate f_measure metrics i.e. metrics which depend on confusion matrics (number of TP, TN, FP, FN)
Further descriptions of these metrics can be found in this paper https://arxiv.org/ftp/arxiv/papers/2202/2202.05273.pdf
'''


def intersection_over_union(tp, fp, fn):
    '''
    Calculates the ratio between intersection and union in two sets
    We only calculate and average the values for images where cancer truly exists
    This images have sum of tp and fn greater than one
    If image has no cancer then its metric should not be used when calculating average
    Inputs:
        -tp (int): number of true positives
        -fp (int): number 0f false positives
        -fn (int): number of false negatives
    Returns:
        - ratio (float): ratio between intersection and union in two sets
        - has_cancer (int): 1 if there is cancer on image, 0 if there is not
    '''
    smooth = 1
    IoU = (tp + smooth) / (tp + fp + fn + smooth)
    return IoU


def dice(tp, fp, fn):
    '''
    Calculates the dice score for 2 sets
    We only calculate and average the values for images where cancer truly exists
    This images have sum of tp and fn greater than one
    If image has no cancer then its metric should not be used when calculating average
    Inputs:
        -tp (int): number of true positives
        -fp (int): number 0f false positives
        -fn (int): number of false negatives
    Returns:
        - ratio (float): dice score
        - has_cancer (int): 1 if there is cancer on image, 0 if there is not
    '''
    smooth = 1
    dice_score = (2*tp + smooth) / (2*tp + fp + fn + smooth)
    return dice_score


def precision(tp, fp):
    '''
    Calculates precision 
    How many cancer cells we correctly predicted vs how many we predicted in total
    '''
    smooth = 1
    sens = (tp + smooth) / (tp + fp + smooth)
    return sens


def recall(tp, fn):
    '''
    Calculated recall
    How many cancer cells we correctly predicted vs how many there actually are
    '''
    smooth = 1
    rec = (tp + smooth) / (tp + fn + smooth)
    return rec


# Area under the ROC curve:
def area_under_curve(tp, fp, fn, tn):
    '''
    Calculates the area under roc curve
    '''
    smooth = 1
    auc = 1 - 1/2 * (fp / (fp + tn + smooth) + fn / (fn + tp + smooth))
    return auc
