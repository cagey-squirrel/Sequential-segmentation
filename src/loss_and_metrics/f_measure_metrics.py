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
    has_cancer = int((tp+fn) > 0)
    # has_cancer = 1
    smooth = 1
    IoU = (tp + smooth) / (tp + fp + fn + smooth)
    return (IoU * has_cancer, has_cancer)


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
    has_cancer = int((tp+fn) > 0)
    # has_cancer = 1
    smooth = 1
    dice_score = (2*tp + smooth) / (2*tp + fp + fn + smooth)
    return (dice_score * has_cancer, has_cancer)

def sensitivity(tp, fn):
    '''
    Calculates sensitivity i.e. how many positives we found divided by how many positives there actually were
    We only calculate and average the values for images where cancer truly exists
    This images have sum of tp and fn greater than one
    If image has no cancer then its metric should not be used when calculating average
    Inputs:
        -tp (int): number of true positives
        -fn (int): number of false negatives
    Returns:
        - ratio (float): sensitivity
        - has_cancer (int): 1 if there is cancer on image, 0 if there is not
    '''
    smooth = 1
    has_cancer = int((tp+fn) > 0)
    # has_cancer = 1
    sens = (tp + smooth) / (tp + fn + smooth)
    return (sens * has_cancer, has_cancer)

def specificity(tn, fp):
    '''
    Calculates specificity i.e. how many negatives we found divided by how many negatives there actually were
    Since negatives usually declare background in image segmentation problems, in cases such as ours where 
    background takes up most of the image this metric should always have values close to 1
    Inputs:
        -tn (int): number of true negatives
        -fp (int): number of false positives
    Returns:
        - ratio (float): specificity
        - has_cancer (int): always returns 1 (since it is not important for this metric)
    '''
    smooth = 1
    specificity = (tn + smooth) / (tn + fp + smooth)
    has_cancer = 1
    return (specificity, 1)

# Area under the ROC curve:
def aoc(tp, fp, fn, tn):
    '''
    Calculates the area under roc curve
    Inputs:
        -tp (int): number of true positives
        -fp (int): number of false positives
        -fn (int): number of false negatives
        -tn (int): number of true negatives
    Returns:
        - ratio (float): area under ROC curve
        - has_cancer (int): always returns 1 (since it is not important for this metric)
    '''
    smooth = 1
    has_cancer = 1
    aoc = 1 - 1/2 * (fp / (fp + tn + smooth) + fn / (fn + tp + smooth))
    return (aoc, 1)
