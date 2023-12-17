import torch.nn as nn
import torch
from src.data_loading.util import save_prediction_and_truth


class DiceLoss(nn.Module):

    def __init__(self, aggregation):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0
        self.aggregation = aggregation
    
    def forward(self, y_pred, y_true):
        return self.basic_dice_loss(y_pred, y_true)


    def basic_dice_loss(self, y_pred, y_true):
        '''
        This function calculates basic dice loss as descriobed in paper
        '''
        # Sum of all probabilities inside intersection between prediction and truth
        cancer_intersection = (y_pred * y_true).sum(axis=(1, 2, 3))
        cancer_union = (y_pred).sum(axis=(1, 2, 3)) + y_true.sum(axis=(1, 2, 3))

        score = (2. * cancer_intersection + self.smooth) / (cancer_union + self.smooth)
        loss = 1. - score

 
        if self.aggregation == 'mean':
            return loss.mean()
        if self.aggregation == 'sum':
            return loss.sum()
