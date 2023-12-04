import torch.nn as nn
import torch
from src.data_loading.data_loader import save_prediction_and_truth


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0
    
    def forward(self, y_pred, y_true, loss_track_parameters):

        return basic_dice_loss(y_pred, y_true, loss_track_parameters)


def background_dice(y_pred, y_true, loss_track_parameters):
    '''
    This method is modified version of dice loss
    Dice loss is calculated and averaged for both background and cancer cells
    '''
    losses = []

    assert y_pred.size() == y_true.size()
    batch_size = y_true.shape[0]
    batch_loss = 0
    image_num = loss_track_parameters[-1]
    mode = loss_track_parameters[-2]
    epoch = loss_track_parameters[3]
    num_epochs = loss_track_parameters[-3]
    for i in range(batch_size):
        smooth = 1.
        
        # Probabilities of pixel being a cancer cell
        predictions_probabilities = y_pred[i].flatten()
        # Probabilities of pixel not being a cancer cell (its just background)
        background_probabilities = 1 - predictions_probabilities

        # Prediction and background prediction labels: 
        # pixels with probability 0.5 and more are declared cancerous(1)
        # pixels with probability less than 0.5 are declared as background(0)
 

        # True labels are labels of cancer cells (1)
        # Background labels are pixels wihout cancer cells (0)
        true_labels = y_true[i].flatten()
        background_labels = torch.logical_not(true_labels)
        
        # Sum of all probabilities inside intersection between prediction and truth
        cancer_intersection = (predictions_probabilities * true_labels).sum()
        cancer_union = (predictions_probabilities).sum() + true_labels.sum()
        
        # Sum of all probabilities inside intersection between predicted background and true background
        background_intersection = (background_probabilities * background_labels).sum()
        background_union = (background_probabilities).sum() + background_labels.sum()
        

        cancer_score = (2. * cancer_intersection + smooth) / (cancer_union + smooth)
        cancer_loss = 1. - cancer_score

        background_score = (2. * background_intersection + smooth) / (background_union + smooth)
        background_loss = 1. - background_score 

        loss = (cancer_loss + background_loss) / 2
        batch_loss += loss

        if image_num < 3:
            losses.append(loss)
        elif mode == "valid" and epoch == num_epochs - 1:
            losses.append(loss)
    
    if image_num < 3:
        # save_prediction_and_truth(path, inputs, y_pred, y_true, epoch_no, losses, image_num)
        save_prediction_and_truth(loss_track_parameters, losses, y_pred, y_true)
        #print(f"cancer loss = {cancer_loss}")
        #print(f"background loss = {background_loss}")
    elif mode == "valid" and epoch == num_epochs - 1:
          save_prediction_and_truth(loss_track_parameters, losses, y_pred, y_true)

    individual_dsc_loss = batch_loss / batch_size                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    return individual_dsc_loss
    

def splitting_background_dice(y_pred, y_true, loss_track_parameters):
    '''
    This method is modified version of dice loss
    It splits the prediction into regions of pixels based on if pixels probability of being a cancer cell is over or under 0.5
    Dice loss is calculated and averaged for both background and cancer cells
    '''
    image_num = loss_track_parameters[-1]
    mode = loss_track_parameters[-2]
    epoch = loss_track_parameters[3]
    num_epochs = loss_track_parameters[-3]
    losses = []
    smooth = 1.

    assert y_pred.size() == y_true.size()
    batch_size = y_true.shape[0]
    batch_loss = 0
    for i in range(batch_size):
        
        
        # Probabilities of pixel being a cancer cell
        predictions_probabilities = y_pred[i].flatten()
        # Probabilities of pixel not being a cancer cell (its just background)
        background_probabilities = 1 - predictions_probabilities

        # Prediction and background prediction labels: 
        # pixels with probability 0.5 and more are declared cancerous(1)
        # pixels with probability less than 0.5 are declared as background(0)
        prediction_labels = predictions_probabilities >= 0.5
        background_prediction_labels = torch.logical_not(prediction_labels)

        # True labels are labels of cancer cells (1)
        # Background labels are pixels wihout cancer cells (0)
        true_labels = y_true[i].flatten()
        background_labels = torch.logical_not(true_labels)
        
        # Sum of all probabilities inside intersection between prediction and truth
        cancer_intersection = (prediction_labels * predictions_probabilities * true_labels).sum()
        cancer_union = (prediction_labels * predictions_probabilities).sum() + true_labels.sum()
        
        # Sum of all probabilities inside intersection between predicted background and true background
        background_intersection = (background_prediction_labels * background_probabilities * background_labels).sum()
        background_union = (background_prediction_labels * background_probabilities).sum() + background_labels.sum()
        

        cancer_score = (2. * cancer_intersection + smooth) / (cancer_union + smooth)
        cancer_loss = 1. - cancer_score

        background_score = (2. * background_intersection + smooth) / (background_union + smooth)
        background_loss = 1. - background_score 

        loss = (cancer_loss + background_loss) / 2
        batch_loss += loss

        if image_num < 3:
            losses.append(loss)
        elif mode == "valid" and epoch == num_epochs - 1:
            losses.append(loss)

    if image_num < 3:
        # save_prediction_and_truth(path, inputs, y_pred, y_true, epoch_no, losses, image_num)
        save_prediction_and_truth(loss_track_parameters, losses, y_pred, y_true)
        #print(f"cancer loss = {cancer_loss}")
        #print(f"background loss = {background_loss}")
    elif mode == "valid" and epoch == num_epochs - 1:
        save_prediction_and_truth(loss_track_parameters, losses, y_pred, y_true)
        

    individual_dsc_loss = batch_loss / batch_size                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    return individual_dsc_loss


def splitted_dice(y_pred, y_true, loss_track_parameters):
    '''
    This method is modified version of dice loss
    It splits the prediction into regions of pixels based on if pixels probability of being a cancer cell is over or under 0.5
    Dice loss is calculated and averaged for both background and cancer cells
    '''

    epoch_path, inputs, names, epoch, num_epochs, mode, image_num, treshold = loss_track_parameters

    losses = []
    smooth = 1.

    assert y_pred.size() == y_true.size()
    batch_size = y_true.shape[0]
    batch_loss = 0
    for i in range(batch_size):
        
        # Probabilities of pixel being a cancer cell
        predictions_probabilities = y_pred[i].flatten()
        # Probabilities of pixel not being a cancer cell (its just background)

        # Prediction and background prediction labels: 
        # pixels with probability above treshold are declared cancerous(1)
        prediction_labels = predictions_probabilities >= treshold

        # True labels are labels of cancer cells (1)
        # Background labels are pixels wihout cancer cells (0)
        true_labels = y_true[i].flatten()
        
        # Sum of all probabilities inside intersection between prediction and truth
        cancer_intersection = (prediction_labels * predictions_probabilities * true_labels).sum()
        cancer_union = (prediction_labels * predictions_probabilities).sum() + true_labels.sum()
        
        cancer_score = (2. * cancer_intersection + smooth) / (cancer_union + smooth)
        cancer_loss = 1. - cancer_score

        loss = cancer_loss
        batch_loss += loss

        if image_num < 3:
            losses.append(loss)
        elif mode == "valid" and epoch == num_epochs - 1:
            losses.append(loss)

    if image_num < 3:
        save_prediction_and_truth(loss_track_parameters, losses, y_pred, y_true)
    elif mode == "valid" and epoch == num_epochs - 1:
        save_prediction_and_truth(loss_track_parameters, losses, y_pred, y_true)
        

    individual_dsc_loss = batch_loss / batch_size                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    return individual_dsc_loss

def basic_bce_loss(y_pred, y_true, loss_track_parameters):
    '''
    This function calculates basic dice loss as descriobed in paper
    '''

    epoch_path, inputs, names, epoch, num_epochs, mode, image_num, treshold, bce_pos_weight = loss_track_parameters
    
    losses = []
    smooth = 1.

    #y_pred = y_pred.squeeze()
    #y_true = y_true[:, None, :, :]
    #assert y_pred.size() == y_true.size()
    batch_size = y_true.shape[0]
    batch_loss = 0
    all_metrics = []
    for i in range(batch_size):
        
        # Probabilities of pixel being a cancer cell
        predictions_probabilities = y_pred[i].flatten()

        # True labels are labels of cancer cells (1)
        true_labels = y_true[i].float().flatten()
        
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([bce_pos_weight]).to("cuda:1"))
        cancer_loss = bce_loss(predictions_probabilities, true_labels)

        loss = cancer_loss
        batch_loss += loss

        if image_num < 3:
            losses.append(loss)
        elif mode == "valid" and (((epoch + 1) % 100) == 0):
            losses.append(loss)
    
    if image_num < 3:
        save_prediction_and_truth(loss_track_parameters, losses, y_pred, y_true)
    elif mode == "valid" and (((epoch + 1) % 100) == 0):
        save_prediction_and_truth(loss_track_parameters, losses, y_pred, y_true)

    individual_dsc_loss = batch_loss / batch_size     

    return individual_dsc_loss 

def basic_dice_loss(y_pred, y_true, loss_track_parameters):
    '''
    This function calculates basic dice loss as descriobed in paper
    '''

    epoch_path, inputs, names, epoch, num_epochs, mode, image_num, treshold, _ = loss_track_parameters

    losses = []
    smooth = 1.

    #y_pred = y_pred.squeeze()
    y_true = y_true[:, None, :, :]
    #assert y_pred.size() == y_true.size()
    batch_size = y_true.shape[0]
    batch_loss = 0
    all_metrics = []
    for i in range(batch_size):
        
        # Probabilities of pixel being a cancer cell
        predictions_probabilities = y_pred[i].flatten()

        # True labels are labels of cancer cells (1)
        true_labels = y_true[i].flatten()
        
        # Sum of all probabilities inside intersection between prediction and truth
        cancer_intersection = (predictions_probabilities * true_labels).sum()
        cancer_union = (predictions_probabilities).sum() + true_labels.sum()
        
        cancer_score = (2. * cancer_intersection + smooth) / (cancer_union + smooth)
        cancer_loss = 1. - cancer_score

        loss = cancer_loss
        batch_loss += loss

        if image_num < 3:
            losses.append(loss)
        elif mode == "valid" and (((epoch + 1) % 100) == 0):
            losses.append(loss)
    
    if image_num < 3:
        save_prediction_and_truth(loss_track_parameters, losses, y_pred, y_true)
    elif mode == "valid" and (((epoch + 1) % 100) == 0):
        save_prediction_and_truth(loss_track_parameters, losses, y_pred, y_true)

    individual_dsc_loss = batch_loss / batch_size     

    return individual_dsc_loss 