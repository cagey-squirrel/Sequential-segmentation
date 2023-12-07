from src.data_loading.data_loader_brats import get_brats_dataloaders
from src.data_loading.data_loader_brain import get_brain_dataloaders

from src.loss_and_metrics.dice_loss import DiceLoss 
from src.loss_and_metrics.util import get_batch_metrics

import torch
from src.models.smp_unet import SmpUnet

from src.data_loading.util import info_dump, prepare_output_files, save_prediction_and_truth
from matplotlib import pyplot as plt

from datetime import datetime
from time import time
import numpy as np
import random
import os
import sys
sys.path.append("...") # Adds higher directory to python modules path.


def training(unet, training_data, device, optimizer, loss_function, epoch_num, output_file, output_train_path, params):
    unet.train()

    total_loss = 0
    metrics = torch.zeros(5, device=device)
    total_slices = 0                                                                                                                                   
    epoch_path = os.path.join(output_train_path, str(epoch_num))
    os.mkdir(epoch_path)

    with torch.set_grad_enabled(True):
        for batch_index, data in enumerate(training_data):
            optimizer.zero_grad()

            inputs, labels, names = data
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = unet(inputs)

            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            metrics += get_batch_metrics(predictions, labels, params['probability_treshold'], device)
            total_slices += labels.shape[0]

            save_prediction_and_truth(inputs, predictions, labels, epoch_path, names, epoch_num, batch_index, "training")

    total_loss /= len(training_data)
    metrics /= total_slices
    info_dump(total_loss, metrics, epoch_num, output_file, 'train')


def validation(unet, eval_data, device, loss_function, epoch_num, output_file, output_valid_path, params):
    unet.eval()

    metrics = torch.zeros(5, device=device)
    total_slices = 0
                                                                                                                                     
    epoch_path = os.path.join(output_valid_path, str(epoch_num))
    os.mkdir(epoch_path)

    with torch.set_grad_enabled(False):
        total_loss = 0
        for batch_index, data in enumerate(eval_data):

            inputs, labels, names = data
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = unet(inputs)
            
            loss = loss_function(predictions, labels)
            metrics += get_batch_metrics(predictions, labels, params['probability_treshold'], device)
            total_slices += labels.shape[0]
            total_loss += loss.item()

            save_prediction_and_truth(inputs, predictions, labels, epoch_path, names, epoch_num, batch_index, "validation")
    
    total_loss /= len(eval_data)
    metrics /= total_slices
    info_dump(total_loss, metrics, epoch_num, output_file, 'valid')

    dice_score = metrics[1]
    return dice_score


def train(params, split_seed=1302):

    torch.manual_seed(1302)
    torch.cuda.manual_seed(1302)
    random.seed(1302)
    np.random.seed(1302)

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    if params['dataset_type'] == 'brats':
        get_dataloaders = get_brats_dataloaders
        input_channels = 4
    elif params['dataset_type'] == 'brain':
        get_dataloaders = get_brain_dataloaders
        input_channels = 3

    encoder_acrhitecture = params['encoder_acrhitecture']
    attention = params['attention']
    weights = 'imagenet'
    unet = SmpUnet(encoder_acrhitecture, input_channels, 1, weights=weights, attention=attention)
    #preprocess_fn = smp.encoders.get_preprocessing_fn(encoder_acrhitecture, weights)

    loader_train, loader_valid = get_dataloaders(
                                        params['data_dir'], batch_size=params['batch_size'], 
                                        test_data_percentage=params['val_percentage'], 
                                        ensemble=params['ensemble'], 
                                        split_by_patient=params['split_by_patient'], 
                                        augment=params['augment'], shuffle_training=params['shuffle_training'],
                                        split_seed=split_seed
                                    )
    #state_dict = torch.load("src/trained_models/unet_model_time_2022-08-25 16:43:55.471303.pt")
    # unet.load_state_dict(state_dict['model_state_dict'])
    #unet.load_state_dict(state_dict)

    num_epochs = params['num_epochs']
                                                                                                    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")                                                                                                                             
    unet.to(device)
    
    optimizer = torch.optim.Adam(unet.parameters(), lr=params['learning_rate'], weight_decay=params['regularization'])
    loss_function = DiceLoss()

    folder_name = 'resnet34'
    output_dir_path, train_output_text_file, test_output_text_file = prepare_output_files(params, folder_name)
    output_valid_path = os.path.join(output_dir_path, 'valid')   
    output_train_path = os.path.join(output_dir_path, 'train')   
    
    trained_model_dir = os.path.join(params['trained_models_path'], f"{folder_name}_{str(datetime.now())[-6:]}.pt")
    os.mkdir(trained_model_dir)

    max_dice_score = 0
    epoch = 0
    patience = params["patience"]
    no_progress_epochs = 0
    best_model_state = None
    save_period = 100

    while epoch != num_epochs:

        start = time()
        current_dice_score = validation(unet, loader_valid, device, loss_function, epoch, test_output_text_file, output_valid_path, params)
        training(unet, loader_train, device, optimizer, loss_function, epoch, train_output_text_file, output_train_path, params)
        print(f'Epoch finished in {time() - start} seconds \n\n')


        # Learning rate optimization----------------------------------------------
        if epoch == 3:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
            print("Lowered lr")
        
        if epoch == 50:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
            print("Lowered lr")
        #-------------------------------------------------------------------------


        # Checking if no progress was made in last 'patience' epochs--------------
        if current_dice_score > max_dice_score:
            best_model_state = unet.state_dict()
            max_dice_score = current_dice_score
            no_progress_epochs = 0
        else:
            no_progress_epochs += 1
        
        if no_progress_epochs >= patience:
            break
        # ------------------------------------------------------------------------


        # Occasionally saving the model so we dont have to wait for it to finish
        if (epoch + 1) % save_period == 0:
            torch.save(best_model_state, os.path.join(trained_model_dir, f"model_{(epoch + 1) // save_period}.pt"))
        # ------------------------------------------------------------------------


        epoch += 1
    
    torch.save(best_model_state, os.path.join(trained_model_dir, f"best_model.pt"))
    # torch.save(unet.state_dict(), os.path.join(params['trained_models_path'], f"unet_model_time_{str(datetime.now())}.pt"))