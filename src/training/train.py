from src.data_loading.data_loader_brats    import get_brats_dataloaders
from src.data_loading.data_loader_brain    import get_brain_dataloaders
from src.data_loading.data_loader_patient  import get_patient_dataloaders
from src.data_loading.data_loader_3x       import get_3x_dataloaders
from src.data_loading.data_loader_brats_3x import get_brats_3x_dataloaders
from src.data_loading.data_loader_3d       import get_3d_dataloaders


from src.loss_and_metrics.dice_loss import DiceLoss 
from src.loss_and_metrics.util import get_batch_metrics

import torch
from src.models.smp_unet import SmpUnet
from src.models.unet_3d import ResidualUNet3D, UNet3D

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


    aggregation = params['aggregation']
    total_loss, total_slices, slices_in_batch, last_slices_in_batch = 0, 0, 0, 1
    metrics = torch.zeros(5, device=device)                                                                                                        
    epoch_path = os.path.join(output_train_path, str(epoch_num))
    os.mkdir(epoch_path)
    #print(training_data.dataset.data)
    with torch.set_grad_enabled(True):
        for batch_index, data in enumerate(training_data):
            
            
            inputs, labels, names = data

            if '3d' in params['dataset_type']:
                inputs = torch.permute(inputs, (0, 2, 1, 3, 4))


            if '3x' in params['dataset_type']:
                
                batches, seq, channels, width, height = inputs.shape
                inputs = inputs.view(batches * seq, channels, width, height)

            inputs, labels = inputs.to(device), labels.to(device)
            predictions = unet(inputs)

            if '3d' in params['dataset_type']:
                predictions = torch.permute(predictions, (0, 2, 1, 3, 4))
            
            loss = loss_function(predictions, labels)
 
            if aggregation == 'mean':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            elif aggregation == 'sum':
                loss.backward()
                slices_in_batch += labels.shape[0]
                if slices_in_batch >= 128 or batch_index == len(training_data) - 1:
                    
                    
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / slices_in_batch * last_slices_in_batch
                    optimizer.step()
                    
                    last_slices_in_batch = slices_in_batch
                    slices_in_batch = 0
                    optimizer.zero_grad()
            
            if '3d' in params['dataset_type']:
                inputs = torch.permute(inputs, (0, 2, 1, 3, 4))
                batch, slices, channels, width, height = inputs.shape
                inputs = inputs.view(batch * slices, channels, width, height)
                batch, slices, channels, width, height = predictions.shape
                predictions = predictions.view(batch * slices, channels, width, height)
                labels = labels.view(batch * slices, channels, width, height)
            
            
            total_loss += loss.detach().item()
            metrics += get_batch_metrics(predictions, labels, params['probability_treshold'], device)
            total_slices += labels.shape[0]

            save_prediction_and_truth(inputs, predictions, labels, epoch_path, names, epoch_num, batch_index, "training", params['dataset_type'])
    
    if aggregation == 'mean':
        total_loss /= len(training_data)
    elif aggregation == 'sum':
        total_loss /= total_slices
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * last_slices_in_batch
        
    metrics /= total_slices
    info_dump(total_loss, metrics, epoch_num, output_file, 'train')

    if params['dataset_type'] == 'patient':
        training_data.schuffle_data()


def validation(unet, eval_data, device, loss_function, epoch_num, output_file, output_valid_path, params):
    unet.eval()

    metrics = torch.zeros(5, device=device)
    total_slices = 0
    aggregation = params['aggregation']                                                                                                                                 
    epoch_path = os.path.join(output_valid_path, str(epoch_num))
    os.mkdir(epoch_path)

    with torch.set_grad_enabled(False):
        total_loss = 0
        for batch_index, data in enumerate(eval_data):

            inputs, labels, names = data

            if '3d' in params['dataset_type']:
                inputs = torch.permute(inputs, (0, 2, 1, 3, 4))

            if '3x' in params['dataset_type']:
                batches, seq, channels, width, height = inputs.shape
                inputs = inputs.view(batches * seq, channels, width, height)

            inputs, labels = inputs.to(device), labels.to(device)
            predictions = unet(inputs)

            if '3d' in params['dataset_type']:
                predictions = torch.permute(predictions, (0, 2, 1, 3, 4))
            
            #print(f'predictions = {predictions.shape} labels = {labels.shape}')
            loss = loss_function(predictions, labels)
            
            if '3d' in params['dataset_type']:
                inputs = torch.permute(inputs, (0, 2, 1, 3, 4))
                batch, slices, channels, width, height = inputs.shape
                inputs = inputs.view(batch * slices, channels, width, height)
                batch, slices, channels, width, height = predictions.shape
                predictions = predictions.view(batch * slices, channels, width, height)
                labels = labels.view(batch * slices, channels, width, height)
                
            metrics += get_batch_metrics(predictions, labels, params['probability_treshold'], device)
            total_slices += labels.shape[0]
            total_loss += loss.detach().item()

            save_prediction_and_truth(inputs, predictions, labels, epoch_path, names, epoch_num, batch_index, "validation", params['dataset_type'])
    
    if aggregation == 'mean':
        total_loss /= len(eval_data)
    elif aggregation == 'sum':
        total_loss /= total_slices
   
    metrics /= total_slices
    info_dump(total_loss, metrics, epoch_num, output_file, 'valid')

    dice_score = metrics[1]
    return dice_score


def train(params, split_seed=1302):

    torch.manual_seed(1302)
    torch.cuda.manual_seed(1302)
    random.seed(1302)
    np.random.seed(1302)

    if not 'ma' in params['encoder_acrhitecture']:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    #torch.autograd.set_detect_anomaly(True)
    if params['parallel']:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    channels = params['channels']
    if params['dataset_type'] == 'brats':
        get_dataloaders = get_brats_dataloaders
        input_channels = 4
    elif params['dataset_type'] == 'brain':
        get_dataloaders = get_brain_dataloaders
        if channels == 'single':
            input_channels = 3
        elif channels == 'triple':
            input_channels = 9
    elif params['dataset_type'] == 'patient':
        get_dataloaders = get_patient_dataloaders
        input_channels = 3
    elif params['dataset_type'] == '3x':
        get_dataloaders = get_3x_dataloaders
        input_channels = 3
    elif params['dataset_type'] == 'brats_3x':
        get_dataloaders = get_brats_3x_dataloaders
        input_channels = 4
    elif params['dataset_type'] == '3d':
        get_dataloaders = get_3d_dataloaders
        input_channels = 3

    encoder_acrhitecture = params['encoder_acrhitecture']
    attention = params['attention']
    aggregation = params['aggregation']
    weights = 'imagenet'
    layer_normalization_type = params['layer_normalization_type']
    
    
    unet = SmpUnet(encoder_acrhitecture, input_channels, 1, weights=weights, attention=attention, layer_normalization_type=layer_normalization_type, device=device)
    #unet = ResidualUNet3D(input_channels, out_channels=1)
    #unet = UNet3D(input_channels, out_channels=1)

    if torch.cuda.device_count() > 1 and params['parallel']:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        unet = torch.nn.DataParallel(unet)
    unet.to(device)
    print(sum(p.numel() for p in unet.parameters() if p.requires_grad))#; exit(0)
    
    
    #preprocess_fn = smp.encoders.get_preprocessing_fn(encoder_acrhitecture, weights)

    #if "triple" in attention: 
    #    params['shuffle_training'] = False

    loader_train, loader_valid = get_dataloaders(
                                        params['data_dir'], batch_size=params['batch_size'], 
                                        test_data_percentage=params['val_percentage'], 
                                        ensemble=params['ensemble'], 
                                        split_by_patient=params['split_by_patient'], 
                                        augment=params['augment'], shuffle_training=params['shuffle_training'],
                                        split_seed=split_seed, channels=channels
                                    )
    #state_dict = torch.load("src/trained_models/unet_model_time_2022-08-25 16:43:55.471303.pt")
    # unet.load_state_dict(state_dict['model_state_dict'])
    #unet.load_state_dict(state_dict)

    num_epochs = params['num_epochs']    


    
    
    optimizer = torch.optim.Adam(unet.parameters(), lr=params['learning_rate'], weight_decay=params['regularization'])
    loss_function = DiceLoss(aggregation)

    folder_name = 'resnet34-patience-100'
    output_dir_path, train_output_text_file, test_output_text_file = prepare_output_files(params, folder_name)
    output_valid_path = os.path.join(output_dir_path, 'valid')   
    output_train_path = os.path.join(output_dir_path, 'train')   
    
    trained_model_dir = os.path.join(params['trained_models_path'], f"{folder_name}_{str(datetime.now())[-6:]}.pt")
    os.mkdir(trained_model_dir)

    max_dice_score = torch.zeros(1, device=device)
    epoch = 0
    patience = params["patience"]
    no_progress_epochs = 0
    best_model_state = None
    save_period = 50

    while epoch != num_epochs:

        start = time()
        current_dice_score = validation(unet, loader_valid, device, loss_function, epoch, test_output_text_file, output_valid_path, params)
        training(unet, loader_train, device, optimizer, loss_function, epoch, train_output_text_file, output_train_path, params)
        print(f'Epoch finished in {str(time() - start)[:5]} seconds, lr = {str(optimizer.param_groups[0]["lr"])[:8]},  dice-score = {str(current_dice_score.item())[:5]} while max is {str(max_dice_score.item())[:5]} \n\n')


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