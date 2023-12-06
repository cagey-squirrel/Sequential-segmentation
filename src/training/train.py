from src.data_loading.data_loader_brats import get_brats_dataloaders
from src.data_loading.data_loader_brain import get_brain_dataloaders

from src.data_loading.util import rename_files, info_dump, prepare_output_files, save_prediction_and_truth
from matplotlib import pyplot as plt
from torch import nn, optim
import torch
from src.loss_and_metrics.dice_loss import DiceLoss 
from src.loss_and_metrics.util import get_batch_tp_fp_fn_tn, calculate_metrics
import os
from src.models.unet import UNet
from datetime import datetime
from src.models.unet_resnext50 import UNetWithResnet50Encoder
from src.models.smp_unet import SmpUnet
import sys
sys.path.append("...") # Adds higher directory to python modules path.
import my_segmentation_models_pytorch as smp 
from time import time
import numpy as np
import random


def training(unet, training_data, device, optimizer, loss_function, epoch_num, output_file, output_dir_path, params):
    unet.train()

    num_epochs = params['num_epochs']
    total_loss = 0
    tp_fp_fn_tn = torch.zeros(4, device=device)
    total_slices = 0
    
    total_time_spent_on_predicting = 0
    total_time_spent_on_loss = 0
    total_time_spent_on_step = 0
    total_time_spent_on_backward = 0
    total_time_spent_on_adding_metrics = 0

    output_train_path = os.path.join(output_dir_path, 'train')                                                                                                                                    
    epoch_path = os.path.join(output_train_path, str(epoch_num))
    os.mkdir(epoch_path)

    with torch.set_grad_enabled(True):
        for i, data in enumerate(training_data):
            inputs, labels, names = data

            inputs, labels = inputs.to(device), labels.to(device)
            
            
            # inputs = inputs[:,None,:,:]
            # labels = labels[:,None,:,:]

            optimizer.zero_grad()

            start_predicting = time()
            predictions = unet(inputs.float())
            total_time_spent_on_predicting += time() - start_predicting

            start_predicting = time()
            loss_track_parameters = epoch_path, inputs, names, epoch_num, num_epochs, "train", i, params['probability_treshold'], params['bce_pos_weight']
            loss = loss_function(predictions, labels, loss_track_parameters)
            total_time_spent_on_loss += time() - start_predicting

            start_predicting = time()
            loss.backward()
            total_time_spent_on_backward += time() - start_predicting

            start_predicting = time()
            optimizer.step()
            total_loss += loss.item()
            total_time_spent_on_step += time() - start_predicting

            start_predicting = time()
            tp_fp_fn_tn += get_batch_tp_fp_fn_tn(predictions, labels, params['probability_treshold'], device)
            total_time_spent_on_adding_metrics += time() - start_predicting
            total_slices += labels.shape[0]

            #print(names)
            #print(loss.item())
            #if i == 2:
            #    exit(-1)

            
    tp_fp_fn_tn /= total_slices
    total_loss /= len(training_data)
    metrics = calculate_metrics(tp_fp_fn_tn)
    info_dump(total_loss, metrics, epoch_num, output_file, 'train')
    #print(f'Total time predicting in infodump = {time() - start_predicting}')
#
    #print(f'Total time predicting in training = {total_time_spent_on_predicting}')
    #print(f'Total time predicting in loss = {total_time_spent_on_loss}')
    #print(f'Total time predicting in metrics = {total_time_spent_on_adding_metrics}')
    #print(f'Total time predicting in backward = {total_time_spent_on_backward}')
    #print(f'Total time predicting in step = {total_time_spent_on_step}')


def validation(unet, eval_data, device, loss_function, epoch_num, output_file, output_dir_path, params):
    unet.eval()

    num_epochs = params['num_epochs']
    tp_fp_fn_tn = torch.zeros(4, device=device)
    total_slices = 0
    output_valid_path = os.path.join(output_dir_path, 'valid')                                                                                                                                    
    epoch_path = os.path.join(output_valid_path, str(epoch_num))
    os.mkdir(epoch_path)

    with torch.set_grad_enabled(False):
        total_loss = 0
        for i, data in enumerate(eval_data):
            inputs, labels, names = data

            inputs, labels = inputs.to(device), labels.to(device)

            predictions = unet(inputs.float())
            
            loss_track_parameters = epoch_path, inputs, names, epoch_num, num_epochs, "valid", i, params['probability_treshold'], params['bce_pos_weight']

            loss = loss_function(predictions, labels, loss_track_parameters)
            tp_fp_fn_tn += get_batch_tp_fp_fn_tn(predictions, labels, params['probability_treshold'], device)
            total_slices += labels.shape[0]

            total_loss += loss.item()

    tp_fp_fn_tn /= total_slices
    total_loss /= len(eval_data)
    metrics = calculate_metrics(tp_fp_fn_tn)
    info_dump(total_loss, metrics, epoch_num, output_file, 'valid')


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

    encoder_acrhitecture = 'resnet50' 
    weights = 'imagenet'
    unet = SmpUnet(encoder_acrhitecture, input_channels, 1, weights=weights)
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
                                                                                                    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                                                                                                                             
    unet.to(device)
    
    optimizer = optim.Adam(unet.parameters(), lr=params['learning_rate'], weight_decay=params['regularization'])
    loss_function = DiceLoss()

    output_dir_path, train_output_text_file, test_output_text_file = prepare_output_files(params)
    
    trained_model_dir = os.path.join(params['trained_models_path'], f"trained_models_{str(datetime.now())}.pt")
    os.mkdir(trained_model_dir)

    save_period = 100

    for epoch in range(num_epochs):
        start = time()
        validation(unet, loader_valid, device, loss_function, epoch, test_output_text_file, output_dir_path, params)
        #print(f'Validation finished in {time() - start} seconds \n\n')
        start = time()
        training(unet, loader_train, device, optimizer, loss_function, epoch, train_output_text_file, output_dir_path, params)
        print(f'Training finished in {time() - start} seconds \n\n')
        if epoch == 3:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
            print("Lowered lr")
        
        if epoch == 50:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
            print("Lowered lr")
        
        
   
        if (epoch + 1) % save_period == 0:
            torch.save(unet.state_dict(), os.path.join(trained_model_dir, f"unet_model__{(epoch + 1)//save_period}.pt"))
    
    # torch.save(unet.state_dict(), os.path.join(params['trained_models_path'], f"unet_model_time_{str(datetime.now())}.pt"))