from src.data_loading.data_loader import get_dataloaders, rename_files, info_dump, prepare_output_files, save_prediction_and_truth
from matplotlib import pyplot as plt
from torch import nn, optim
import torch
from src.loss_and_metrics.dice_loss import DiceLoss 
from src.loss_and_metrics.util import calculate_metrics_for_batch, add_new_metrics, average_metrics
import os
from src.models.unet import UNet
from datetime import datetime
from src.models.unet_resnext50 import UNetWithResnet50Encoder
from src.models.smp_unet import SmpUnet
import sys
sys.path.append("...") # Adds higher directory to python modules path.
import my_segmentation_models_pytorch as smp 


def training(unet, training_data, device, optimizer, loss_function, epoch_num, output_file, output_dir_path, params):
    unet.train()

    num_epochs = params['num_epochs']
    total_loss = 0
    metrics = [[0, 0] for _ in range(6)]

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

            predictions = unet(inputs.float())

            loss_track_parameters = epoch_path, inputs, names, epoch_num, num_epochs, "train", i, params['probability_treshold'], params['bce_pos_weight']
            loss = loss_function(predictions, labels, loss_track_parameters)
            new_metrics = calculate_metrics_for_batch(predictions, labels, params['probability_treshold'])


            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            add_new_metrics(metrics, new_metrics)
    
    metrics = average_metrics(metrics)
    total_loss /= len(training_data)

    info_dump(total_loss, metrics, epoch_num, output_file, 'train')
    print('\n')

def validation(unet, eval_data, device, loss_function, epoch_num, output_file, output_dir_path, params):
    unet.eval()

    num_epochs = params['num_epochs']
    metrics = [[0, 0] for _ in range(6)]
    output_valid_path = os.path.join(output_dir_path, 'valid')                                                                                                                                    
    epoch_path = os.path.join(output_valid_path, str(epoch_num))
    os.mkdir(epoch_path)

    counter = 0

    with torch.set_grad_enabled(False):
        total_loss = 0
        for i, data in enumerate(eval_data):
            inputs, labels, names = data

            inputs, labels = inputs.to(device), labels.to(device)

            # inputs = inputs[:,None,:,:]
            # labels = labels[:,None,:,:]

            predictions = unet(inputs.float())
            
            loss_track_parameters = epoch_path, inputs, names, epoch_num, num_epochs, "valid", i, params['probability_treshold'], params['bce_pos_weight']

            loss = loss_function(predictions, labels, loss_track_parameters)
            new_metrics = calculate_metrics_for_batch(predictions, labels, params['probability_treshold'])

            total_loss += loss.item()
            add_new_metrics(metrics, new_metrics)
            counter += 1

    metrics = average_metrics(metrics)
    total_loss /= counter
  
    info_dump(total_loss, metrics, epoch_num, output_file, 'valid')

def train(params):
    # rename_files(params['data_dir'])


    print(f'starting')
    encoder_acrhitecture = 'resnet34' 
    weights = 'imagenet'
    unet = SmpUnet(encoder_acrhitecture, 3, 1, weights=weights)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_acrhitecture, weights)
    preprocessing_fn(torch.randn(224, 224, 3))
    print(f'exiting')
    exit(-1)


    loader_train, loader_valid = get_dataloaders(
                                        params['data_dir'], batch_size=params['batch_size'], 
                                        test_data_percentage=params['val_percentage'], 
                                        normalization=params['normalization'], 
                                        standard_histogram_path=params['standard_histogram_path'], 
                                        ensemble=params['ensemble'], 
                                        split_by_patient=params['split_by_patient'], 
                                        augment=params['augment']
                                    )
    #state_dict = torch.load("src/trained_models/unet_model_time_2022-08-25 16:43:55.471303.pt")
    # unet.load_state_dict(state_dict['model_state_dict'])
    #unet.load_state_dict(state_dict)

    num_epochs = params['num_epochs']
                                                                                                    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")                                                                                                                             
    unet.to(device)
    
    optimizer = optim.Adam(unet.parameters(), lr=params['learning_rate'], weight_decay=params['regularization'])
    loss_function = DiceLoss()

    output_dir_path, train_output_text_file, test_output_text_file = prepare_output_files(params)
    
    trained_model_dir = os.path.join(params['trained_models_path'], f"trained_models_{str(datetime.now())}.pt")
    os.mkdir(trained_model_dir)

    for epoch in range(num_epochs):
        validation(unet, loader_valid, device, loss_function, epoch, test_output_text_file, output_dir_path, params)
        training(unet, loader_train, device, optimizer, loss_function, epoch, train_output_text_file, output_dir_path, params)
        
        if epoch == 10:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
            print("Promenjen lr")
        
        
   
        if (epoch + 1) % 100 == 0:
            torch.save(unet.state_dict(), os.path.join(trained_model_dir, f"unet_model__{(epoch + 1)//100}.pt"))
    
    # torch.save(unet.state_dict(), os.path.join(params['trained_models_path'], f"unet_model_time_{str(datetime.now())}.pt"))