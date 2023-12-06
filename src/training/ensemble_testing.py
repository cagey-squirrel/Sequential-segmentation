from src.data_loading.data_loader_brats import get_brats_dataloaders
from src.data_loading.data_loader_brain import get_brain_dataloaders

from src.data_loading.util import rename_files, info_dump, prepare_output_files, save_prediction_and_truth
from matplotlib import pyplot as plt
from torch import nn, optim
import torch
from src.loss_and_metrics.dice_loss import DiceLoss 
from src.loss_and_metrics.util import calculate_metrics, get_batch_tp_fp_fn_tn
import os
from src.models.unet import UNet
from datetime import datetime
from src.models.unet_resnext50 import UNetWithResnet50Encoder
import numpy as np


def validation(unet, eval_data, device, loss_function, epoch_num, output_file, output_dir_path, params, ensembled_models):

    num_ensembled_models = len(ensembled_models)

    num_epochs = params['num_epochs']
   
    total_loss = [0 for _ in range(num_ensembled_models )]
    output_valid_path = os.path.join(output_dir_path, 'valid')                                                                                                                                    

    tp_fp_fn_tn = np.zeros((num_ensembled_models, 4))
    total_slices = 0

    with torch.set_grad_enabled(False):
        
        for j, data in enumerate(eval_data):
            inputs, labels, names = data

            inputs, labels = inputs.to(device), labels.to(device)

            #inputs = inputs[:,None,:,:]
            #labels = labels[:,None,:,:]

            #averaged_predictor = torch.zeros(labels.shape).to(device)

            for i, ensemble_model in enumerate(ensembled_models):
                ensemble_model.to(device)
                ensemble_model.eval()
                predictions = ensemble_model(inputs.float())
            
                dirname = dirname = os.path.join(output_valid_path, str(i))
                loss_track_parameters = dirname, inputs, names, epoch_num, num_epochs, "valid", j, params['probability_treshold'], params['bce_pos_weight']

                loss = loss_function(predictions, labels, loss_track_parameters)

                tp_fp_fn_tn[i] += get_batch_tp_fp_fn_tn(predictions, labels, params['probability_treshold'])
                total_loss[i] += loss.item()
                total_slices += labels.shape[0]
              

    for i, loss_val in enumerate(total_loss):
        tp_fp_fn_tn[i] /= total_slices
        total_loss[i] /= len(eval_data)
  
        info_dump(total_loss[i], tp_fp_fn_tn[i], epoch_num, output_file, 'valid')


def test_ensemble(params):


    if params['dataset_type'] == 'brats':
        get_dataloaders = get_brats_dataloaders
    elif params['dataset_type'] == 'brain':
        get_dataloaders = get_brain_dataloaders

    loader_train, loader_valid = get_dataloaders(
                                        params['data_dir'], batch_size=params['batch_size'], 
                                        test_data_percentage=params['val_percentage'], 
                                        normalization=params['normalization'], 
                                        standard_histogram_path=params['standard_histogram_path'], 
                                        ensemble=params['ensemble'], 
                                        split_by_patient=params['split_by_patient'], 
                                        augment=params['augment']
                                    )
    unet = UNet()
    #unet = UNetWithResnet50Encoder()

    num_epochs = params['num_epochs']
                                                                                                    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")                                                                                                                             
    unet.to(device)
    
    optimizer = optim.Adam(unet.parameters(), lr=params['learning_rate'], weight_decay=params['regularization'])
    loss_function = DiceLoss()

    output_dir_path, train_output_text_file, test_output_text_file = prepare_output_files(params)

    trained_models = []
    trained_models_dir = '/home/workstation/Documents/repos/Brain-Cancer-Segmentation/src/trained_models/last resnet'

    for i, model_name in enumerate(os.listdir(trained_models_dir)):
        state_dict = torch.load(os.path.join(trained_models_dir, model_name))
        # unet = UNet()
        unet = UNetWithResnet50Encoder()
        unet.load_state_dict(state_dict)
        trained_models.append(unet)
        directory_name_valid = os.path.join(output_dir_path, 'valid')
        epoch_directory_name_valid = os.path.join(directory_name_valid, str(i))
        os.mkdir(epoch_directory_name_valid)
    
    dirname = os.path.join(output_dir_path, 'valid', 'ensembled')
    os.mkdir(dirname)

    validation(None, loader_valid, device, loss_function, 0, test_output_text_file, output_dir_path, params, trained_models)
 