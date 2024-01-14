import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from matplotlib import pyplot as plt
import os
import random
from src.data_loading.augmentors import augment_images
import h5py
from src.data_loading.util import crop_square, split_data_train_test, merge_patient_data, map_index_to_patient
import pickle


def get_brats_3x_dataloaders(data_dir, batch_size, test_data_percentage, ensemble, split_by_patient, augment, shuffle_training, split_seed, channels):
    '''
    Dataloader for BRATS2020 dataset
    Inputs:
        -data_dir: directory which contains subdirectories: patient1, patient2...
        -batch_size: size of one batch for training and testing
        -test_data_percentage: percentage of data which will be used for testing                                                                                                
    Returns:
        -Dataloader instances for training and test dataset
    '''
    torch.manual_seed(1302)

    all_data_paths = get_all_data_paths_brats(data_dir)
    
    training_data, testing_data = split_data_train_test(all_data_paths, test_data_percentage, ensemble, split_by_patient, split_seed)
    train_mapping = map_index_to_patient(training_data)
    valid_mapping = map_index_to_patient(testing_data)
    
    print(f'Total number of scans = {len(training_data) + len(testing_data)}')
    training_dataset = BRATSDataset(training_data, train_mapping)
    testing_dataset = BRATSDataset(testing_data, valid_mapping)

    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=shuffle_training)
    testing_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

    return training_loader, testing_loader


def augment_data(training_data):
    '''
    This function is used for augmenting training data
    Augmentation is done on ~half of the training dataset
    Training data is extended in place for all new augmented images
    Input:
        -training data (list): list containing all training data
    Returns:
        -None : training data is extended in place so there is no return value
    '''

    augmented_training_data = []
        
    for v in training_data:
        scan, mask, name = v

        if random.uniform(0, 1) > 0.5:
            augmented_scan, augmented_mask = augment_images(scan, mask)
            augmented_training_data.append((augmented_scan, augmented_mask, "augmented_" + name))    

    training_data.extend(augmented_training_data)


def get_all_data_paths_brats(data_dir):
    '''
    Returns paths to all slices in data_dir
    Slices are grouped by patient
    patients are recognizable by volume name

    Inputs:
        data_dir: path to dir containing all slices
    Returns:
        List: [ [p1_slice1_path, p1_slice2_path, p1_slice3_path], [p2_slice1_path, p2_slice2_path, p2_slice3_path] ...]
    '''

    with open('populated_dict.pkl', 'rb') as f:
        populated_dict = pickle.load(f)

    all_data_paths = []
    last_volume_name = '-1'
    all_paths_for_single_patient = []

    for slice_name in sorted(os.listdir(data_dir)):
        volume_name = slice_name.split('_')[1]

        # We encountered a new patient 
        if (volume_name != last_volume_name) and (last_volume_name != '-1'):
            start, end, total_scans = populated_dict[last_volume_name]
            all_data_paths.append(all_paths_for_single_patient[start:end])  # Save data for old patient 
            all_paths_for_single_patient = []                    # Reset data for new patient
        
        last_volume_name = volume_name
        slice_path = os.path.join(data_dir, slice_name)
        all_paths_for_single_patient.append(slice_path)

        
    # Appending data for last patient
    start, end, total_scans = populated_dict[last_volume_name]
    all_data_paths.append(all_paths_for_single_patient[start:end])

    return all_data_paths


class BRATSDataset(Dataset):
    def __init__(self, data_paths, mapping):
        self.data_paths = data_paths
        self.mapping = mapping

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index):

        patient_index, path_index = self.mapping[index]
        patient_paths = self.data_paths[patient_index]

        path_current = patient_paths[path_index]
        if path_index == 0:
            path_before = path_current
        else:
            path_before = patient_paths[path_index - 1]
            
        if path_index == len(patient_paths) - 1:
            path_after = path_current
        else:
            path_after = patient_paths[path_index + 1]
        

        f = h5py.File(path_current, 'r') 
        image_current, mask_current = f['image'][()], f['mask'][()]
        image_current, mask_current = preprocess_image(image_current), preprocess_mask(mask_current) 
        f.close()

        f = h5py.File(path_before, 'r') 
        image_before, mask_before = f['image'][()], f['mask'][()]
        image_before = preprocess_image(image_before)
        f.close()

        f = h5py.File(path_after, 'r') 
        image_after, mask_after = f['image'][()], f['mask'][()]
        image_after = preprocess_image(image_after)
        f.close()

        scans = torch.stack([image_before, image_current, image_after])

        return scans, mask_current, path_current


def preprocess_mask(mask):
    
    # cropping
    # resizing
    # normalizing
    # channel manipulation

    mask = crop_square(mask)

    mask = np.logical_or(mask[..., 0], mask[..., 1], mask[..., 2])
    mask = mask[..., None]

    mask  = np.transpose(mask, (2, 0, 1))

    return torch.tensor(mask)



def preprocess_image(image):
    
    # cropping
    # resizing
    # normalizing
    # channel manipulation

    image = crop_square(image)

    if image.min() == image.max():
        image *= 0
        print(f'empty')
    else:
        image = (image - image.min()) / (image.max() - image.min())
    image -= 0.5
    
    image = np.transpose(image, (2, 0, 1))

    return torch.tensor(image, dtype=torch.float)

