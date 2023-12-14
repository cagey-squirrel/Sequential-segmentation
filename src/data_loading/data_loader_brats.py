import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from matplotlib import pyplot as plt
import os
import random
from src.data_loading.augmentors import augment_images
import h5py
from src.data_loading.util import crop_square, split_data_train_test, merge_patient_data


def get_brats_dataloaders(data_dir, batch_size, test_data_percentage, ensemble, split_by_patient, augment, shuffle_training, split_seed, channels):
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
    training_data, testing_data = merge_patient_data(training_data, testing_data)
    training_dataset = BRATSDataset(training_data)
    testing_dataset = BRATSDataset(testing_data)

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

    all_data_paths = []
    last_volume_name = '-1'
    all_paths_for_single_patient = []

    for slice_name in sorted(os.listdir(data_dir)):
        volume_name = slice_name.split('_')[1]

        # We encountered a new patient 
        if (volume_name != last_volume_name) and (last_volume_name != '-1'):
            all_data_paths.append(all_paths_for_single_patient)  # Save data for old patient 
            all_paths_for_single_patient = []                    # Reset data for new patient
        
        last_volume_name = volume_name
        slice_path = os.path.join(data_dir, slice_name)
        all_paths_for_single_patient.append(slice_path)

        
    # Appending data for last patient
    all_data_paths.append(all_paths_for_single_patient)

    return all_data_paths


class BRATSDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):

        path = self.data_paths[index]
        f = h5py.File(path, 'r') 
        image, mask = f['image'][()], f['mask'][()]
        image, mask = preprocess_image_and_mask(image, mask) 
        f.close()

        return image, mask, index
    

def preprocess_image_and_mask(image, mask):
    
    # cropping
    # resizing
    # normalizing
    # channel manipulation
    
    # Getting only FLAIR data
    image_precrop = image.copy() 
    mask_pre_crop = mask.copy()

    print(f'old image shape = {image_precrop.shape}')
    print(f'old image shape = {mask_pre_crop.shape}')
    image, mask = crop_square(image), crop_square(mask)
    print(f'image.min() = {image.min()} image.max() = {image.max()}')
    if image.min() == image.max():
        image *= 0
    else:
        image = (image - image.min()) / (image.max() - image.min())
    image -= 0.5
    image /= 0.2

    print(f'new image shape = {image.shape}')
    

    mask = np.logical_or(mask[..., 0], mask[..., 1], mask[..., 2])
    mask = mask[..., None]

    print(f'new mask shape = {mask.shape}')

    fig, axis = plt.subplots(2, 4)
    axis[0][0].imshow(image_precrop[..., 0])
    axis[0][1].imshow(image_precrop[..., 1])
    axis[0][2].imshow(image_precrop[..., 2])
    axis[0][3].imshow(image_precrop[..., 3])

    axis[1][0].imshow(image[..., 0])
    axis[1][1].imshow(image[..., 1])
    axis[1][2].imshow(image[..., 2])
    axis[1][3].imshow(image[..., 3])

    plt.show()



    image = np.transpose(image, (2, 0, 1))
    mask  = np.transpose(mask, (2, 0, 1))

    #print(f'image.shape = {image.shape}, mask.shape = {mask.shape}')

    return torch.tensor(image), torch.tensor(mask)

