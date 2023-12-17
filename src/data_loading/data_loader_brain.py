import torch
from torch import Generator
from torch.utils.data import random_split, Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from datetime import datetime
import os
import random
from src.data_loading.augmentors import augment_images, rotate_images, shear_images
import h5py
from src.data_loading.util import crop_square, split_data_train_test, merge_patient_data, group_patients_into_tensors



def get_brain_dataloaders(data_dir, batch_size, test_data_percentage, ensemble, split_by_patient, augment, shuffle_training, split_seed, channels):
    '''
    Loads images from data_dir
    Inputs:
        -data_dir: directory which contains subdirectories: patient1, patient2...
        -batch_size: size of one batch for training and testing
        -test_data_percentage: percentage of data which will be used for testing                                                                                                                           
    Returns:
        -Dataloader instances for training and test dataset
    '''
    torch.manual_seed(1302)

    all_data = get_brain_data(data_dir)
    #all_data = preprocess_data(all_data)
    
    
    training_data, testing_data = split_data_train_test(all_data, test_data_percentage, ensemble, split_by_patient, split_seed)
    training_data, testing_data = merge_patient_data(training_data, testing_data)

    if augment:
        print(f'Data size pre augmentation = {len(training_data)}')
        augment_data(training_data)
        print(f'Data size after augmentation = {len(training_data)}')

    training_dataset = BrainCancerDataset(training_data, channels)
    testing_dataset = BrainCancerDataset(testing_data, channels)

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

        augmented_scan, augmented_mask = rotate_images(scan, mask)
        augmented_training_data.append((augmented_scan, augmented_mask, "rotated_" + name))

        augmented_scan, augmented_mask = shear_images(scan, mask)
        augmented_training_data.append((augmented_scan, augmented_mask, "sheared_" + name))    

    training_data.extend(augmented_training_data)


def get_brain_data(data_dir):
    '''
    Gets images from data_dir directory
    data_dir contains directories for each patient separatelly
    Inputs:
        data_dir: path to dir containing dirs of images for each patient
    Returns:
        List: [ [(p1_img1, p1_msk1), (p1_img2, p1_msk2)], [(p2_img1, p2_msk1), (p2_img2, p2_msk2)]...]
    '''
    

    data = []
    for patient_dir in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, patient_dir)):
            continue
        image_names = sorted(os.listdir(os.path.join(data_dir, patient_dir)))

        current_patient_data = []

        image_is_next = True
        current_img_pair = []
        for image_name in image_names:
            image_path = os.path.join(data_dir, patient_dir, image_name)
            image = read_tif_image(image_path)

            if image_is_next:
                scan = image
                scan = crop_square(scan)
                scan = (scan - scan.min(axis=(0, 1))) / (scan.max(axis=(0, 1)) - scan.min(axis=(0, 1)))
                scan -= 0.5
                #scan /= 2
                scan = np.transpose(scan, (2, 0, 1))
                scan = torch.tensor(scan, dtype=torch.float)

                current_img_pair.append(scan)
                image_is_next = False
            
            else:
                mask = image
                mask[mask < 1e-3] = 0
                mask[mask > 0] = 1
                mask = mask[..., None]
                mask = crop_square(mask)
                mask = np.transpose(mask, (2, 0, 1))
                mask = torch.tensor(mask)

                current_img_pair.append(mask)
                current_img_pair.append(image_name)
                image_is_next = True
                current_patient_data.append(current_img_pair)
                current_img_pair = []
        
        #print(f'scan = {scan.shape}, mask = {mask.shape}')
        data.append(current_patient_data)
    
    return data


def read_tif_image(path):
    '''
    Opens .tif image from path
    Input:
        -path: path to image
    Returns:
        -image: np array representing image
    '''
    image = Image.open(path)
    image = np.array(image)

    return image




class BrainCancerDataset(Dataset):
    def __init__(self, data, channels='single'):
        self.data = data
        self.channels = channels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

    
        image, mask, name = self.data[index]

        if self.channels == 'single':
            return image, mask, name
        
             
        if index == 0:
            image_before = image
        else:
            image_before = self.data[index-1][0]

        if index == (len(self.data)-1):
            image_after = image
        else:
            image_after = self.data[index+1][0]

        #images = torch.cat([image_before, image, image_after])
        images = torch.stack([image_before, image, image_after])
        
        return images, mask, name
    

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







