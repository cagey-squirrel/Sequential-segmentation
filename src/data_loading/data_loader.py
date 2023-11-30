import torch
from torch import Generator
from torch.utils.data import random_split, Dataset, DataLoader
import os
from PIL import Image
from PIL.ImageOps import grayscale
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import json 
from datetime import datetime
import os
import random
from data_loading.augmentors import augment_images
import h5py
import psutil
from collections import defaultdict


def load_data(data_dir, batch_size, test_data_percentage, normalization, standard_histogram_path, ensemble, split_by_patient, augment):
    '''
    Loads images from data_dir
    Inputs:
        -data_dir: directory which contains subdirectories: patient1, patient2...
        -batch_size: size of one batch for training and testing
        -test_data_percentage: percentage of data which will be used for testing
        -normalization: String which sets the type of normalization: 'Basic' or 'Nyul'
        -standard_histogram_path: path to histogram used for Nyul normalization
                                  if there is no histogram on this path then it will be made there during Nyul normalization                                                                                                                                
    Returns:
        -Dataloader instances for training and test dataset
    '''
    torch.manual_seed(1302)

    all_data_paths = get_all_data_paths(data_dir)
    
    training_data, testing_data = split_data_train_test(all_data_paths, test_data_percentage, ensemble, split_by_patient)

    training_dataset = BrainCancerDataset(training_data)
    testing_dataset = BrainCancerDataset(testing_data)

    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
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

        plt.show()

        if random.uniform(0, 1) > 0.5:
            augmented_scan, augmented_mask = augment_images(scan, mask)
            augmented_training_data.append((augmented_scan, augmented_mask, "augmented_" + name))    

    training_data.extend(augmented_training_data)

    
def ensemble_data(training_data):
    '''
    Makes a dataset for current model from global dataset via sampling with replacement
    Both datasets have same size but new dataset can have recurring members
    Input:
        - training data (list): list of patients data used for training
    Returns:
        - sampled_training_data (list): new dataset sampled from training_data with replacement
    
    Example:
        - training_data = [0, 1, 2, 3, 4]
        - sampled_training_data = [3, 1, 3, 2, 0]
    '''
    random.seed(datetime.now())
    sampled_training_data = random.choices(training_data, k=len(training_data))

    return sampled_training_data


def make_dataloaders(datasets, batch_size):
    '''
    Wraps datasets into a torch.utils.data.DataLoader classes
    Input:
        -datasets (list): list of training and validation datasets: [(training_data1, test_data1), ... (training_data_n, test_data_n)]
        -batch_size (int): size of batches used for training and validation
    Returns:
        -dataloaders (list): same shape as input list but all dataset object are wrapped in torch.utils.data.DataLoader class
    '''
    dataloaders = []

    for dataset in datasets:
        training_data, testing_data = dataset
        training_dataset, testing_dataset = BrainCancerDataset(training_data), BrainCancerDataset(testing_data)
        training_dataloader, testing_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True), DataLoader(testing_data, batch_size=batch_size, shuffle=False)
        dataloaders.append((training_dataloader, testing_dataloader))
    
    return dataloaders


def get_all_data_paths(data_dir):
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
    shapes = defaultdict(lambda: 0)
    found_cancer = False
    total_cancer_patients = 0
    total_cancer_slices = 0

    for slice_name in sorted(os.listdir(data_dir)):
        volume_name = slice_name.split('_')[1]

        # We encountered a new patient 
        if (volume_name != last_volume_name) and (last_volume_name != '-1'):
            total_cancer_patients += found_cancer

            all_data_paths.append(all_paths_for_single_patient)  # Save data for old patient 
            all_paths_for_single_patient = []          # REset data for new patient
            found_cancer = False
        
        last_volume_name = volume_name
        slice_path = os.path.join(data_dir, slice_name)
        all_paths_for_single_patient.append(slice_path)

        f = h5py.File(slice_path, 'r') 
        image, mask = f['image'][()], f['mask'][()]
        if mask.sum():
            found_cancer = True
            total_cancer_slices += 1
        #shapes[image.shape] += 1
        #f.close()
        
    print(dict(shapes))
    print(f'total_cancer_patients = {total_cancer_patients}, total_cancer_slices = {total_cancer_slices}')
    exit(1)
    # Appending data for last patient
    all_data_paths.append(all_paths_for_single_patient)
    
    return all_data_paths


def read_tif_image(path):
    '''
    Opens .tif image from path
    Input:
        -path: path to image
    Returns:
        -image: np array representing image
    '''
    image = Image.open(path)
    # image = grayscale(image)
    image = np.array(image)
    if len(image.shape) > 2:
        image = np.transpose(image, [2,0,1])

    return image


def split_data_train_test(data, test_data_percentage, ensemble, split_by_patient):
    '''
    Splits data into training and test datasets
    Preprocesses data
    Inputs:
        -data: Iterable which will be split
        -test_data_percentage: Percentage of data which will go te testing dataset
    Returns:
        -train and test datasets (iterables) which have been preprocessed
    '''

    if split_by_patient:
        data_len = len(data)
        test_data_len = int(data_len / 100 * test_data_percentage)
        train_data_len = data_len - test_data_len

        train_data, test_data = random_split(data, (train_data_len, test_data_len), generator=Generator().manual_seed(1302))

        # Data is grouped by patient
        # Now that we split data into train and test datasets by patient
        # we can unify images of all patients in these sets together
        train_data = [volume for patient in train_data for volume in patient]
        test_data =  [volume for patient in test_data for volume in patient]
    
    else:

        # Data is grouped by patient
        # First we unify images of all patients in these sets together
        data = [volume for patient in data for volume in patient]

        # Calculate length of train and test data
        data_len = len(data)
        test_data_len = int(data_len / 100 * test_data_percentage)
        train_data_len = data_len - test_data_len

        # Split the data in train and test sets
        train_data, test_data = random_split(data, (train_data_len, test_data_len), generator=Generator().manual_seed(1302))

    # If ensembling is enabled then every dataset will be made from original dataset via sampling with replacement
    if ensemble:
        train_data = ensemble_data(train_data)

    return train_data, test_data


class BrainCancerDataset(Dataset):
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

        return image, mask
    

def preprocess_image_and_mask(image, mask):

    # cropping
    # resizing
    # normalizing
    # channel manipulation
    
    # Getting only FLAIR data
    image = image[..., 0]
    #image = image / 


def normalize_images(training_data, testing_data, normalization = 'Basic', standard_hist_path = ''):

    training_images, training_masks, training_names = [list(image_pair) for image_pair in zip(*training_data)]
    
    testing_images, testing_masks, testing_names = [list(image_pair) for image_pair in zip(*testing_data)]

    # Normalizing masks
    training_masks = np.array(training_masks)
    testing_masks = np.array(testing_masks)
    training_masks[training_masks > 0] = 1
    testing_masks[testing_masks > 0] = 1

    if normalization == 'Basic':
        training_images, testing_images = basic_normalization(training_images, testing_images)

    
    if normalization == 'Nyul':
        training_images, testing_images = nyul_normalization(training_images, testing_images, standard_hist_path)

    # Zipping data back to (scan, mask) pairs                                                                                                                                                                                                                                                                                                                                                                
    normalized_training_data = list(zip(training_images, training_masks, training_names))
    normalized_testing_data =  list(zip(testing_images, testing_masks, testing_names))

    return normalized_training_data, normalized_testing_data


def basic_normalization(training_images, testing_images):
    '''
    Returns normalized tensors
    Normalization is done based on parameters from training_images
    Values of tensor images are put between 0 and 1
    Input:
        -training_images: tensor which will be normalized based on its values
        -testing_images: tensor which will be normalized based on values from training_images
    Returns:
        -normalized_training_images: tensor with all values between 0 and 1
        -normalized_testing_images: tensor with all values between 0 and 1
    '''

    training_images = np.array(training_images)
    testing_images = np.array(testing_images)

    max_value = np.max(training_images)
    min_value = np.min(training_images)

    normalized_training_images = (training_images - min_value) / (max_value - min_value)
    normalized_testing_images = (testing_images - min_value) / (max_value - min_value)

    return normalized_training_images, normalized_testing_images


def nyul_normalization(training_images, testing_images, standard_hist_path):
    
    if not os.path.exists(standard_hist_path):
        nyul_train_standard_scale(training_images, standard_hist_path)
    
    data = np.load(standard_hist_path)
    standard_scale = data['standard_scale']
    percs = data['percs']

    training_landmarks = np.percentile(training_images, percs)
    testing_landmarks = np.percentile(testing_images, percs)

    training_f = interp1d(training_landmarks, standard_scale, kind='linear', fill_value='extrapolate')  # define interpolating function

    testing_f = interp1d(testing_landmarks, standard_scale, kind='linear', fill_value='extrapolate')  # define interpolating function            

    training_normalized_images = training_f(training_images)
    testing_normalized_images  = testing_f(testing_images)

    training_normalized_images = (training_normalized_images - np.min(training_normalized_images)) / np.ptp(training_normalized_images)
    testing_normalized_images = (testing_normalized_images - np.min(testing_normalized_images)) / np.ptp(testing_normalized_images)

    #if len(normalized_images) > 0:
    #    volumes[id_] = np.transpose(normalized_images, (2, 0, 1))
    #    masks[id_] = np.transpose(labels, (2, 0, 1))
    print(np.max(training_normalized_images))
    print(np.min(training_normalized_images))
    return training_normalized_images, testing_normalized_images


def nyul_train_standard_scale(images,
                              standard_hist_path,
                              i_min=1,
                              i_max=99,
                              i_s_min=1,
                              i_s_max=100,
                              l_percentile=10,
                              u_percentile=90,
                              step=10):

    percs = np.concatenate(([i_min], np.arange(l_percentile, u_percentile + 1, step), [i_max]))
    standard_scale = np.zeros(len(percs))

    for scan_image in images:
        landmarks_1 = np.percentile(scan_image, percs)

        min_p = np.percentile(scan_image, i_min)
        max_p = np.percentile(scan_image, i_max)
        f = interp1d([min_p, max_p], [i_s_min, i_s_max])  # create interpolating function

        landmarks_2 = np.array(f(landmarks_1))  # interpolate landmarks
        standard_scale += landmarks_2  # add landmark values of this volume to standard_scale

    standard_scale = standard_scale / len(images)  # get mean values
    np.savez(standard_hist_path, standard_scale=standard_scale, percs=percs)


def info_dump(total_loss, metrics, epoch_num, output_file, mode):

    metrics = [metric.item() for metric in metrics]
    info = f'epoch {epoch_num} | {mode}loss: {round(total_loss, 6)} IoU = {round(metrics[0], 3)} dice = {round(metrics[1], 3)} sens = {round(metrics[2], 3)} spec = {round(metrics[3], 3)} aoc = {round(metrics[4], 3)} hsdf = {round(metrics[5], 3)}'
    output_file.writelines(info + '\n')
    short_info = f'Epoch = {epoch_num} in {mode} loss = {total_loss}'
    print(short_info)


def prepare_output_files(params):

    output_path = params['output_path']
    directory_name = os.path.join(output_path, "output_folder_at_time=_" + str(datetime.now()))

    os.mkdir(directory_name)

    directory_name_train = os.path.join(directory_name, 'train')
    directory_name_valid = os.path.join(directory_name, 'valid')

    os.mkdir(directory_name_train)
    os.mkdir(directory_name_valid)

    train_text_file_name = os.path.join(directory_name_train, 'train_error.txt')
    valid_text_file_name = os.path.join(directory_name_valid, 'valid_error.txt')
    
    train_text_file = open(train_text_file_name, 'a+')
    valid_text_file = open(valid_text_file_name, 'a+')

    train_text_file.writelines(json.dumps(params) + '\n')
    valid_text_file.writelines(json.dumps(params) + '\n')
    
    return directory_name, train_text_file, valid_text_file


# Saves the picture of input MRI, model prediction and true label to location from path
def save_prediction_and_truth(loss_track_parameters, losses, y_pred, y_true):
 
    path, x, names, epoch, num_epochs, mode, image_num, treshold, bce_pos_weight = loss_track_parameters
    figure, axis = plt.subplots(2, 2)
    
    for i in range(y_true.shape[0]):
        loss = losses[i].item()
        name = names[i]

        local_x = x.detach().cpu().numpy()[i,0,:,:]
        local_y_pred = y_pred.detach().cpu().numpy()[i,0,:,:]
        local_y_pred_divided = local_y_pred >= treshold
        local_y_true = y_true.detach().cpu().numpy()[i,0,:,:]
        
        plt.title(name, loc='right')
        axis[0, 0].imshow(local_x)
        axis[0, 1].imshow(local_y_pred)
        axis[1, 0].imshow(local_y_pred_divided)
        axis[1, 1].imshow(local_y_true)
        num = image_num * y_true.shape[0] + i
        plt.savefig(path + f'/epoch_{epoch}__image_{num}__loss_{round(loss,4)}.jpg')
        plt.cla()
    plt.close()
    #plt.show()


def rename_files(data_dir):
    '''
    Renames files in subdirectories from data_dir
    This was necessary because when sorting file 'img_11' would come before 'img_2'
    To solve this we add '0' before every single digit so 'img_2' becomes 'img_02'
    '''
    for patient_dir in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, patient_dir)):
            continue
        image_names = sorted(os.listdir(os.path.join(data_dir, patient_dir)))
        for image_name in image_names:
            image_path = os.path.join(data_dir, patient_dir, image_name)
            words = image_path.split('_')
            if words[-1] == 'mask.tif':
                if(len(words[-2]) == 1):
                    words[-2] = '0' + words[-2]
                    new_name = '_'.join(words)
                    os.rename(image_path, new_name)
            else:
                if(len(words[-1]) == 5):
                    words[-1] = '0' + words[-1]
                    new_name = '_'.join(words)
                    os.rename(image_path, new_name)