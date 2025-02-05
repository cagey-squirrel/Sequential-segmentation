import random
from datetime import datetime
from torch import Generator
from torch.utils.data import random_split
import os
import json
from matplotlib import pyplot as plt
import torch


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


def split_data_train_test(data, test_data_percentage, ensemble, split_by_patient, split_seed):
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

        train_data, test_data = random_split(data, (train_data_len, test_data_len), generator=Generator().manual_seed(split_seed))
        
        # Data is grouped by patient
        # Now that we split data into train and test datasets by patient
        # we can unify images of all patients in these sets together
        
        # DEPRECATED: Moved to separate function called merge_patient_data
        #patients_len = []
        #for patient in train_data:
        #    patients_len.append(len(patient))
#
        #train_data = [volume for patient in train_data for volume in patient]
        #random.shuffle(train_data)
        #start_index = 0
        #new_train_data = []
        #for patient_len in patients_len:
        #    patient_data = train_data[start_index:start_index+patient_len]
        #    new_train_data.append(patient_data)
        #    start_index += patient_len
        #train_data = new_train_data
        #test_data =  [volume for patient in test_data for volume in patient]
    
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


def map_index_to_patient(data):

    index = 0
    mapping = {}

    for patient_index, patient in enumerate(data):
        for data_index, patient_data in enumerate(patient):
            mapping[index] = ((patient_index, data_index))
            
            index += 1

    return mapping


def merge_patient_data(data):
    '''
    Input data scans are gouped by patient
    This function merges all scans into a single list
    '''

    data = [volume for patient in data for volume in patient]

    return data


def group_patients_into_tensors(data):
    '''
    Input data scans are gouped by patient
    For each patient, list of scans is transformed into a tensor of scans
    This tensor of scans now serves as a batch dimension
    '''
    
    tensor_data = []
    for patient_data in data:
        patient_scans = []
        patient_masks = []
        patient_names = []
        for scan, mask, name in patient_data:
            patient_scans.append(scan)
            patient_masks.append(mask)
            patient_names.append(name)
    
        patient_tensor_data = torch.stack(patient_scans), torch.stack(patient_masks), patient_names
        tensor_data.append(patient_tensor_data)
    return tensor_data


def crop_square(image, square_size=224):
    '''
    Crops image to a square shape with width and height equal to square size
    If image dimension is smaller than square size then no cropping is done
    '''

    width, height, channels = image.shape

    if width > square_size:
        side_crop = (width - square_size) // 2
        image = image[side_crop : -side_crop, :, :]
    
    if height > square_size:
        top_crop = (height - square_size) // 2
        image = image[:, top_crop : -top_crop, :]


    return image


def info_dump(total_loss, metrics, epoch_num, output_file, mode):

    metrics = [metric.item() for metric in metrics]
    info = f'epoch {epoch_num} | {mode} loss: {round(total_loss, 6)} IoU = {round(metrics[0], 3)} dice = {round(metrics[1], 3)} prec = {round(metrics[2], 3)} rec = {round(metrics[3], 3)} auc = {round(metrics[4], 3)}'
    output_file.writelines(info + '\n')
    output_file.flush()
    short_info = f'Epoch = {epoch_num} {mode} loss = {total_loss}'
    print(short_info)


def prepare_output_files(params, folder_name):

    output_path = params['output_path']
    directory_name = os.path.join(output_path, folder_name + '_' + str(datetime.now())[-6:])

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
def save_prediction_and_truth(inputs, y_pred, y_true, path, names, epoch_num, batch_index, mode, dataset_type):

    if '3x' in dataset_type:  
        batches_x_seq, channels, width, height = inputs.shape
        inputs = inputs.view(batches_x_seq // 3, 3, channels, width, height)
        inputs = inputs[:, 1, ...]
    
    if mode == "validation":
        if not ((epoch_num % 100 == 0) or ((epoch_num % 20 == 0) and batch_index < 3)):
            return 
    if mode == 'training':
        if not ((epoch_num % 20 == 0) and batch_index < 3):
            return

    figure, axis = plt.subplots(1, 3)
    num_iter = min(y_true.shape[0], 8)
    #if mode == 'validation':
    #    num_iter = y_true.shape[0]
    
    for i in range(num_iter):

        name = names[i]

        local_inputs = inputs.detach().cpu().numpy()[i,0,:,:]
        local_y_pred = y_pred.detach().cpu().numpy()[i,0,:,:]
        local_y_true = y_true.detach().cpu().numpy()[i,0,:,:]
        
        plt.title(name, loc='right')
        axis[0].imshow(local_inputs)
        axis[1].imshow(local_y_pred)
        axis[2].imshow(local_y_true)
        num = batch_index * y_true.shape[0] + i
        plt.savefig(path + f'/image_{num}_loss.jpg')
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