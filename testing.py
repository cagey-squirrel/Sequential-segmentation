import os 
import numpy as np
import my_segmentation_models_pytorch as smp
from PIL import Image
from collections import defaultdict
from torchview import draw_graph
import h5py
from matplotlib import pyplot as plt


def rename(data_dir):

    for slice_name in sorted(os.listdir(data_dir)):
        volume_name = slice_name.split('_')[1]
        slice_num = slice_name.split('_')[3][:-3]

        volume_name = '0' * (3 - len(volume_name)) + volume_name
        slice_num = '0' * (3 - len(slice_num)) + slice_num
        original_slice_path = os.path.join(data_dir, slice_name)
        new_slice_name = 'volume_' + volume_name + '_slice_' + slice_num + '.h5'
        new_slice_path = os.path.join(data_dir, new_slice_name)
        os.rename(original_slice_path, new_slice_path)
        print(new_slice_path)


def get_data_from_dir(data_dir):
    '''
    Gets images from data_dir directory
    data_dir contains directories for each patient separatelly
    Inputs:
        data_dir: path to dir containing dirs of images for each patient
    Returns:
        List: [ [(p1_img1, p1_msk1), (p1_img2, p1_msk2)], [(p2_img1, p2_msk1), (p2_img2, p2_msk2)]...]
    '''
    total_patients = 0
    total_patients_with_cancer = 0
    has_cancer = False
    total_slices = 0
    total_slices_with_cancer = 0 
    shapes = defaultdict(lambda: 0)

    min_val_img = float('inf')
    min_val_mask = float('inf')
    max_val_img = float('-inf')
    max_val_mask = float('-inf')


    data = []
    for patient_dir in os.listdir(data_dir):
        if not os.path.isdir(os.path.join(data_dir, patient_dir)):
            continue
        image_names = sorted(os.listdir(os.path.join(data_dir, patient_dir)))

        has_cancer = False
        total_patients += 1

        current_patient_data = []

        image_is_next = True
        current_img_pair = []
        for image_name in image_names:
            image_path = os.path.join(data_dir, patient_dir, image_name)
            image = read_tif_image(image_path)

            if image_is_next:
                min_val_img = min(min_val_img, image.min())
                max_val_img = max(max_val_img, image.max())
                
                current_img_pair.append(image)
                image_is_next = False
            
            else:
                min_val_mask = min(min_val_mask, image.min())
                max_val_mask = max(max_val_mask, image.max())

                current_img_pair.append(image)
                current_img_pair.append(image_name)
                image_is_next = True
                current_patient_data.append(current_img_pair)
                current_img_pair = []

                shapes[image.shape] += 1
                total_slices += 1
                if image.sum() > 0:
                    has_cancer = True
                    total_slices_with_cancer += 1
        
        total_patients_with_cancer +=  has_cancer

        data.append(current_patient_data)
    
    print(f'paients with cancer = {total_patients_with_cancer}/{total_patients}')
    print(f'slices with cancer = {total_slices_with_cancer}/{total_slices}')
    print(f'image shapes = {shapes}')
    print(f'min_val_img = {min_val_img} max_val_img = {max_val_img} min_val_mask = {min_val_mask} max_val_mask = {max_val_mask}\n')
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
    # image = grayscale(image)
    image = np.array(image)
    if len(image.shape) > 2:
        image = np.transpose(image, [2,0,1])

    return image

def get_brats_data(path):
    min_val_img = float('inf')
    min_val_mask = float('inf')
    max_val_img = float('-inf')
    max_val_mask = float('-inf')

    for image_name in sorted(os.listdir(path)):
        image_path = os.path.join(path, image_name)
        #print(f'path = {path} image_name = {image_name} image_path = {image_path}\n')
        f = h5py.File(image_path, 'r') 
        image, mask = f['image'][()], f['mask'][()]
        print(image.shape)
        image = image[..., :-1]
        if image.min() == image.max():
            image *= 0
        else:
            image = (image - image.min()) / (image.max() - image.min())
        f.close()

        total_mask = (mask != 0).sum()
        mask *= 255
        #image[image < 1e-3] = 0
        print(image_name)
        print(mask.shape)
        total_non_zero = (mask != 0).sum()
        total_mask = mask.shape[0] * mask.shape[1] * mask.shape[2]
        print(f'filled {100*total_non_zero/total_mask}% where tnz = {total_non_zero} and total = {total_mask}')
        fig, axis = plt.subplots(1, 2)
        axis[0].imshow(image)
        axis[1].imshow(mask)
        plt.show()

        min_val_img = min(min_val_img, image.min())
        max_val_img = max(max_val_img, image.max())
        min_val_mask = min(min_val_mask, mask.min())
        max_val_mask = max(max_val_mask, mask.max())
    
    print(f'min_val_img = {min_val_img} max_val_img = {max_val_img} min_val_mask = {min_val_mask} max_val_mask = {max_val_mask}\n')


def rename_files(data_dir):
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

test_data = True
if test_data:
    data_dir = '/home/dzi/Documents/repos/brain_seg_data/kaggle_3m/'
    data_dir_cist = '/home/dzi/Documents/repos/kaggle_3m-cist/'
    data_dir_brats = '/home/dzi/Documents/repos/archive/BraTS2020_training_data/content/data_small'
    #get_data_from_dir(data_dir)
    #get_data_from_dir(data_dir_cist)
    get_brats_data(data_dir_brats)
    #rename_files(data_dir_cist)
    # rename('/home/dzi/Documents/repos/archive/BraTS2020_training_data/content/data')
    #dirs = set(list(os.listdir(data_dir)))
    #dirs_cist = set(list(os.listdir(data_dir_cist)))

    #print(dirs.difference(dirs_cist))
    #print(dirs_cist.difference(dirs))


#model = smp.Unet('resnet34', encoder_weights='imagenet')
#model_graph = draw_graph(model, input_size=(8, 3, 256, 256))
#model_graph.visual_graph.view()
#print(type(model_graph.visual_graph))