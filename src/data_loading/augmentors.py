'''
This script is used for augmenting images
Images consist of scan image and mask image which must be equally augmented
Images are in form of np arrays
'''

import numpy as np
from torchvision.transforms.functional import affine, adjust_contrast
import torch 
import random
from matplotlib import pyplot as plt


def augment_images(scan, mask):
    '''
    This function does random augmentation of image and mask or returns them unchanged
    Input:
        -scan (np array): single image of a MRI scan
        -scan (np array): single image of a mask from that scan
    Output:
        -augmented scan (np_array)
        -augmented mask (np_array) 
    '''
    chance = random.uniform(0, 1)

    # random augmentation
    if chance < 0.33:
        return rotate_images(scan, mask)
    if chance < 0.66:
        return scale_images(scan, mask)
    else:                                                                                                
    # if chance < 0.75:
        return shear_images(scan, mask)
    #else:
        #return change_image_contrast(scan, mask, 0.7)

def flip_images(scan, mask):
    '''
    Flips image on y axis like a mirror
    Inputs:
        - scan (np array): image of a scan 
        - mask (np array): image of a mask 
    Returns:
        - flipped_scan (np array): flipped image of a scan
        - flipped_mask (np array): flipped image of a mask
    '''
    flipped_scan = np.fliplr(scan)
    flipped_mask = np.fliplr(mask)
    
    return flipped_scan, flipped_mask

def rotate_images(scan, mask, angle=5):
    '''
    Rotates images for angle degrees in clock-wise direction
    Inputs:
        - scan (np array) (np array): image of a scan 
        - mask (np array)(np array): image of a mask
        - angle (int): number of degrees that images will be rotated
    Returns:
        - rotated_scan (np array): rotated image of a scan
        - rotated_mask (np array): rotated image of a mask
    '''

    # affine takes tensor as input so we need to convert images
    scan = torch.from_numpy(scan)
    mask = torch.from_numpy(mask)

    # Tensor takes the shape of (C, H, W) so we need to add a dummy channel dim
    scan = scan[None, :, :]
    mask = mask[None, :, :]
    
    rotated_scan = affine(scan, angle=angle, translate=[0, 0], scale=1, shear=0)
    rotated_mask = affine(mask, angle=angle, translate=[0, 0], scale=1, shear=0)
    
    # Now we need to change back the dimensions of tensor from (C, H, W) to (H, W, C)
    rotated_scan = torch.permute(rotated_scan, (1,2,0))
    rotated_mask = torch.permute(rotated_mask, (1,2,0))

    # Now we drop the dummy dimension C:
    rotated_scan = rotated_scan.squeeze()
    rotated_mask = rotated_mask.squeeze()
    
    # Now we convert rotated scan and image back to np arrays
    rotated_scan = rotated_scan.numpy()
    rotated_mask = rotated_mask.numpy()

    return rotated_scan, rotated_mask

def scale_images(scan, mask, scale=1.1):
    '''
    Zooms images scale times
    Inputs:
        - scan (np array): image of a scan 
        - mask (np array): image of a mask
        - scale (int): scale decalring how much to zoom in a pic
    Returns:
        - scaled_scan (np array): rotated image of a scan
        - scaled_mask (np array): rotated image of a mask
    '''

    # affine takes tensor as input so we need to convert images
    scan = torch.from_numpy(scan)
    mask = torch.from_numpy(mask)

    # Tensor takes the shape of (C, H, W) so we need to add a dummy channel dim
    scan = scan[None, :, :]
    mask = mask[None, :, :]
    
    scaled_scan = affine(scan, angle=0, translate=[0, 0], scale=scale, shear=0)
    scaled_mask = affine(mask, angle=0, translate=[0, 0], scale=scale, shear=0)
    
    # Now we need to change back the dimensions of tensor from (C, H, W) to (H, W, C)
    scaled_scan = torch.permute(scaled_scan, (1,2,0))
    scaled_mask = torch.permute(scaled_mask, (1,2,0))

    # Now we drop the dummy dimension C:
    scaled_scan = scaled_scan.squeeze()
    scaled_mask = scaled_mask.squeeze()
    
    # Now we convert rotated scan and image back to np arrays
    scaled_scan = scaled_scan.numpy()
    scaled_mask = scaled_mask.numpy()

    return scaled_scan, scaled_mask

def shear_images(scan, mask, shear=5):
    '''
    Zooms images scale times
    Inputs:
        - scan (np array) (np array): image of a scan 
        - mask (np array)(np array): image of a mask
        - shear (int): sheard decalring how much to shear a pic
    Returns:
        - sheard_scan (np array): sheard image of a scan
        - sheard_mask (np array): sheard image of a mask
    '''

    # affine takes tensor as input so we need to convert images
    scan = torch.from_numpy(scan)
    mask = torch.from_numpy(mask)

    # Tensor takes the shape of (C, H, W) so we need to add a dummy channel dim
    scan = scan[None, :, :]
    mask = mask[None, :, :]
    
    sheard_scan = affine(scan, angle=0, translate=[0, 0], scale=1, shear=shear)
    sheard_mask = affine(mask, angle=0, translate=[0, 0], scale=1, shear=shear)
    
    # Now we need to change back the dimensions of tensor from (C, H, W) to (H, W, C)
    sheard_scan = torch.permute(sheard_scan, (1,2,0))
    sheard_mask = torch.permute(sheard_mask, (1,2,0))

    # Now we drop the dummy dimension C:
    sheard_scan = sheard_scan.squeeze()
    sheard_mask = sheard_mask.squeeze()
    
    # Now we convert rotated scan and image back to np arrays
    sheard_scan = sheard_scan.numpy()
    sheard_mask = sheard_mask.numpy()

    return sheard_scan, sheard_mask

def change_image_contrast(scan, mask, contrast=0.9):
    '''
    Changing contrast of images
    Note: this transformation is applied to scan image only and not to mask
    Inputs:
        - scan (np array) (np array): image of a scan 
        - mask (np array)(np array): image of a mask
        - contrast (int): contrast which will be applied to picture
    Returns:
        - contrasted_scan (np array): image of a scan with applied contrast change
        - mask (np array): same as input mask
    '''

    # affine takes tensor as input so we need to convert images
    scan = torch.from_numpy(scan)

    # Tensor takes the shape of (C, H, W) so we need to add a dummy channel dim
    scan = scan[None, :, :]
    
    contrasted_scan = adjust_contrast(scan, contrast)
    
    # Now we need to change back the dimensions of tensor from (C, H, W) to (H, W, C)
    contrasted_scan = torch.permute(contrasted_scan, (1,2,0))

    # Now we drop the dummy dimension C:
    contrasted_scan = contrasted_scan.squeeze()
    
    # Now we convert rotated scan and image back to np arrays
    contrasted_scan = contrasted_scan.numpy()

    return contrasted_scan, mask