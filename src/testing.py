import os 
import numpy as np
import segmentation_models_pytorch as smp


model = smp.Unet('resnet34', encoder_weights='imagenet')