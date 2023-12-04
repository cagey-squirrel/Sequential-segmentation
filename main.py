from __future__ import absolute_import
import json
from src.training.train import train
from src.training.ensemble_testing import test_ensemble


if __name__ == '__main__':
    
    params_file = open('src/parameters/brain_cancer_segmentation_params.json', "r")
    params = json.load(params_file)
    
    train(params)
    #test_ensemble(params)
    