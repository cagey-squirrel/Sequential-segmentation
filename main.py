from __future__ import absolute_import
import json
from src.training.train import train
from src.training.ensemble_testing import test_ensemble


if __name__ == '__main__':
    
    params_file = open('src/parameters/brain_cancer_segmentation_params.json', "r")
    params = json.load(params_file)
    
    split_seed = 1302

    #train(params)
    for i in range(10):
        train(params, split_seed=split_seed)
        split_seed = (75*split_seed + 74) % (2**16 + 1)
    #test_ensemble(params)

    
    