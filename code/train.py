#Basics
import numpy as np
import sys
import yaml
import os

#Siamese
import pairwise_data_preparation

if __name__=='__main__':

    config = sys.argv[1] # Set the net configuration to be used

    try:
        with open('configs.yaml') as f:
            yaml_configs = yaml.safe_load(f)
        configs = yaml_configs[config]
    except KeyError:
        print(f"Configuration {config} not found")
        exit()
    
    data_prep = pairwise_data_preparation.DataPrep(configs['validation_size']) # Class that implements the paiwwise data preparation
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data_prep.load_data() # Load the mnist data

    pairs_train, labels_train = data_prep.create_pairwise_data(x_train, y_train) # make train pairs
    pairs_val, labels_val = data_prep.create_pairwise_data(x_val, y_val) # make validation pairs
    pairs_test, labels_test = data_prep.create_pairwise_data(x_test, y_test) # make test pairs

