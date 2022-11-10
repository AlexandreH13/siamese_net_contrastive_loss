#Basics
import numpy as np
import sys
import yaml
import os

#Siamese
import pairwise_data_preparation
from build_siamese_net import build_siamese
from build_siamese_net import contrastive_loss
import utils

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

    """
    Split training pairs
    """
    x_train_1 = pairs_train[:, 0]  # x_train_1.shape is (20000, 28, 28)
    x_train_2 = pairs_train[:, 1]

    """
    Split validation pairs
    """
    x_val_1 = pairs_val[:, 0]  # x_val_1.shape = (20000, 28, 28)
    x_val_2 = pairs_val[:, 1]

    """
    Split test pairs
    """
    x_test_1 = pairs_test[:, 0]
    x_test_2 = pairs_test[:, 1]

    siamese_net = build_siamese()
    siamese_net.compile(loss=contrastive_loss, optimizer="RMSprop")
    print(siamese_net.summary())

    history = siamese_net.fit(
        [x_train_1, x_train_2],
        labels_train,
        validation_data=([x_val_1, x_val_2], labels_val),
        batch_size=configs['batch_size'],
        epochs=configs['epochs'],
    )

    utils.plt_metrics(history=history.history, metric="loss", title="Contrastive Loss")

    '''
    Saving the weights
    '''
    print(f'Saving the weights for {config}')
    is_exists = os.path.exists(f'{config}/weights')
    if not is_exists:
        os.makedirs(f'data/{config}/weights', exist_ok=True)
    siamese_net.save_weights(f'data/{config}/weights/{config}_siamese_weights.h5')