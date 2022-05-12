"""
Here we can store the hyperparameter configurations for each model
"""
from config import config

params = {

    'BetaVAEConv': {
        'lr': 1e-6,
        'batch_size': 32,
        'epochs': 200,
        #backbone
        'beta': 1.0,
        'gamma': 1.0,
        'input_channels': 3,
        'input_dim': 128,
        'latent_dim': 12,
        'loss_type': 'H',
        'trainset_size':5000
    },

}
