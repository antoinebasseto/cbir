"""
Here we can store the hyperparameter configurations for each model
"""

from config import config
from .DETRtime.losses import LossLabels, LossCardinality, LossBoxes

params = {

    'DETRtime': {
        'lr': 1e-6,
        'batch_size': 32,
        'epochs': 200,
        #backbone
        'backbone': 'inception_time',
        'timestamps': 512,
        'in_channels': 3,
        'out_channels': 3, #backbone output channel
        'kernel_size' : 32,
        'nb_filters' : 64,
        'use_residual': False,
        'backbone_depth' : 4,
        #transformer
        'hidden_dim' : 64,
        'dropout' : False,
        'nheads': 8,
        'dim_feedforward' : 45,
        'enc_layers' : 5,
        'dec_layers' : 5,
        'pre_norm': False,
        #model size
        'num_classes': 5,
        'device': None,
        'position_embedding': 'sine',
        'num_queries': 12,
        'maxpools': [6, 4, 4, 2],

        #loss factors
        'bbox_loss_coef':2,
        'giou_loss_coef':2,
        'coord_loss_coef': 2,
        'losses': [LossBoxes(num_classes=5), LossCardinality(num_classes=5), LossLabels(num_classes=5)]

    },

}
