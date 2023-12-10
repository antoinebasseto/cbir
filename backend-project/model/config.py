"""
Training and evaluation settings
"""
config = dict()

"""
Training or inference mode
"""
config['mode'] = 'train'  # 'train' or 'eval'

"""
Data related settings 
"""
config['dataset'] = 'HAM10000'

"""
Model related settings 
"""
config['model'] = {'skin': 'BetaVAEConv',
                   'mnist': '',}

config['model_path'] = {'skin': 'model/epoch=61-step=9734.ckpt',
                        'mnist': 'disentangling_vae/results/btcvae_mnist',}

config['image_path'] = {'skin': 'data/images',
                        'mnist': 'data/mnist/images',}

config['metadata_path'] = {'skin': 'data/HAM10000_latent_space_umap_processed.csv',
                           'mnist': 'data/mnist/mnist_latent_space_umap_processed.csv',}

config['cache_dir'] = 'data/cache'

config['projector_path'] = {'skin': 'data/umap.sav',
                            'mnist': 'data/mnist/umap_mnist.sav',}

config['num_dim'] = {'skin': 12,
                     'mnist': 10,}

"""
Logging and Analysis 
"""
config['results_dir'] = 'reports/logs'
