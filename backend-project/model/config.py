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
config['dataset'] = 'HAM10000'  # options: sleep-edf-153, ...
# load input size from json file of the dataset 

# with open(f"data/{config['dataset']}/info.json") as f:
#     data = json.load(f)
#     config['input_width']  = data['input_width']
#     config['input_height'] = data['input_height']

"""
Model related settings 
"""
config['model'] = 'BetaVAEConv'

config['model_path'] = 'model/epoch=61-step=9734.ckpt'
config['image_path'] = 'data/images'
config['metadata_path'] = 'data/HAM10000_latent_space_umap_processed.csv'
config['cache_dir'] = 'data/cache'
"""
Training related settings
"""
# Most of them are moved to hyperparameters.py for model specific settings

"""
Logging and Analysis 
"""
config['results_dir'] = 'reports/logs'

"""
Logging and Analysis 
"""
#config['model_path'] = f"model/{config['model']}"
