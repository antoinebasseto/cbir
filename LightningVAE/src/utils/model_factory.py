
from src.models.BetaVAEConv import build_betavaeconv

def get_model(args, model_name, log_dir):
    if model_name == 'BetaVAEConv':
        return build_betavaeconv(args, log_dir)
    else:
        raise ValueError('Model {} not found'.format(model_name))

