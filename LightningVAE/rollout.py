import os
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from src.utils.model_factory import get_model
from src.models.hyperparameters import params
from config import config

def create_rollout(model, mu, dimension, num_rollouts, upper_bound, lower_bound, log_dir):
    """

    :param model:
    :param mu:
    :param dimension:
    :param num_rollouts:
    :param range:
    :return:
    """
    mu = mu.repeat(num_rollouts, 1)
    #print(mu.size())
    perturbation = torch.zeros_like(mu)
    step = (upper_bound - lower_bound) / num_rollouts
    perturbation[:, dimension] = torch.arange(lower_bound, upper_bound, step)
    mu = mu + perturbation

    images = model.decoder(mu)
    #print(images.size())
    img_width = images.shape[2]
    img_height = images.shape[3]
    dst = Image.new('RGB', (img_width * num_rollouts, img_height))
    to_image = T.ToPILImage()

    for i in range(num_rollouts):
        #print(images[i])
        #to_image(images[i]).save(os.path.join(log_dir, '{}.png'.format(i)))
        dst.paste(to_image(images[i].squeeze()), (i * img_width, 0))
    return dst

model_path = '/home/jimmy/Medical1-xai-iml22/LightningVAE/reports/logs/20220511-040101_BetaVAEConv/epoch=69-step=10990.ckpt'
image_path = '/home/jimmy/Medical1-xai-iml22/LightningVAE/reports/logs/20220512-182832_BetaVAEConv/pictures/18_4_1val_real.png'
input_dim = 128
lower_bound = -2
upper_bound = 2
num_rollouts = 10

if __name__ == '__main__':
    # Get params
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"reports/logs/rollouts_{run_id}_{config['model']}_{lower_bound}_{upper_bound}_{num_rollouts}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(f"{log_dir}/pictures"):
        os.makedirs(f"{log_dir}/pictures")

    # target_example = 2  # Spider
    # (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
    #     get_example_params(target_example)
    # #
    img = Image.open(image_path)
    transform = T.Compose([T.Resize((input_dim, input_dim)), T.ToTensor()])
    image = transform(img)
    image = image.unsqueeze(0)
    pretrained_model = get_model(params[config['model']], config['model'], f'{log_dir}/pictures')

    pretrained_model = pretrained_model.load_from_checkpoint(checkpoint_path=model_path)

    mu, logvar = pretrained_model.encoder(image)
    decoded_image = pretrained_model.forward_pass(image)

    to_image = T.ToPILImage()
    to_image(image[0]).save(f"{log_dir}/pictures/{config['model']}_original.png")
    to_image(decoded_image[0]).save(f"{log_dir}/pictures/forward_pass_original.png")
    # Create rollout
    print(mu.shape)
    latent_dimension = mu.shape[1]
    for i in range(latent_dimension):
        rollout = create_rollout(pretrained_model, mu, i, 10, -1, 1, log_dir)
        rollout.save(f"{log_dir}/pictures/{i}_rollout.png")