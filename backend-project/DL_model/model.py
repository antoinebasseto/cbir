import os

import torch
import torchvision.transforms as T
from PIL.Image import Image

from BetaVAEConv import build_betavaeconv

def get_model(args, model_name, log_dir, model_path):
    if model_name == 'BetaVAEConv':
        model = build_betavaeconv(args, log_dir)
        pretrained_model = model.load_from_checkpoint(checkpoint_path=model_path)
        return pretrained_model
    else:
        raise ValueError('Model {} not found'.format(model_name))

def get_image_preprocessor(args, model_name):
    if model_name == 'BetaVAEConv':
        input_dim = args["input_dim"]
        return T.Compose([T.Resize((input_dim, input_dim)), T.ToTensor()])
    else:
        raise ValueError('Model {} not found'.format(model_name))

def rollout_i(model, img_name, mu, dimension, num_rollouts, upper_bound, lower_bound, cache_dir):
    """
    batched rollout
    :param model:
    :param mu:
    :param dimension:
    :param num_rollouts:
    :param range:
    :return:
    """

    mu = mu.repeat(num_rollouts, 1)
    perturbation = torch.zeros_like(mu)
    step = (upper_bound - lower_bound) / num_rollouts
    perturbation[:, dimension] = torch.arange(lower_bound, upper_bound, step)
    mu = mu + perturbation

    images = model.decoder(mu)
    img_width = images.shape[2]
    img_height = images.shape[3]
    to_image = T.ToPILImage()
    for i in range(num_rollouts):
        #print(images[i])
        #to_image(images[i]).save(os.path.join(log_dir, '{}.png'.format(i)))
        img_name = hash(f"{img_name}{dimension}{i}{upper_bound}{lower_bound}")
        img = to_image(images[i].squeeze())
        img.save(os.path.join(cache_dir, f'{img_name}.png'))

def rollout(model, img_name, img_dir, cache_dir, lower_bound, upper_bound, num_rollouts):
    image_path = os.path.join(img_dir, img_name)
    img = Image.open(image_path)
    mu, logvar = model.encoder(img)
    latent_dimension = mu.shape[1]
    for j in range(latent_dimension):
        if hash(f"{img_name}{j}{0}{upper_bound}{lower_bound}.png") in os.listdir(cache_dir):
            continue
        rollout_i(model, img_name, mu, j, num_rollouts, lower_bound, upper_bound, cache_dir)