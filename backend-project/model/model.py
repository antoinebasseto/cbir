import os

import torch
import torchvision.transforms as T

from .BetaVAEConv import BetaVAEConv, build_betavaeconv


def get_model(args, model_name, log_dir, model_path):
    if model_name == 'BetaVAEConv':
        model = build_betavaeconv(args, log_dir)
        chkt = torch.load(model_path, map_location=torch.device('cpu'))
        model = BetaVAEConv.load_from_checkpoint(checkpoint_path=model_path)
        model.load_state_dict(chkt['state_dict'])
        model.eval()
        return model
    else:
        raise ValueError('Model {} not found'.format(model_name))


def get_image_preprocessor(args, model_name):
    if model_name == 'BetaVAEConv':
        input_dim = args["input_dim"]
        return T.Compose([T.Resize((input_dim, input_dim)), T.ToTensor()])
    else:
        raise ValueError('Model {} not found'.format(model_name))


def rollout_i(mu, dimension, num_rollouts, upper_bound, lower_bound):
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
    return mu


def generate_images_for_rollout_i(model, latent_origin, latents_rollout, dimension, num_rollouts, upper_bound, lower_bound, cache_dir):
    images = model.decoder(latents_rollout)
    # img_width = images.shape[2]
    # img_height = images.shape[3]
    to_image = T.ToPILImage()
    for i in range(num_rollouts):
        # print(images[i])
        # to_image(images[i]).save(os.path.join(log_dir, '{}.png'.format(i)))
        img_name = f"{torch.max(latent_origin)}_{torch.min(latent_origin)}_{dimension}_{i}{upper_bound}{lower_bound}{num_rollouts}.png"
        img = to_image(images[i].squeeze())
        img.save(os.path.join(cache_dir, img_name))


def rollout(model, mu, cache_dir, lower_bound, upper_bound, num_rollouts):
    # image_path = os.path.join(img_dir, img_name)
    # img = Image.open(image_path)
    # mu, logvar = model.encoder(img)
    latent_dimension = mu.shape[1]
    ret = []
    for j in range(latent_dimension):
        ret.append([f'{torch.max(mu)}_{torch.min(mu)}_{j}_{i}{upper_bound}{lower_bound}{num_rollouts}.png' for i in range(num_rollouts)])
        if f"{torch.max(mu)}_{torch.min(mu)}_{j}_{0}{upper_bound}{lower_bound}{num_rollouts}.png" in os.listdir(cache_dir):
            continue
        latents_rollout = rollout_i(mu, j, num_rollouts, upper_bound, lower_bound)
        generate_images_for_rollout_i(model,  mu, latents_rollout, j, num_rollouts, upper_bound, lower_bound, cache_dir)

    return ret