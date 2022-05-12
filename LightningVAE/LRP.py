import copy
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
#from misc_functions import apply_heatmap


class LRP():
    """
        Layer-wise relevance propagation with gamma+epsilon rule
        This code is largely based on the code shared in: https://git.tu-berlin.de/gmontavon/lrp-tutorial
        Some stuff is removed, some stuff is cleaned, and some stuff is re-organized compared to that repository.
    """
    def __init__(self, model):
        self.model = model

    def LRP_forward(self, layer, input_tensor, gamma=None, epsilon=None):
        # This implementation uses both gamma and epsilon rule for all layers
        # The original paper argues that it might be beneficial to sometimes use
        # or not use gamma/epsilon rule depending on the layer location
        # Have a look a the paper and adjust the code according to your needs

        # LRP-Gamma rule
        if gamma is None:
            gamma = lambda value: value + 0.05 * copy.deepcopy(value.data.detach()).clamp(min=0)
        # LRP-Epsilon rule
        if epsilon is None:
            eps = 1e-9
            epsilon = lambda value: value + eps

        # Copy the layer to prevent breaking the graph
        layer = copy.deepcopy(layer)

        # Modify weight and bias with the gamma rule
        try:
            layer.weight = nn.Parameter(gamma(layer.weight))
        except AttributeError:
            pass
            # print('This layer has no weight')
        try:
            layer.bias = nn.Parameter(gamma(layer.bias))
        except AttributeError:
            pass
            # print('This layer has no bias')
        # Forward with gamma + epsilon rule
        return epsilon(layer(input_tensor))

    def LRP_step(self, forward_output, layer, LRP_next_layer):
        # Enable the gradient flow
        forward_output.retain_grad()
        # Get LRP forward out based on the LRP rules
        print(forward_output)
        lrp_rule_forward_out = self.LRP_forward(layer, forward_output, None, None)
        # Perform element-wise division
        ele_div = (LRP_next_layer / lrp_rule_forward_out).data
        # Propagate
        (lrp_rule_forward_out * ele_div).sum().backward(retain_graph=True)
        # Get the visualization
        LRP_this_layer = (forward_output * forward_output.grad).data

        return LRP_this_layer

    def generate(self, input_space, target_image):
        #layers_in_model = self.model._modules['features']) + list(self.model._modules['classifier'])
        layers_in_model = [self.model.dec_fc_1, self.model.dec_fc_2] + [self.model.dec._modules[t] for t in  self.model.dec._modules]
        number_of_layers = len(layers_in_model)
        # Needed to know where flattening happens
        features_to_rehape_loc = 2

        # Forward outputs start with the input image
        forward_output = [input_space]
        # Then we do forward pass with each layer

        #decoder
        forward_output.append(self.model.dec_fc_1.forward(forward_output[-1].detach()))
        forward_output.append(self.model.dec_fc_2.forward(forward_output[-1].detach()))
        # for conv_layer in list(self.model._modules['features']):
        #     forward_output.append(conv_layer.forward(forward_output[-1].detach()))

        forward_output[-1] = forward_output[-1].view(-1, 64, 4, 4)
        # To know the change in the dimensions between features and classifier
        print(forward_output[-1].size())
        # for index, classifier_layer in enumerate(list(self.model.dec._modules)):
        #     forward_output.append(classifier_layer.forward(forward_output[-1].detach()))
        #for i, layer in enumerate(self.model.dec._modules):
        for i, idx in enumerate(self.model.dec._modules):
            layer =  self.model.dec._modules[idx]
            print(layer)
            forward_output.append(layer.forward(forward_output[-1].detach()))
        # Target for backprop
        #target_class_one_hot = torch.FloatTensor(1, 1000).zero_()
        #target_class_one_hot[0][target_class] = 1
        print(forward_output[-1].size())
        print(target_image.size())
        target_image
        # This is where we accumulate the LRP results
        LRP_per_layer = [None] * number_of_layers + [(forward_output[-1] * target_image).data]
        print(f'LRP_layer size ={len(LRP_per_layer)}')
        for layer_index in range(0, number_of_layers)[::-1]:
            # This is where features to classifier change happens
            # Have to flatten the lrp of the next layer to match the dimensions
            if layer_index == features_to_rehape_loc-1:
                LRP_per_layer[layer_index+1] = (LRP_per_layer[layer_index+1]).flatten(1)

            if isinstance(layers_in_model[layer_index], (torch.nn.Linear, torch.nn.Conv2d, torch.nn.MaxPool2d)):
                # In the paper implementation, they replace maxpool with avgpool because of certain properties
                # I didn't want to modify the model like the original implementation but
                # feel free to modify this part according to your need(s)
                lrp_this_layer = self.LRP_step(forward_output[layer_index], layers_in_model[layer_index], LRP_per_layer[layer_index+1])
                LRP_per_layer[layer_index] = lrp_this_layer
            else:
                lrp_this_layer = self.LRP_step(forward_output[layer_index], layers_in_model[layer_index],
                                               LRP_per_layer[layer_index + 1])
                LRP_per_layer[layer_index] = lrp_this_layer
                #LRP_per_layer[layer_index] = LRP_per_layer[layer_index+1]
        self.model.zero_grad()
        return LRP_per_layer

def apply_heatmap(R, sx, sy):
    """
        Heatmap code stolen from https://git.tu-berlin.de/gmontavon/lrp-tutorial
        This is (so far) only used for LRP
    """
    b = 10*((np.abs(R)**3.0).mean()**(1.0/3))
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis('off')
    heatmap = plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
    return heatmap

model_path = '/home/jimmy/Medical1-xai-iml22/LightningVAE/reports/logs/20220511-040101_BetaVAEConv/epoch=69-step=10990.ckpt'
image_path = '/home/jimmy/Medical1-xai-iml22/LightningVAE/reports/logs/20220511-040101_BetaVAEConv/pictures/62_7_0val_real.png'
target_path = '/home/jimmy/Medical1-xai-iml22/LightningVAE/reports/logs/20220511-040101_BetaVAEConv/pictures/62_7_0val_recon.png'
input_dim = 128
if __name__ == '__main__':
    # Get params
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"reports/logs/LRP_{run_id}_{config['model']}"
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

    pretrained_model.load_from_checkpoint(model_path)

    mu, logvar = pretrained_model.encoder(image)
    target_image = Image.open(target_path)
    target_image = transform(target_image)
    #cut out middle of the image
    target_image_subsection = torch.zeros_like(target_image)
    target_image_subsection[ :, int(input_dim/4):int(input_dim*3/4), int(input_dim/4):int(input_dim*3/4)] = \
        target_image[ :, int(input_dim/4):int(input_dim*3/4), int(input_dim/4):int(input_dim*3/4)]
    target_image_subsection = target_image_subsection.unsqueeze(0)
    target_image = target_image.unsqueeze(0)
    # LRP
    layerwise_relevance = LRP(pretrained_model)

    # Generate visualization(s)
    LRP_per_layer = layerwise_relevance.generate(mu, target_image)

    # Convert the output nicely, selecting the first layer
    lrp_to_vis = np.array(LRP_per_layer[0])
    #lrp_to_vis = np.array(Image.fromarray(lrp_to_vis).resize((prep_img.shape[2],
    #                      prep_img.shape[3]), Image.ANTIALIAS))

    lrp_subsection = layerwise_relevance.generate(mu, target_image_subsection)
    lrp_subsection = np.array(lrp_subsection[0])
    # Apply heatmap and save
    heatmap = apply_heatmap(lrp_to_vis, 4, 4)
    heatmap.figure.savefig(f'{log_dir}/LRP_out.png')

    heatmap = apply_heatmap(lrp_subsection, 4, 4)
    heatmap.figure.savefig(f'{log_dir}/LRP_subsection.png')
    heatmap = apply_heatmap(np.abs(lrp_to_vis - lrp_subsection), 4, 4)
    print(np.abs(lrp_to_vis - lrp_subsection).max())
    heatmap.figure.savefig(f'{log_dir}/LRP_diff.png')
    to_img = T.ToPILImage()
    target_image = to_img(target_image[0])
    target_image.save(f'{log_dir}/target_image.png')
    target_image_subsection = to_img(target_image_subsection[0])
    target_image_subsection.save(f'{log_dir}/target_image_subsection.png')
