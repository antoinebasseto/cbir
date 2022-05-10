import os, os.path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import matplotlib.patches as patches
import glob
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
from torch import nn, Tensor
import torch.optim as optim
from torchvision.transforms import ToPILImage, Resize, Normalize
import torchvision.transforms.functional as TF
from typing import List
from random import randint


def read_image(path):
    '''
    Read an image in memory

    Parameters
    ----------
    path : str
        The path to the image (folder + filename)

    Return
    ----------
    np_img: np.array(X, Y):
        The image (not scaled) stored in this path
    '''
    img = Image.open(path)
    # np_img = np.array(img)

    return img


def scale_image(np_img):
    '''
    Scale an image from range [0, 255] to range [0, 1]

    Parameters
    ----------
    np_img : np.array(X, Y)
        The unscaled ([0, 255]) image

    Return
    ----------
    np.array(X, Y):
        The scaled ([0, 1]) image
    '''
    return np_img / 255


# %%
class XRayDataset(Dataset):
    '''
    Class to represent our dataset of XRay images. Subclass of torch.utils.data.Dataset. Store the image directory,
    images name, bounding boxes and data of the images we will use either for training or testing.
    '''

    def __init__(self, img_dir, transform=None, train=True):
        '''
        Constructor

        Parameters
        ----------
        img_dir : str
            Directory where the images are stored. Will take all the png files present in this directory
        transform: optional, function
            Transformation used on every image before returning it such as scaling.
        train: optional, Boolean
            If set to true, we keep only the training data in the corresponding folder and otherwise the testing one
        '''
        self.img_dir = img_dir
        # Read all the names of the PNG files in the directory
        images_name = np.array([name for name in os.listdir(img_dir) if
                                os.path.isfile(os.path.join(img_dir, name)) and re.search('jpg$', name) is not None])
        print("Disables loading data")
        # data, train_names, test_names, bounding_boxes = load_data(filenames_to_keep = images_name)
        #
        # # Keep only the training or test ones
        # self.data = data[data["Filename"].isin((train_names if train else test_names))]
        # self.bounding_boxes = bounding_boxes[bounding_boxes["Filename"].isin((train_names if train else test_names))]
        # self.images_name = images_name[np.isin(images_name, (train_names if train else test_names))]
        self.images_name = images_name
        self.transform = transform

        self.to_tensor = ToTensor()
        # self.normalize = Normalize()

    def __len__(self):
        '''
        Overload: return the length of the dataset

        Return
        ----------
        length: int:
            The length of the dataset
        '''
        return self.images_name.shape[0]-1

    def __getitem__(self, idx):
        '''
        Overload; get the item (only image for the moment) corresponding to the given idx.

        Parameters
        ----------
        idx : int
            Index of the item we want to get.

        Return
        ----------
        image: np.array(X, Y):
            The corresponding image transformed with the transform method
        '''
        img_path = os.path.join(self.img_dir, self.images_name[idx])
        image = read_image(img_path)

        if TF.get_image_num_channels(image) > 1:
            image = TF.to_grayscale(image)
            # image = image.unsqueeze(0)
        # uncomment later
        # data = self.data[self.data["Filename"] == self.images_name[idx]]
        #
        # bbox = None
        # if not self.bounding_boxes[self.bounding_boxes["Filename"] == self.images_name[idx]].empty:
        #     bbox = self.bounding_boxes[self.bounding_boxes["Filename"] == self.images_name[idx]]
        #

        if self.transform:
            image = self.transform(image)
            #             if image.shape[0] == 1:
            #                 image = image.squeeze()
            image = self.to_tensor(image)
            # image = self.normalize(image)

        return image  # , data, bbox # Cannot pass data and bbox for the moment because they contain entries of type "object" which is a problem for dataloader. Should transform the few entries having object type


# %% md

class ConvUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, scale_factor=2):
        super(ConvUpsampling, self).__init__()

        self.scale_factor = scale_factor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        return self.conv(x)


# %% md
# Model
# %% md
## Variational Encoder
# %%
class VariationalEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super().__init__()

        modules = []
        # if hidden_dims is None:
        #     hidden_dims = [32, 64, 128]
        #
        # # # Build Encoder
        # for h_dim in hidden_dims:
        #     modules.append(
        #         nn.Sequential(
        #             nn.Conv2d(in_channels, out_channels=h_dim,
        #                       kernel_size=3),
        #             # nn.Conv2d(h_dim, out_channels=h_dim,
        #             #           kernel_size=2, stride=2),
        #             nn.Conv2d(h_dim, out_channels=h_dim,
        #                       kernel_size=3),
        #
        #             nn.BatchNorm2d(h_dim),
        #             nn.LeakyReLU())
        #     )
        #     in_channels = h_dim
        # #
        # self.encoder = nn.Sequential(*modules)
        self.linear = nn.Sequential(
            nn.Linear(28*28*in_channels, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            # nn.Linear(hidden_dims[-1] * 16 * 16, 1024),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_var = nn.Linear(1024, latent_dim)
        # self.fc_mu = nn.Linear(224*224*in_channels, latent_dim)
        # self.fc_var = nn.Linear(224*224*in_channels, latent_dim)

        print("Setting log_variance init to 0")
        torch.nn.init.zeros_(self.fc_var.weight)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = input
        #result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        result = self.linear(result)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]


# %% md
## Decoder
# %%
class Decoder(nn.Module):
    def __init__(self,
                 encoder_in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super().__init__()

        self.encoder_in_channels = encoder_in_channels

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Build Decoder
        #self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 32 * 32)
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LeakyReLU(),
            # nn.Linear(latent_dim, 1024),
            # nn.LeakyReLU(),
            #nn.Linear(1024, hidden_dims[0] * 16 * 16),
            nn.Linear(1024, 28*28*encoder_in_channels),
            nn.Sigmoid()
            #nn.LeakyReLU()
        )

        # self.hidden = hidden_dims[0]
        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             # ConvUpsampling(hidden_dims[i],
        #             #                hidden_dims[i+1],
        #             #                kernel_size=3,
        #             #                scale_factor=2,
        #             #                padding=1),
        #             # nn.Conv2d(hidden_dims[i + 1], hidden_dims[i + 1],
        #             #           kernel_size=3, stride=1, padding=1),
        #             nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
        #                                kernel_size=3, stride=1),
        #             nn.ConvTranspose2d(hidden_dims[i+1], hidden_dims[i + 1],
        #                                kernel_size=3, stride=1),
        #             nn.BatchNorm2d(hidden_dims[i + 1]),
        #             nn.LeakyReLU())
        #         # nn.ReLU())
        #     )
        #
        # self.decoder = nn.Sequential(*modules)
        #
        # self.final_layer = nn.Sequential(
        #     # ConvUpsampling(hidden_dims[-1],
        #     #                hidden_dims[-1],
        #     #                kernel_size=5,
        #     #                scale_factor=2,
        #     #                padding=2),
        #     # nn.BatchNorm2d(hidden_dims[-1]),
        #     # nn.LeakyReLU(),
        #     nn.ConvTranspose2d(hidden_dims[-1], out_channels=hidden_dims[-1],
        #               kernel_size=3),
        #     nn.ConvTranspose2d(hidden_dims[-1], out_channels=encoder_in_channels,
        #                        kernel_size=3),
        #     nn.Sigmoid())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.encoder_in_channels, 28, 28)
        #result = result.view(-1, self.hidden, 16, 16)
        #result = self.decoder(result)
        #result = self.final_layer(result)
        return result


# %% md
## Beta-VAE
# %%
class BetaVAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims_encoder: List = None,
                 hidden_dims_decoder: List = None,
                 beta: int = 4,
                 gamma: float = 2.0,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type: str = 'B',
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        print("Using Loss type: {}".format(self.loss_type))

        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.num_iter = 0

        self.variational_encoder = VariationalEncoder(in_channels, latent_dim, hidden_dims_encoder)
        self.decoder = Decoder(in_channels, latent_dim, hidden_dims_decoder)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        mu, log_var = self.variational_encoder(input)
        z = self.reparameterize(mu, log_var)
        return [self.decoder(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # print(f'mu: {mu}')
        # recons_loss = F.mse_loss(recons, input, reduction='mean')
        recons_loss = F.binary_cross_entropy(recons, input)
        # print(f'recons_loss: {recons_loss}')
        # why the sum factor?
        # kl_loss = torch.mean(0.5 * torch.sum(torch.exp(log_var) + mu ** 2 - 1. - log_var, 1))
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        # kld_loss = 0
        # print(f'kld_loss: {kl_loss}')
        if self.loss_type == 'H':  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_loss
            # print(f'loss: {loss}')
        elif self.loss_type == 'B':  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'reconstruction_Loss': recons_loss.detach(), 'KLD': kld_loss.detach()}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]


# %% md

# %% md
# Training
# %%
from tqdm import tqdm

log_dir = "/home/jimmy/Medical1-xai-iml22/VAE/log/skin_lesions_beta_1_linear/"


def train(model, train_loader, optimizer, test_dataset, num_epochs=5, verbose=1):
    '''
    Train the model.

    Parameters
    ----------
    model: nn.Module
        The model.
    train_loader: DataLoader
        A Pytorch Dataloader containing the training images.
    num_epochs: optional, int
        The number of epochs to train.
    verbose: optional, int
        Set the number of printing we want during training. For the moment, only 0 for no printing or another value
        to get some information about the update.

    Return
    ----------
    loss_history: list
        A list of training losses. One for each epoch of training.
    '''
    loss_history = []
    best_loss = 0
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    for e in range(num_epochs):
        if verbose:
            print(f"=========== EPOCH {e} ===========")
        total_loss = 0
        recons_loss = 0
        kld_loss = 0
        number_samples = 0
        model.num_iter = 0
        for batch in tqdm(train_loader):
            X = batch
            X = X.to(device)
            output = model(X)
            loss = loss_fn(*output)
            total_loss += loss["loss"].detach()
            recons_loss += loss["reconstruction_Loss"]
            kld_loss += loss["KLD"]
            number_samples += X.shape[0]
            optimizer.zero_grad()
            loss["loss"].backward()
            optimizer.step()

            # if verbose and i%50 == 0and not i==0:
            #     print(f"Done processing batch number {i}")
        print(f"Mean loss = {total_loss / number_samples}")
        print(f"Mean reconstruction loss = {recons_loss / number_samples}")
        print(f"Mean KLD loss = {kld_loss / number_samples}")

        if e == 0 or total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), log_dir + "best_model.pth")
            print(f'Saved the best model')
            print(f'New best loss {best_loss}')
        loss_history.append(total_loss.cpu() / number_samples)

        transform = ToPILImage()
        # %%
        for i in range(10):
            j = randint(0, len(test_dataset))
            image_generated = model.generate(test_data[j].unsqueeze(dim=0).to(device))
            transform(image_generated[0]).save(
                f'{log_dir}images/epoch_{e}_{j}_gen.png')
            transform(test_data[j]).save(f'{log_dir}images/epoch_{e}_{j}_real.png')
    return loss_history


# %%
def plot_loss(epoch_losses, title='Loss'):
    '''
    Simple plot for the loss at each epoch.

    Parameters
    ----------
    epoch_losses: list
        A list of training losses. One for each epoch of training.
    title: optional, str
        the title of the plot.
    '''
    plt.figure()
    plt.plot(epoch_losses)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('/home/jimmy/Medical1-xai-iml22/VAE/log/loss.png')


# %%
# Compute the dataloader for training and testing using our custom dataset
IMG_DIR = "../backend-project/data/skin/HAM10000_images_part_1/"
BATCH_SIZE = 2

training_data = XRayDataset(
    img_dir=IMG_DIR,
    train=True,
    transform=Resize((28, 28))  # This method directly scale the image in [0, 1] range
)

test_data = XRayDataset(
    img_dir=IMG_DIR,
    train=False,
    transform=Resize((28, 28))  # This method directly scale the image in [0, 1] range
)

train_loader = DataLoader(training_data,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = DataLoader(test_data,
                         batch_size=BATCH_SIZE,
                         shuffle=True)
# %%
# Create our autoencoder and train it
# Parameters
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4

LATENT_SPACE_DIM = 1000
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')
# Model
beta_vae = BetaVAE(in_channels=1, latent_dim=128, beta=1, loss_type='H')

beta_vae.to(device)
# Loss function & optimizer
loss_fn = beta_vae.loss_function
optimizer = optim.Adam(beta_vae.parameters(),
                       lr=LEARNING_RATE)

epoch_losses = train(beta_vae, train_loader, optimizer, test_data, num_epochs=NUM_EPOCHS)
plot_loss(epoch_losses, 'Training Loss')
