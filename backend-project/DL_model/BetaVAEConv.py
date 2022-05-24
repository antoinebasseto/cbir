import os

from src.models.losses import betatc_loss

"""
TO DO: Add additional MLP classifier with gradient ascent possibilities
"""

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.init import weights_init

class BetaVAEConv(pl.LightningModule):
    """beta-TCVAE model with encoder and decoder, inherit from Lightning module.
    Contains the following methods:
     - encoder
     - decoder
     - reparameterize
     - forward
     - init_weights
     - sample_latent
     - training_step
     - validation_step
     - configure_optimizers
     - get_progress_bar_dict
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        input_channels=1,
        input_dim=32,
        latent_dim=20,
        loss_type='H',
        trainset_size=5000,
        anneal_steps=200,
        is_mss=False,
        lr=0.001,
        batch_size=32,
        num_epochs=100,
        weight_decay=1e-4,
        log_dir="logs",
    ):
        """Called upon initialization. Constructs convolutional encoder and decoder architecture,
        initializes weights and loss.
        Parameters
        ----------
        alpha : float, optional
            Weight of the mutual information term, by default 1.0.
        anneal_steps : int, optional
            Number of annealing steps where gradually adding the regularisation, by default 200.
        beta : float, optional
            weight for beta VAE term, by default 1.0.
        gamma : float, optional
            Weight of clabing term, by default 1.0.
        input_channels : int, optional
            Input channels for colored or black and white image, by default 1.
        input_dim : int, optional
            Dimension of quadratic input image, by default 32.
        latent_dim : int, optional
            Latent dimension of encoder, by default 10.
        """
        super(BetaVAEConv, self).__init__()
        self.save_hyperparameters()
        self.num_iter = 0

        if input_dim == 32:
            hidden_dims = [32, 32, 64]
        elif input_dim == 64:
            hidden_dims = [32, 32, 64, 64]
        elif input_dim == 128:
            hidden_dims = [32, 32, 32, 64, 64]

        modules = []

        modules.append(
            nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    out_channels=hidden_dims[1],
                    kernel_size=4,
                    padding=1,
                    stride=2,
                ),
                nn.ReLU(),
            )
        )

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.ReLU(),
                )
            )

        self.enc = nn.Sequential(*modules)

        self.enc_fc = nn.Linear(in_features=1024, out_features=256)
        self.fc_mu = nn.Linear(in_features=256, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=256, out_features=latent_dim)

        # Decoder
        hidden_dims = list(reversed(hidden_dims))
        modules = []

        self.dec_fc_1 = nn.Linear(latent_dim, 256)
        self.dec_fc_2 = nn.Linear(256, 1024)

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.ReLU(),
                )
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    hidden_dims[-1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.ReLU(),
                nn.Conv2d(
                    hidden_dims[-1],
                    out_channels=input_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.Sigmoid(),
            )
        )

        self.dec = nn.Sequential(*modules)

    def encoder(self, x):
        """Takes image(s), returns mean and log variance vectors of latent space.
        Parameters
        ----------
        x : torch.Tensor
            Input image(s).
        Returns
        -------
        torch.Tensor, torch.Tensor
            Mean and log variance.
        """
        x = self.enc(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.enc_fc(x))
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

    def decoder(self, z):
        """Reconstructs image(s) from latent samples
        Parameters
        ----------
        z : torch.Tensor
            Latent samples.
        Returns
        -------
        torch.Tensor
            Reconstructed image(s).
        """
        x = F.relu(self.dec_fc_1(z))
        x = F.relu(self.dec_fc_2(x))
        x = x.view(-1, 64, 4, 4)
        x = self.dec(x)
        return x

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.
        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim).
        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim).
        Returns
        -------
        torch.Tensor
            Returns sampled value or maximum a posteriori (MAP) estimator.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x):
        mu, logvar = self.encoder(x)
        latent_sample = self.reparameterize(mu, logvar)
        reconstruct = self.decoder(latent_sample)
        return reconstruct, mu, logvar, latent_sample

    def init_weights(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """Returns a sample from the latent distribution.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        Returns
        -------
        torch.Tensor
            Samples from latent distribution.
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample

    def forward_pass(self,x):
        mu, log_var = self.encoder(x)
        return self.decoder(mu)

    def configure_optimizers(self):
        """Optimizer and learning rate scheduler configuration.
        Returns
        -------
        torch.optim.Adam, torch.optim.lr_scheduler.CosineAnnealingLR
            Returns optimizer and learning rate scheduler.
        """
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = {
            "scheduler": CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, verbose=True
            ),
            "interval": "epoch",
        }
        return [optimizer], [scheduler]

def build_betavaeconv(args, log_dir):
    return BetaVAEConv(beta=args['beta'], gamma=args['gamma'], input_channels=args['input_channels'], input_dim=args['input_dim'], latent_dim=args['latent_dim'],
                       loss_type=args['loss_type'], trainset_size=args['trainset_size'],
                       log_dir=log_dir)