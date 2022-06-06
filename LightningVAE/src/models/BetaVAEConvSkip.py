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

        # Loss
        self.loss = betatc_loss(
            is_mss=is_mss,
            steps_anneal=anneal_steps,
            n_data=trainset_size,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        # Weight init
        self.init_weights()
        self.warm_up = 0

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

    # def loss(self, *args):
    #     self.num_iter += 1
    #     recons = args[0]
    #     input = args[1]
    #     mu = args[2]
    #     log_var = args[3]
    #
    #     recons_loss =F.mse_loss(recons, input)
    #
    #     kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(),
    #                                            dim=1), dim=0)
    #
    #     if self.hparams.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
    #         kld_loss = self.hparams.beta * kld_loss
    #     elif self.hparams.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
    #         self.C_max = self.C_max.to(input.device)
    #         C = torch.clamp(self.hparams.C_max/self.hparams.C_stop_iter * self.hparams.num_iter, 0, self.hparams.C_max.data[0])
    #         kld_loss + self.hparams.gamma * (kld_loss - C).abs()
    #     else:
    #         raise ValueError('Undefined loss type.')
    #
    #     return recons_loss, kld_loss


    def training_step(self, batch, batch_idx):
        x = batch

        recon_batch, mu, logvar, latent_sample = self(x)

        rec_loss, kld = self.loss(
            x, recon_batch, (mu, logvar), self.training, latent_sample)

        if self.current_epoch == 0:
            self.warm_up += x.shape[0]
            warm_up_weight = self.warm_up / self.hparams.trainset_size
        else:
            warm_up_weight = 1

        loss = rec_loss + warm_up_weight * kld

        self.log("kl_warm_up", warm_up_weight,)

        self.log("kld", kld)

        self.log("rec_loss", rec_loss)

        self.log("Training Loss", loss )

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch

        recon_batch, mu, logvar, latent_sample = self(x)

        rec_loss, kld_val = self.loss(
            x, recon_batch, (mu, logvar) ,self.training, latent_sample)

        val_loss = rec_loss + kld_val

        self.log(
            "Validation Loss", val_loss
        )

        self.log("val_kld", kld_val)

        if batch_idx < 10:
            recon = self.forward_pass(batch)
            transform = T.ToPILImage()
            for i in range(2):
                img = transform(batch[i])
                img.save(os.path.join(self.hparams.log_dir, f'{self.current_epoch}_{batch_idx}_{i}val_real.png'))
                recon_img = transform(recon[i])
                recon_img.save(os.path.join(self.hparams.log_dir, f'{self.current_epoch}_{batch_idx}_{i}val_recon.png'))
        return val_loss

    def test_step(self, batch, batch_idx):
        x = batch

        recon_batch, mu, logvar, latent_sample = self(x)

        rec_loss, kld_val = self.loss(
            x, recon_batch, (mu, logvar), self.training, latent_sample)
        val_loss = rec_loss + kld_val

        self.log(
            "Test Loss", val_loss
        )

        self.log("test_kld", kld_val)

        if batch_idx < 10:
            recon = self.forward_pass(batch)
            transform = T.ToPILImage()
            for i in range(2):
                img = transform(batch[i])
                img.save(os.path.join(self.hparams.log_dir, f'{self.current_epoch}_{batch_idx}_{i}val_real.png'))
                recon_img = transform(recon[i])
                recon_img.save(os.path.join(self.hparams.log_dir, f'{self.current_epoch}_{batch_idx}_{i}val_recon.png'))
        return val_loss

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