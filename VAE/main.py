import imageio
import matplotlib.patches as patches
import glob
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
import torch.optim as optim

from models.autoencoder import AutoEncoder, Encoder, Decoder
from models.beta_vae import BetaVAE
from dataset.dataset import XRayDataset
from train import run_training, plot_loss

IMG_DIR = "../backend-project/data/images/images_001/"
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
LATENT_SPACE_DIM = 16


def main():
    """
    Example training
    """
    #initializing data sets
    training_data = XRayDataset(
        img_dir=IMG_DIR,
        train=True,
        transform=ToTensor()  # This method directly scale the image in [0, 1] range
    )

    test_data = XRayDataset(
        img_dir=IMG_DIR,
        train=False,
        transform=ToTensor()  # This method directly scale the image in [0, 1] range
    )
    #put data sets into Dataloader. We only use training and validation set, no test sets here.
    train_loader = DataLoader(training_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    val_loader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

    # Create our autoencoder and train it
    # Parameters
    #data moving to gpu if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Model
    beta_vae = BetaVAE(in_channels=1, latent_dim=LATENT_SPACE_DIM)

    # Loss function & optimizer
    loss_fn = beta_vae.loss_function
    optimizer = optim.Adam(beta_vae.parameters(),
                        lr=LEARNING_RATE)

    train_history, val_history = run_training(beta_vae, train_loader, val_loader, loss_fn, optimizer, num_epochs=NUM_EPOCHS)
    
    plot_loss(train_history, 'Training Loss')
    plot_loss(val_history, 'Validation Loss')

if __name__ == "__main__":
    main()