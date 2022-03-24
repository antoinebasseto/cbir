import imageio
import matplotlib.patches as patches
import glob
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim

from models.autoencoder import AutoEncoder, Encoder, Decoder
from dataset.dataset import XRayDataset
from train import run_training, plot_loss

IMG_DIR = "data/images/images_001/"
BATCH_SIZE = 2
NUM_EPOCHS = 2
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
    #put data sets into Dataloader
    train_loader = DataLoader(training_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=True)
    # %%
    # Create our autoencoder and train it
    # Parameters
    #data moving to gpu if available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Model
    autoencoder = AutoEncoder(
        encoder=Encoder(
            image_dimension=training_data[0].shape,
            encoded_dimension=LATENT_SPACE_DIM,
        ).to(device),
        decoder=Decoder(
            image_dimension=training_data[0].shape,
            encoded_dimension=LATENT_SPACE_DIM,
        ).to(device)
    )

    # Loss function & optimizer
    loss_fn = nn.MSELoss()  # Mean squared error
    optimizer = optim.Adam(autoencoder.parameters(),
                           lr=LEARNING_RATE)

    epoch_losses = run_training(autoencoder, train_loader, num_epochs=NUM_EPOCHS)
    plot_loss(epoch_losses, 'Training Loss - Simple Model')

if __name__ == "__main__":
    main()