import matplotlib.pyplot as plt
import torch


def run_training(model, train_loader, val_loader, loss_fn, optimizer, num_epochs = 10, verbose=1):
    train_history = []
    val_history = []
    for e in range(num_epochs):
        if verbose:
            print(f"=========== EPOCH {e} ===========")
        train_loss = train_loop(model, train_loader, loss_fn, optimizer, verbose)
        val_loss = validation_loop(model, val_loader, loss_fn,  verbose)

        train_history += train_loss
        val_history += val_loss

    return train_history, val_history


def train_loop(autoencoder, train_loader, loss_fn, optimizer,  verbose=1):
    '''
    Train the autoencoder.

    Parameters
    ----------
    autoencoder: nn.Module
        The autoencoder model.
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
    total_loss = 0
    number_samples = 0
    for i, batch in enumerate(train_loader):
            # X, data, bbox = batch
        X = batch
        output = autoencoder(X)
        loss = loss_fn(output, X)
        total_loss += loss.detach()
        number_samples += X.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and not i % 100 and not i == 0:
            print(f"Done processing batch number {i}")

        if verbose:
            print(f"Mean loss = {total_loss / number_samples}")

    loss_history = total_loss / number_samples
    return loss_history


def validation_loop(model, val_loader, loss_fn, verbose=1):
    '''
    test
    Parameters
    ----------
    model: nn.Module model
    train_loader: DataLoader
        A Pytorch Dataloader containing the training images.
    verbose: optional, int
        Set the number of printing we want during training. For the moment, only 0 for no printing or another value
        to get some information about the update.
    Return
    ----------
    loss_history: list
        A list of training losses. One for each epoch of training.
    '''
    loss_history = []
    with torch.no_grad:
        total_loss = 0
        number_samples = 0
        for i, batch in enumerate(val_loader):
            # X, data, bbox = batch
            X = batch
            output = model(X)
            loss = loss_fn(output, X)
            total_loss += loss.detach()
            number_samples += X.shape[0]

            if verbose and not i % 100 and not i == 0:
                print(f"Done processing batch number {i}")

        if verbose:
            print(f"Mean loss = {total_loss / number_samples}")

        loss_history.append(total_loss / number_samples)
    return loss_history



def test_loop(model, test_loader, loss_fn, verbose=1):
    '''
    test
    Parameters
    ----------
    model: nn.Module model
    train_loader: DataLoader
        A Pytorch Dataloader containing the training images.
    verbose: optional, int
        Set the number of printing we want during training. For the moment, only 0 for no printing or another value
        to get some information about the update.
    Return
    ----------
    loss_history: list
        A list of training losses. One for each epoch of training.
    '''
    loss_history = []
    with torch.no_grad:
        total_loss = 0
        number_samples = 0
        for i, batch in enumerate(test_loader):
            # X, data, bbox = batch
            X = batch
            output = model(X)
            loss = loss_fn(output, X)
            total_loss += loss.detach()
            number_samples += X.shape[0]

            if verbose and not i % 100 and not i == 0:
                print(f"Done processing batch number {i}")

        if verbose:
            print(f"Mean loss = {total_loss / number_samples}")

        loss_history.append(total_loss / number_samples)
    return loss_history


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
    plt.show()