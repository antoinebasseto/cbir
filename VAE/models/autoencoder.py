import torch.nn as nn

class Encoder(nn.Module):
    '''
    An encoder that flattens the image first and then only uses 2 linear layers. Subclass of nn.Module.
    '''

    def __init__(self, image_dimension, encoded_dimension=16):
        '''
        Constructor

        Parameters
        ----------
        image_dimension: Tuple(int, int):
            The image dimension of a single image.
        encoded_dimension: int
            The dimension for the final encoding.
        '''
        super().__init__()

        flat_image_dimension = image_dimension[0] * image_dimension[1]

        self.model = nn.Sequential(
            nn.Linear(in_features=flat_image_dimension, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=encoded_dimension),
            nn.ReLU(),
        )

    def forward(self, images):
        """Overload. Encode the image."""
        batch_size = images.shape[0]
        flat_images = images.view(batch_size, -1)
        return self.model(flat_images)


class Decoder(nn.Module):
    '''
    A decoder that only uses linear layers. It's the counterpart to the Encoder. Subclass of nn.Module.
    '''

    def __init__(self, image_dimension, encoded_dimension=16):
        '''
        Constructor

        Parameters
        ----------
        image_dimension: Tuple(int, int):
            The image dimension of a single image.
        encoded_dimension: int
            The dimension for the final encoding.
        '''
        super().__init__()

        self.image_dimension = image_dimension
        flat_image_dimension = image_dimension[0] * image_dimension[1]

        self.model = nn.Sequential(
            nn.Linear(in_features=encoded_dimension, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=flat_image_dimension),
            nn.Sigmoid(),
        )

    def forward(self, encodings):
        """Overload: Decode the latent representation."""
        batch_size = encodings.shape[0]
        flat_images = self.model(encodings)
        images = flat_images.view(batch_size, *self.image_dimension)
        return images


class AutoEncoder(nn.Module):
    '''
    An auto-encoder model. Subclass of nn.Module.
    '''

    def __init__(self, encoder, decoder):
        '''
        Constructor

        Parameters
        ----------
        encoder: nn.Module
            The encoder
        decoder: nn.Module
            The decoder
        '''
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images):
        """Overload: Decode the latent representation encoded by itself."""
        return self.decoder(self.encoder(images))