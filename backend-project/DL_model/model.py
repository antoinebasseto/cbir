import numpy
import torch.nn as nn
import numpy as np

class BaseModel(nn.Module):
    '''
    An encoder that flattens the image first and then only uses 2 linear layers. Subclass of nn.Module.
    '''

    def __init__(self,  encoded_dimension=16):
        '''
        Constructor

        Parameters
        ----------
        image_dimension: Tuple(int, int):
            The image dimension of a single image.
        encoded_dimension: int
            The dimension for the final encoding.
            @rtype:
        '''
        super().__init__()
        self.encoded_dimension = encoded_dimension

    #abstract method forward also has to be implmented

    def get_embedding(self, image):
        return numpy.zeros(self.encoded_dimension)


def get_model():
    """
    @return: backend model to be used
    """
    model = BaseModel()
    return model

def get_embedding(image, model: BaseModel):
    """

    @param image: image file
    @param model:
    @return:
    """
    return model.get_embedding(image)