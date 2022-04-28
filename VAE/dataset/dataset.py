import os
from torch.utils.data import Dataset
import numpy as np
import re
from dataset.utils import load_data, read_image

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
        images_name = np.array([name for name in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, name)) and re.search('png$', name) is not None])
        data, train_names, test_names, bounding_boxes = load_data(filenames_to_keep = images_name)
    
        # Keep only the training or test ones
        self.data = data[data["Filename"].isin((train_names if train else test_names))]
        self.bounding_boxes = bounding_boxes[bounding_boxes["Filename"].isin((train_names if train else test_names))]
        self.images_name = images_name[np.isin(images_name, (train_names if train else test_names))]
        self.images_name = images_name
        self.transform = transform
        
    def __len__(self):
        '''
        Overload: return the length of the dataset

        Return
        ----------
        length: int:
            The length of the dataset
        '''
        return self.images_name.shape[0]

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
        
        if(len(image.shape) > 2): #Some images have more than one channel for unknown reasons
            image = image[:, :, 0]

        data = self.data[self.data["Filename"] == self.images_name[idx]]
        
        bbox = None
        if not self.bounding_boxes[self.bounding_boxes["Filename"] == self.images_name[idx]].empty:
            bbox = self.bounding_boxes[self.bounding_boxes["Filename"] == self.images_name[idx]]

        if self.transform:
            image = self.transform(image)

        return image#, data, bbox # Cannot pass data and bbox for the moment because they contain entries of type "object" which is a problem for dataloader. Should transform the few entries having object type