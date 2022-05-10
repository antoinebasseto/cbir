import os
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
import numpy as np
import re
from utils import load_data, read_image


class ImageDataModule(pl.LightningDataModule):

    def __init__(self, root_dir, batch_size = 32, transform=None):

        super().__init__()
        self.data_dir = root_dir
        self.batch_size = batch_size
        self.transform = None  # define dataset specific transforms here

        self.collate_fn = collate_fn

    def prepare_data(self):
        # download data if necessary
        pass

    def setup(self, stage: Optional[str] = None):
        """
        There are also data operations you might want to perform on every GPU. Use setup() to do things like:
            - count number of classes
            - build vocabulary
            - perform train/val/test splits
            - create datasets
            - apply transforms (defined explicitly in your datamodule)
        :param stage: fit, test, predict
        :return: Nothing
        """

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_data = np.load(self.data_dir / 'val.npz')
            eeg = torch.from_numpy(train_data['EEG']).float()
            labels = torch.from_numpy(train_data['labels']).float()
            self.train_set = XRayDataset(self.data_dir, labels)

            val_data = np.load(self.data_dir / 'val.npz')
            eeg = torch.from_numpy(val_data['EEG']).float()
            labels = torch.from_numpy(val_data['labels']).float()
            self.val_set = TensorListDataset(eeg, labels)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            data = np.load(self.data_dir / 'test.npz')
            eeg = torch.from_numpy(data['EEG']).float()
            labels = torch.from_numpy(data['labels']).float()
            self.test_set = TensorListDataset(eeg, labels)

        if stage == "predict" or stage is None:
            self.predict_set = NotImplemented

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)

    def predict_dataloader(self):
        raise NotImplementedError("We do not have a predict set in this datamodule")

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
                                os.path.isfile(os.path.join(img_dir, name)) and re.search('png$', name) is not None])
        data, train_names, test_names, bounding_boxes = load_data(filenames_to_keep=images_name)

        # Keep only the training or test ones
        self.data = data[data["Filename"].isin((train_names if train else test_names))]
        self.bounding_boxes = bounding_boxes[bounding_boxes["Filename"].isin((train_names if train else test_names))]
        self.images_name = images_name[np.isin(images_name, (train_names if train else test_names))]

        self.transform = transform

    def __len__(self):
        '''
        Overload: return the length of the dataset

        Return
        ----------
        length: înt:
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

        if (len(image.shape) > 2):  # Some images have more than one channel for unknown reasons
            image = image[:, :, 0]

        data = self.data[self.data["Filename"] == self.images_name[idx]]

        bbox = None
        if not self.bounding_boxes[self.bounding_boxes["Filename"] == self.images_name[idx]].empty:
            bbox = self.bounding_boxes[self.bounding_boxes["Filename"] == self.images_name[idx]]

        if self.transform:
            image = self.transform(image)
            if image.shape[0] == 1:
                image = image.squeeze()

        return image  # , data, bbox # Cannot pass data and bbox for the moment because they contain entries of
        # type "object" which is a problem for dataloader. Should transform the few entries having object type