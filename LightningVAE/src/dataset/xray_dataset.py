import os
from typing import Optional

import pytorch_lightning as pl
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re


class ImageDataModule(pl.LightningDataModule):

    def __init__(self, root_dir, batch_size=32, input_dim=32, transform=None, collate_fn=None):
        super().__init__()
        self.data_dir = root_dir
        self.batch_size = batch_size
        # define dataset specific transforms here
        if transform is None:
            self.transform = T.Compose([T.Resize((input_dim, input_dim)), T.ToTensor()])
        else:
            self.transform = T.Compose([transform, T.Resize((input_dim, input_dim)), T.ToTensor()])
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

            train_directory = os.path.join(self.data_dir, "HAM10000_images_part_1")
            self.train_set = XRayDataset(train_directory, transform=self.transform, train=True)

            val_directory = os.path.join(self.data_dir, "HAM10000_images_part_2")
            self.val_set = XRayDataset(val_directory, transform=self.transform, train=False)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            test_directory = os.path.join(self.data_dir, "HAM10000_images_part_2")
            self.test_set = XRayDataset(test_directory, transform=self.transform, train=False)

        if stage == "predict" or stage is None:
            self.predict_set = NotImplemented

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True
                          , num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=False,
                          num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=False)

    def predict_dataloader(self):
        raise NotImplementedError("We do not have a predict set in this datamodule")

class XRayDataset(Dataset):
    '''
    Class to represent our dataset of XRay images. Subclass of torch.utils.data.Dataset. Store the image directory,
    images name, bounding boxes and data of the images we will use either for training or testing.
    '''

    def __init__(self, img_dir, transform=None, train=True, return_labels=False, meta_file=None):
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
                                os.path.isfile(os.path.join(img_dir, name)) and
                                (re.search('png$', name) is not None or re.search('jpg$', name) is not None)])
        # data, train_names, test_names, bounding_boxes = load_data(filenames_to_keep=images_name)
        #
        # # Keep only the training or test ones
        # self.data = data[data["Filename"].isin((train_names if train else test_names))]
        # self.bounding_boxes = bounding_boxes[bounding_boxes["Filename"].isin((train_names if train else test_names))]
        # self.images_name = images_name[np.isin(images_name, (train_names if train else test_names))]
        self.images_name = images_name
        self.transform = transform
        self.train = train
        self.return_labels = return_labels# to be implemented

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

        img = Image.open(img_path)
        if self.transform:
            image = self.transform(img)

        if self.return_labels:
            raise NotImplementedError("We do not have a return_labels in this datamodule")
        return image