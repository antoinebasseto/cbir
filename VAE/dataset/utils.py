import pandas as pd
import numpy as np
from PIL import Image


def load_data(folder = "../backend-project/data/", filenames_to_keep=None):
    '''
    Load the data about each image, the names of the images used for training, the name of the ones used for testing
    and the bounding boxes.

    Parameters
    ----------
    folder : optional, str
        The folder where the data is stored
    filenames_to_keep : optional, np.array(M,)
        List of all the filenames we want to keep, the other will be filtered. Default is None, meaning we
        want to keep all the data in the folder.

    Return
    ----------
    data, train_val_filenames, test_filenames, bounding_boxes:
    pd.DataFrame(N, 11), np.array(V, ), np.array(T, ),  pd.DataFrame(B, 6)
        The data contains the info on the image such as disease, patient id/age, resolution.
        The train_val_filenames contains the name of the images used for training/validation
        The test_filenames contains the name of the images used for testing
        The bounding_boxes contains the info on the bboxes of known diseases which are the name of the file
        and the disease, the x,y coordinates of the top left corner and the width and height of the box.
    '''
    data = pd.read_csv(folder + "Data_Entry_2017_v2020.csv")
    train_val_filenames = np.array(pd.read_csv(folder + "train_val_list.txt", names=["filename"])['filename'].tolist())
    test_filenames = np.array(pd.read_csv(folder + "test_list.txt", names=["filename"])['filename'].tolist())
    bounding_boxes = pd.read_csv(folder + "BBox_List_2017.csv")

    data = data.rename(
        columns={"Image Index": "Filename", "Finding Labels": "Diseases", "OriginalImage[Width": "Original Width",
                 "Height]": "Original Height", "OriginalImagePixelSpacing[x": "Original Pixel Spacing x",
                 "y]": "Original Pixel Spacing y"})
    bounding_boxes = bounding_boxes.loc[:, ~bounding_boxes.columns.str.contains('^Unnamed')].rename(
        columns={"Image Index": "Filename", "Finding Label": "Disease", "Bbox [x": "x", "h]": "h"})

    if filenames_to_keep is not None:
        data = data[data["Filename"].isin(filenames_to_keep)]
        bounding_boxes = bounding_boxes[bounding_boxes["Filename"].isin(filenames_to_keep)]
        test_filenames = test_filenames[np.isin(test_filenames, filenames_to_keep)]
        train_val_filenames = train_val_filenames[np.isin(train_val_filenames, filenames_to_keep)]

    return data, train_val_filenames, test_filenames, bounding_boxes

def read_image(path):
    '''
    Read an image in memory

    Parameters
    ----------
    path : str
        The path to the image (folder + filename)

    Return
    ----------
    np_img: np.array(X, Y):
        The image (not scaled) stored in this path
    '''
    img = Image.open(path)
    np_img = np.array(img)

    return np_img

def scale_image(np_img):
    '''
    Scale an image from range [0, 255] to range [0, 1]

    Parameters
    ----------
    np_img : np.array(X, Y)
        The unscaled ([0, 255]) image

    Return
    ----------
    np.array(X, Y):
        The scaled ([0, 1]) image
    '''
    return np_img/255