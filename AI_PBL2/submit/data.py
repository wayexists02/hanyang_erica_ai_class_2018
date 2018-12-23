import numpy as np
import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def read(dataset_img, dataset_lbl, path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    # if dataset is "training":
    #     fname_img = os.path.join(path, 'train-images.idx3-ubyte')
    #     fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    # elif dataset is "testing":
    #     fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
    #     fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    # else:
    #     raise Exception("dataset must be 'testing' or 'training'")

    fname_img = os.path.join(path, dataset_img)
    fname_lbl = os.path.join(path, dataset_lbl)

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return lbl, img


def read_test_data(dataset, path = "."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    # if dataset is "training":
    #     fname_img = os.path.join(path, 'train-images.idx3-ubyte')
    #     fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    # elif dataset is "testing":
    #     fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
    #     fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    # else:
    #     raise Exception("dataset must be 'testing' or 'training'")

    fname_img = os.path.join(path, dataset)
    img = None

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(-1, rows, cols)

    return img

def show_sample(data, label=None):
    plt.imshow(data.reshape(28, 28), cmap='gray')
    plt.show()
    
    if label is not None:
        print(label)