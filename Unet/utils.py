import torch
import os
import numpy as np
from glob import glob
import logging
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt


class preproccesing(dataset):
    def __init__(self, imgs_dir, masks_dir, mask_suffix=""):
        # temp solution 
        for i in imgs_dir:
            for j in i:
                im = Image.open(f"{j}.tif") 
                imarray = numpy.array(im) 
                imarray.shape 
                plt.imshow(imarray)
                plt.show()

    def __getitem__(self, i):
        pass

if __name__ == "__main__":
    print("utility")
    
