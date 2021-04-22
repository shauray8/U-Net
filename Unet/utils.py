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

class corona_dataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imags_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale

        assert 0 < scale <= 1, 'scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs)
                if not file.startwidth('.')]

        logging.info(f"Createing dataset with {len(self.ids)} example")


if __name__ == "__main__":
    print("utility")
    
