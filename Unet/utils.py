import torch
import numpy as np
import os
from os import listdir
from glob import glob
import logging
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt


#class preproccesing(dataset):
#    def __init__(self, imgs_dir, masks_dir, mask_suffix=""):
#        # temp solution 
#        for i in imgs_dir:
#            for j in i:
#                im = Image.open(f"{j}.tif") 
#                imarray = numpy.array(im) 
#                imarray.shape 
#                plt.imshow(imarray)
#                plt.show()
#
#    def __getitem__(self, i):
#        pass

class corona_dataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, transform):
        self.imags_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.scale = scale

        assert 0 < scale <= 1, 'scale must be between 0 and 1'

        self.ids = [os.path.splitext(file)[0].split("_mask")[0] for file in listdir(masks_dir)
                if not file.startswith('.')]
        
        
        print(len(self.ids))
        logger.info(f"Createing dataset with {len(self.ids)} example")
        print(f"Createing dataset with {len(self.ids)} example : ")

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_image, scale):
        w, h = pil_image.size
        newW, newH = int(scale * w), int(scale * h)
        pil_image = pil_image.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        img_trans = img_nd.transpose((2,0,1))
        if img_Trans.max() > 1:
            img_trans = img_trans / 255

        
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        self.mask_file = glob(self.mask_dir + idx + "_mask" + ".*")
        self.img_file = glob(self.imgs_dir + idx + ".*")

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        print("dataset preprocessing")
        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        logger.info('dataset preprocessing')
        
        return {
            'image': self.transform(torch.from_numpy(img).type(torch.FloatTensor)),
            'mask': self.transform(torch.from_numpy(mask).type(torch.FloatTensor))
        }

if __name__ == "__main__":
    imgs = "../../data/covid19_chest_xray/images/"
    mask = "../../data/covid19_chest_xray/mask/"
    corona_dataset(imgs, mask)  
