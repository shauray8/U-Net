import torch
import numpy as np
import os
from os import listdir
from glob import glob
import logging
from torch.utils.data import Dataset
from PIL import Image, ImageOps
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

image_size = 64

class corona_dataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale, transform):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.scale = scale

        assert 0 < scale <= 1, 'scale must be between 0 and 1'

        self.ids = [os.path.splitext(file)[0].split("_mask")[0] for file in listdir(masks_dir)
                if not file.startswith('.')]
        
        
        print(len(self.ids))
        logging.info(f"Createing dataset with {len(self.ids)} example")
        print(f"Createing dataset with {len(self.ids)} example")

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_image, scale):
        w, h = pil_image.size
        newW, newH = int(scale * w), int(scale * h)
        pil_image = pil_image.resize((newW, newH))

        img_nd = np.array(pil_image)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        img_trans = img_nd.transpose((2,0,1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        
        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        self.mask_file = glob(self.masks_dir + idx + "_mask" + ".*")
        self.img_file = glob(self.imgs_dir + idx + ".*")

        assert len(self.img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {self.img_file}'
        mask = Image.open(self.mask_file[0])
        img = Image.open(self.img_file[0])
        mask = mask.resize((image_size, image_size))
        img = img.resize((image_size, image_size))

        img = self.preprocess(ImageOps.grayscale((img)), self.scale)
        mask = self.preprocess(ImageOps.grayscale((mask)), self.scale)
        logging.info('dataset preprocessing')
        
        return {
            'image': torch.from_numpy((img)).type(torch.FloatTensor),
            'mask': torch.from_numpy((mask)).type(torch.FloatTensor)
        }


def eval(net, loader, device):
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  
    tot = 0
    
    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        if net.n_classes > 1:
            tot += F.cross_entropy(mask_pred, true_masks).item()
        else:
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            #tot += dice_coeff(pred, true_masks).item()

    net.train()
    return tot / n_val

if __name__ == "__main__":
    imgs = "../../data/covid19_chest_xray/images/"
    mask = "../../data/covid19_chest_xray/mask/"
    #corona_dataset(imgs, mask, 1, transform)  
