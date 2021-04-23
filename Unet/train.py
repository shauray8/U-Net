import torch
import logging
import sys, os

import numpy as np
from torch import optimm
from tqdm import trange

from model import UNet

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from utils import *


imgs = "../../data/covid19_chest_xray/images/"
mask = "../../data/covid19_chest_xray/mask/"
checkpoints = "./pretrained"


def Train_this_mf(net, device, epochs, batch_size, lr, val_per=.1, save_cp=True, img_scale=.5):

    dataset = corona_dataset(dir_image, dir_mask, img_scale)

    # train validation split
    val_set = int(len(dataset) * val_per)
    train_set = len(dataset) - val_set
    train, val = random_split(dataset, [train_set, val_set])
    
    # loading the data
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8,
            pin_memory=True)
    validation_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8,
            pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f"LR{lr_BS_{batch_size}_SCALE_{img_scale}}")
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {train_set}
        Validation size: {val_Set}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale} ''')



