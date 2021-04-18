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


# TBD
img_dir = "../../data/BrainMRI"
mask_dir = "../../data/BrainMri"
checkpoints = "./pretrained"


def Train_this_mf(net, device, epochs, batch_size, lr, val_per=.1, save_cp=True, img_scale=.5):

    dataset = "   "

