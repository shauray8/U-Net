import torch
import torch.nn as nn
import argparse
import logging
from torchvision import transforms as T
import sys, os

import numpy as np
from torch import optim
from tqdm import trange
import tqdm

from model import U_net

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from utils import *


imags = "../../data/covid19_chest_xray/images/"
mask = "../../data/covid19_chest_xray/mask/"
checkpoints = "./pretrained"
image_size = 128

transform = []
transform.append(T.Resize(image_size))
#transform.append(T.ToTensor())
transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
transform = T.Compose(transform)


def Train_this_mf(net, device, epochs, batch_size, lr, val_per=.1, save_cp=True, img_scale=.5):

    dataset = corona_dataset(imgs_dir=imags, masks_dir=mask,
            scale = img_scale, transform = transform)

    # train validation split
    val_set = int(len(dataset) * val_per)
    train_set = len(dataset) - val_set
    train, val = random_split(dataset, [train_set, val_set])
    
    # loading the data
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0,
            pin_memory=True)
    validation_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0,
            pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f"LR_{lr}_BS_{batch_size}_SCALE_{img_scale}")
    global_step = 0

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.BCEWithLogitsLoss()

    print(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {train_set}
        Validation size: {val_set}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale} ''')

    logging.info(f'''Starting training:
        Net:             {net}       
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {train_set}
        Validation size: {val_set}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale} ''')

    
    for epoch in (l := trange(epochs)):
        net.train()

        epoch_loss = 0
        with tqdm.tqdm(total=train_set, desc=f"Epoch {epoch + 1}/{epochs}", unit="img") as sigma:
            for batch in train_loader:
                imgs = batch["image"]
                true_mask = batch["mask"]
                assert imgs.shape[1] == net.n_channel, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'
            
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device, dtype=mask_type)

            masks_pred = net(imgs)
            loss = loss_function(masks_pred, trye_masks)
            epoch_loss += loss.item()
            writer.add_scalar("loss/train", loss.item(), global_step)

            pbar.set_postfix(**{"loss (batch)" : loss.item()})

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

            pbar.update(imgs.shape[0])
            global_step += 1

            if global_step % batch_size == 0:
                for tag, value in net.named_parameters():
                    tag = tag.replace(".","/")
                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(),
                            global_step)
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(),
                            global_step)

                val_score = eval_net(net, val_loader, device)
                scheduler.step(val_score)
                writer.add_scalar('learning_rate', optimizeer.param_groups[0][lr],global_step)


                if net.n_classes > 1:
                    print('Validation cross entropy: {}'.format(val_score))
                    logging.info('Validation cross entropy: {}'.format(val_score))
                    writer.add_scalar('Loss/test', val_score, global_step)
                else:
                    print('Validation Dice Coeff: {}'.format(val_score))
                    logging.info('Validation Dice Coeff: {}'.format(val_score))
                    writer.add_scalar('Dice/test', val_score, global_step)

                writer.add_images('images', imgs, global_step)
                if net.n_classes == 1:
                    writer.add_images('masks/true', true_masks, global_step)
                    writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5,
                            global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                print('Created checkpoint directory')
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                   dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
            print(f'Checkpoint {epoch + 1} saved !')

    writer.close()



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=32,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,filename="./log/maybe_imp.log",
            format="format='%(asctime)s --> %(levelname)s: %(message)s") 

    args = get_args()
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    net = U_net(n_channel=1, n_classes=1, bilinear=True)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device)

    try:
        Train_this_mf(net=net,
                  device = device,  
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
