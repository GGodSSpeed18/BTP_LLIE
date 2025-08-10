"""
Training Script for ZERO-IG

Expected directory structures:

Ground truth:
  data_root/
    1/
      high/
        i1.png      <-- ground truth
      low/
        i1.png      <-- low-light image
    2/
      high/
        i2.png
      low/
        i2.png
    ...

This script trains the ZERO-IG model on each low-light image in the dataset.
"""

import os
import sys
import time
import glob
import numpy as np
import utils
from PIL import Image
import logging
import argparse
import torch
import torch.utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model import *
from multi_read_data import DataLoader

parser = argparse.ArgumentParser("ZERO-IG")
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--epochs', type=int, default=2001, help='epochs per image')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--data_root', type=str, required=True, 
                   help='Root directory containing image pairs (main directory)')
parser.add_argument('--save_root', type=str, default='./results',
                   help='Root location for saving results')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("WARNING: It looks like you have a CUDA device, but aren't using CUDA.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def save_images(tensor):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')
    return im

def process_image_pair(img_dir, save_root):
    # unique save directory for this image pair
    img_id = os.path.basename(img_dir)
    save_dir = os.path.join(save_root, img_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # logging
    fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    
    try:
        model = Network()
        model.enhance.in_conv.apply(model.enhance_weights_init)
        model.enhance.conv.apply(model.enhance_weights_init)
        model.enhance.out_conv.apply(model.enhance_weights_init)
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=3e-4)

        # low-light image path
        low_dir = os.path.join(img_dir, 'low')

        if not low_dir:
            logging.warning(f"No low-light images found in {low_dir}")
            return
        # dataLoader for single image
        TrainDataset = DataLoader(img_dir=low_dir, task='train')
        train_queue = torch.utils.data.DataLoader(
            TrainDataset, batch_size=args.batch_size,
            pin_memory=True, num_workers=0, shuffle=False, 
            generator=torch.Generator(device='cuda'))

        epoch_save_dir = os.path.join(save_dir, 'model_epochs')
        os.makedirs(epoch_save_dir, exist_ok=True)
        utils.save(model, os.path.join(epoch_save_dir, 'initial_weights.pt'))
        
        model.train()
        for epoch in range(args.epochs):
            losses = []
            for idx, (input, _) in enumerate(train_queue):
                input = Variable(input, requires_grad=False).cuda()
                optimizer.zero_grad()
                loss = model._loss(input, epoch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                losses.append(loss.item())
                
                logging.info(f'{img_id} - Epoch {epoch:03d} Iter {idx:03d} Loss: {loss.item():.4f}')

            # save model weights
            # torch.save(model.state_dict(), os.path.join(epoch_save_dir, f'weights_{epoch}.pt'))
            if epoch%50==49:
                utils.save(model, os.path.join(epoch_save_dir, 'weights_%d.pt' % epoch))

            logging.info(f'{img_id} - Epoch {epoch:03d} Average Loss: {np.mean(losses):.4f}')

    except Exception as e:
        logging.error(f"Error processing {img_id}: {str(e)}")
    finally:
        # cleaning up logging handler
        logging.getLogger().removeHandler(fh)
        fh.close()

def main():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    args.save_root = os.path.join(args.save_root, f'train-{timestamp}')
    os.makedirs(args.save_root, exist_ok=True)

    # list all image pairs in data root
    image_dirs = []
    for entry in os.scandir(args.data_root):
        if entry.is_dir():
            if os.path.exists(os.path.join(entry.path, 'low')) and os.path.exists(os.path.join(entry.path, 'high')):
                image_dirs.append(entry.path)

    if not image_dirs:
        raise ValueError(f"No valid image pairs found in {args.data_root}")
        
    for img_dir in image_dirs:
        logging.info(f"Starting processing for {os.path.basename(img_dir)}")
        process_image_pair(img_dir, args.save_root)
        logging.info(f"Completed processing for {os.path.basename(img_dir)}")

if __name__ == '__main__':
    main()
