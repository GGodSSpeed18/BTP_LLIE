"""
Testing Script for ZERO-IG

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
      low/s
        i2.png
    ...

Model weights:
    model_root/
        1/
            model_epochs/
                weights_1999.pt  <-- Model weights for image ID 1 (total epochs 2000)
        2/
            model_epochs/
                weights_1999.pt   
        ...    

This script tests the ZERO-IG model weights on each corresponding low-light image in the dataset.
"""

import os
import sys
import logging
import argparse
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
from model import Finetunemodel, Network
from multi_read_data import DataLoader

# configure root directory access
root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir)

parser = argparse.ArgumentParser("ZERO-IG Testing")
parser.add_argument('--data_root', type=str, required=True,
                   help='Root directory with numbered image folders')
parser.add_argument('--model_root', type=str, required=True,
                   help='Root directory containing trained models')
parser.add_argument('--save_root', type=str, default='./test_results',
                   help='Root location for saving results')
parser.add_argument('--final_epoch', type=int, default=2000,
                   help='Final epoch number to load weights from')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

args = parser.parse_args()

# configure logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

def save_images(tensor):
    """Convert tensor to numpy image array"""
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    return np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')

def process_single_image(img_id, model_root, data_root, save_root):
    try:
        # path setup
        model_dir = os.path.join(model_root, str(img_id))
        data_dir = os.path.join(data_root, str(img_id))
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found for image {img_id}")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found for image {img_id}")

        save_dir = os.path.join(save_root, str(img_id))
        os.makedirs(save_dir, exist_ok=True)

        # load model weights
        fnl = args.final_epoch-1
        weights_path = os.path.join(model_dir, 'model_epochs', f'weights_{fnl}.pt')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found at {weights_path}")

        # initialize model using finetunemodel, will work better than simply loading model weights
        # model = Network()
        # model.load_state_dict(torch.load(weights_path))
        model = Finetunemodel(weights_path)
        model = model.cuda()
        model.eval()

        low_dir = os.path.join(data_dir, 'low')
        TestDataset = DataLoader(img_dir=low_dir, task='test')
        test_queue = torch.utils.data.DataLoader(
            TestDataset, batch_size=1,
            pin_memory=True, num_workers=0,
            shuffle=False
        )

        with torch.no_grad():
            for _, (input, img_name) in enumerate(test_queue):
                input = Variable(input, volatile=True).cuda()
                enhance, output = model(input)
                enhance=save_images(enhance)
                output = save_images(output)
                # save results
                base_name = os.path.splitext(os.path.basename(img_name[0]))[0]
                Image.fromarray(output).save(
                    os.path.join(save_dir, f'{base_name}_denoise.png')
                )
                Image.fromarray(enhance).save(
                    os.path.join(save_dir, f'{base_name}_enhance.png')
                )

        logging.info(f'Successfully processed image {img_id}')

    except Exception as e:
        logging.error(f"Error processing image {img_id}: {str(e)}")

def main():
    os.makedirs(args.save_root, exist_ok=True)

    # get all trained image IDs
    trained_images = sorted(
        [int(d.name) for d in os.scandir(args.model_root) if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x)
    )

    for img_id in trained_images:
        logging.info(f"Processing image ID: {img_id}")
        process_single_image(
            img_id=img_id,
            model_root=args.model_root,
            data_root=args.data_root,
            save_root=args.save_root
        )
    torch.set_grad_enabled(True)

if __name__ == '__main__':
    main()
