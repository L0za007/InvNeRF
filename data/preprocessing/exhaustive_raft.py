"""
This script computes all pairwise RAFT optical flow fields
for each pair, we use previous flow as initialization to compute the current flow
"""
import sys
sys.path.append('core')

import argparse, torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from core.raft import RAFT
from core.utils.utils import InputPadder
import warnings

warnings.filterwarnings("ignore")

DEVICE = 'cuda'


def load_image(imfile, scale=1.0):
    img = Image.open(imfile)
    if scale<1.0:
        img = img.resize((int(img.width*scale), int(img.height*scale)), Image.LANCZOS)
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def run_exhaustive_flow(args):
    print("This function does not re-write the RAFT flow files. Please delete the folders RAFT and features if you want to re-write the files")
    # Load the raft model
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    # Send model to device
    model = model.module
    model.to(DEVICE)
    model.eval()
    # Read the data directory
    data_dir = Path(args.data_dir)
    print('computing all pairwise optical flows for {}...'.format(data_dir))
    # Create output directory
    flow_out_dir = data_dir / 'RAFT' / 'flows'
    flow_out_dir.mkdir(exist_ok=True, parents=True)
    # Create output directory for stereo pairs
    if args.stereo:
        flow_out_dir_stereo = data_dir / 'RAFT' / 'stereo_flows'
        flow_out_dir_stereo.mkdir(exist_ok=True, parents=True)
    # Read all the images
    img_files = sorted(data_dir.joinpath('color').glob('*'))
    num_imgs = len(img_files)
    # Read stereo pairs
    if args.stereo:
        stereo_img_files = sorted(data_dir.joinpath('color_r').glob('*'))
        num_stereo_imgs = len(stereo_img_files)
        assert num_imgs == num_stereo_imgs, 'Number of stereo pairs does not match'
    # Compute all pairwise optical flows
    max_op_pairs = args.max_pair
    total_pairs = (2*num_imgs - 1)*max_op_pairs - max_op_pairs**2
    if args.stereo:
        total_pairs += num_imgs
    pbar = tqdm(total=total_pairs)
    with torch.no_grad():
        for i in range(num_imgs - 1): # flow pairs for forward direction
            flow_low_prev = None
            for j in range(i + 1, min(num_imgs, i +1+ max_op_pairs*args.gap), 1):
                # print('forward: ',i, j)
                imfile1 = img_files[i]
                imfile2 = img_files[j]
                image1 = load_image(imfile1, scale=args.scale)
                image2 = load_image(imfile2, scale=args.scale)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True, flow_init=flow_low_prev)
                flow_up = padder.unpad(flow_up)
                flow_low_prev = flow_low

                if (j-i-1) % args.gap == 0: # Save
                    save_file = flow_out_dir / f'{imfile1.stem}_{imfile2.stem}.npy'
                    if save_file.exists():
                        pbar.update(1)
                        continue
                    flow_up_np = flow_up.squeeze().permute(1, 2, 0).cpu().numpy()
                    np.save(save_file, flow_up_np)
                    pbar.update(1)

        for i in range(num_imgs - 1, 0, -1): # flow pairs for backward direction
            flow_low_prev = None
            for j in range(i - 1, max(-1, i-1-max_op_pairs*args.gap), -1):
                # print('backward: ',i, j)
                imfile1 = img_files[i]
                imfile2 = img_files[j]
                
                image1 = load_image(imfile1, scale=args.scale)
                image2 = load_image(imfile2, scale=args.scale)

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True, flow_init=flow_low_prev)
                flow_up = padder.unpad(flow_up)
                flow_low_prev = flow_low

                if abs(j+1-i) % args.gap == 0: # Save
                    save_file = flow_out_dir / f'{imfile1.stem}_{imfile2.stem}.npy'
                    if save_file.exists():
                        pbar.update(1)
                        continue
                    flow_up_np = flow_up.squeeze().permute(1, 2, 0).cpu().numpy()
                    np.save(save_file, flow_up_np)
                    pbar.update(1)

        if args.stereo:
            for i in range(num_imgs):
                imfile1 = img_files[i]
                imfile2 = stereo_img_files[i]
                # Check output file exists
                save_file = flow_out_dir_stereo / f'{imfile1.stem}L_{imfile2.stem}R.npy'
                if save_file.exists():
                    pbar.update(1)
                    continue

                image1 = load_image(imfile1, scale=args.scale)
                image2 = load_image(imfile2, scale=args.scale)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                # Compute optical flow from left to right
                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                flow_up = padder.unpad(flow_up)
                # Save the optical flow
                flow_up_np = flow_up.squeeze().permute(1, 2, 0).cpu().numpy()
                np.save(save_file, flow_up_np)

                pbar.update(1)
        pbar.close()
        print('computing all pairwise optical flows for {} is done \n'.format(data_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--max_pair', type=int, default=10, help='maximum number of pairs to process (optional)')
    parser.add_argument('--data_dir', type=str, default='', help='dataset dir')
    parser.add_argument('--stereo', action='store_true', help='if stereo pairs available (optional)')
    parser.add_argument('--scale', type=float, default=1.0, help='resize image before processing')
    parser.add_argument('--gap', type=int, default=3, help='frame gap for chaining (optional)')
    args = parser.parse_args()

    run_exhaustive_flow(args)


