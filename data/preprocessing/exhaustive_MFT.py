import warnings
warnings.filterwarnings("ignore")
import argparse, torch
import numpy as np, imageio, json
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from MFT.config import load_config
from MFT.point_tracking import convert_to_point_tracking
import MFT.utils.vis_utils as vu
import MFT.utils.io as io_utils
from MFT.utils.misc import ensure_numpy

DEVICE = 'cuda'
CONFIG = "configs/MFT_cfg.py"

def load_image(imfile, scale=1.0):
    img = Image.open(imfile)
    if scale<1.0:
        img = img.resize((int(img.width*scale), int(img.height*scale)), Image.LANCZOS)
    img = np.array(img).astype(np.uint8)
    # img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img#[None].to(DEVICE)

def get_queries(frame_shape, spacing):
    H, W = frame_shape
    xs = np.arange(0, W, spacing)
    ys = np.arange(0, H, spacing)

    xs, ys = np.meshgrid(xs, ys)
    flat_xs = xs.flatten()
    flat_ys = ys.flatten()

    queries = np.vstack((flat_xs, flat_ys)).T
    return torch.from_numpy(queries).float().cuda()

def run_exhaustive_flow(args):
    # Load the MFT model
    config = load_config(CONFIG)
    tracker = config.tracker_class(config)
    # Read the data directory
    data_dir = Path(args.data_dir)
    print('computing all pairwise optical flows for {}...'.format(data_dir))
    # Create output directory
    flow_out_dir = data_dir / 'MFT' / 'flows'
    flow_out_dir.mkdir(exist_ok=True, parents=True)
    mask_out_dir = data_dir / 'MFT' / 'masks'
    mask_out_dir.mkdir(exist_ok=True, parents=True)
    count_out_dir = data_dir / 'MFT' / 'count_maps'
    count_out_dir.mkdir(exist_ok=True, parents=True)
    out_flow_stats_file = data_dir / 'MFT' / 'flow_stats.json'
    # Create output directory for stereo pairs
    if args.stereo:
        flow_out_dir_stereo = data_dir / 'MFT' / 'stereo_flows'
        flow_out_dir_stereo.mkdir(exist_ok=True, parents=True)
    # Read all the images
    img_files = sorted(data_dir.joinpath('color').glob('*'))
    num_imgs = len(img_files)
    h, w = imageio.imread(img_files[0]).shape[:2]
    # Read stereo pairs
    if args.stereo:
        stereo_img_files = sorted(data_dir.joinpath('color_r').glob('*'))
        num_stereo_imgs = len(stereo_img_files)
        assert num_imgs == num_stereo_imgs, 'Number of stereo pairs does not match'
    # Compute all pairwise optical flows
    max_op_pairs = args.max_pair
    total_pairs = 0
    for i in range(num_imgs - 1): # flow pairs for forward direction
        for j in range(i + 1, min(num_imgs, i +1+ max_op_pairs*args.gap), 1):
            if (j-i-1) % args.gap == 0:
                total_pairs += 2
    if args.stereo:
        total_pairs += num_imgs
    flow_stats = {}
    count_maps = np.zeros((num_imgs, h, w), dtype=np.uint16)
    OCC_THRESH = 0.5
    pbar = tqdm(total=total_pairs)
    with torch.no_grad():
        for i in range(num_imgs - 1): # flow pairs for forward direction
            initialized = False
            for j in range(i + 1, min(num_imgs, i +1+ max_op_pairs*args.gap), 1):
                # print('mft-forward: ',i, j)
                imfile1 = img_files[i]
                imfile2 = img_files[j]
                
                image1 = load_image(imfile1, scale=args.scale)
                image2 = load_image(imfile2, scale=args.scale)

                if not initialized:
                    meta = tracker.init(image1)
                    initialized = True
                    queries = get_queries(image1.shape[:2], 1)
                    meta = tracker.track(image2)
                else:
                    meta = tracker.track(image2)

                coords, occlusions = convert_to_point_tracking(meta.result, queries) # [n_pts, 2], [n_pts]

                if (j-i-1) % args.gap == 0: # Save
                    save_file = flow_out_dir / f'{imfile1.stem}_{imfile2.stem}.npy'
                    flow_up_np = coords-queries.cpu().numpy()
                    flow_up_np = flow_up_np.reshape(*image1.shape[:2],2)
                    np.save(save_file, flow_up_np)

                    # occlusions = occlusions.cpu().numpy()
                    mask_file = mask_out_dir / f'{imfile1.stem}_{imfile2.stem}.png'
                    mask = np.zeros((*image1.shape[:2],3), dtype=np.uint8)
                    occlusions = occlusions.reshape(*image1.shape[:2])
                    mask[..., 0] = (occlusions <= OCC_THRESH) * 255
                    if abs(i-j) == 1:
                        mask[..., 1] = ((occlusions >  OCC_THRESH) & (occlusions <  1)) * 255
                    imageio.imwrite(mask_file, mask)

                    if not imfile1.stem in flow_stats.keys():
                        flow_stats[imfile1.stem] = {}
                    flow_stats[imfile1.stem][imfile2.stem] = np.sum(mask).item()
                    count_maps[i] += mask.sum(axis=-1).astype(np.uint16)

                    pbar.update(1)

        for i in range(num_imgs - 1, 0, -1): # flow pairs for backward direction
            initialized = False
            for j in range(i - 1, max(-1, i-1-max_op_pairs*args.gap), -1*args.gap):
                # print('mft-backward: ',i, j)
                imfile1 = img_files[i]
                imfile2 = img_files[j]
                
                image1 = load_image(imfile1, scale=args.scale)
                image2 = load_image(imfile2, scale=args.scale)

                if not initialized:
                    meta = tracker.init(image1)
                    initialized = True
                    queries = get_queries(image1.shape[:2], 1)
                    meta = tracker.track(image2)
                else:
                    meta = tracker.track(image2)

                coords, occlusions = convert_to_point_tracking(meta.result, queries) # [n_pts, 2], [n_pts]

                if abs(j+1-i) % args.gap == 0: # Save
                    save_file = flow_out_dir / f'{imfile1.stem}_{imfile2.stem}.npy'
                    flow_up_np = coords-queries.cpu().numpy()
                    flow_up_np = flow_up_np.reshape(*image1.shape[:2],2)
                    np.save(save_file, flow_up_np)

                    mask_file = mask_out_dir / f'{imfile1.stem}_{imfile2.stem}.png'
                    mask = np.zeros((*image1.shape[:2],3), dtype=np.uint8)
                    occlusions = occlusions.reshape(*image1.shape[:2])
                    mask[..., 0] = (occlusions  <= OCC_THRESH) * 255
                    if abs(i-j) == 1:
                        mask[..., 1] = ((occlusions >  OCC_THRESH) & (occlusions <=  1)) * 255
                    imageio.imwrite(mask_file, mask)

                    if not imfile1.stem in flow_stats.keys():
                        flow_stats[imfile1.stem] = {}
                    flow_stats[imfile1.stem][imfile2.stem] = np.sum(mask).item()
                    count_maps[i] += mask.sum(axis=-1).astype(np.uint16)
                    pbar.update(1)

        if args.stereo:
            for i in range(num_imgs):
                imfile1 = img_files[i]
                imfile2 = stereo_img_files[i]
                # Check output file exists
                save_file = flow_out_dir_stereo / f'{imfile1.stem}L_{imfile2.stem}R.npy'

                image1 = load_image(imfile1, scale=args.scale)
                image2 = load_image(imfile2, scale=args.scale)

                # Compute optical flow from left to right
                meta = tracker.init(image1)
                queries = get_queries(image1.shape[:2], 1)
                meta = tracker.track(image2)

                coords, occlusions = convert_to_point_tracking(meta.result, queries) # [n_pts, 2], [n_pts]
                flow_up = coords-queries.cpu().numpy()
                

                # Save the optical flow
                flow_up_np = flow_up.reshape(*image1.shape[:2],2)
                np.save(save_file, flow_up_np)

                pbar.update(1)
        pbar.close()
        with open(out_flow_stats_file, 'w') as fp:
            json.dump(flow_stats, fp)

        for i in range(num_imgs):
            img_name = img_files[i].stem + '.png'
            imageio.imwrite(count_out_dir / img_name, count_maps[i])
        print('computing all pairwise optical flows for {} is done \n'.format(data_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_pair', type=int, default=10, help='maximum number of pairs to process (optional)')
    parser.add_argument('--data_dir', type=str, default='', help='dataset dir')
    parser.add_argument('--stereo', action='store_true', help='if stereo pairs available (optional)')
    parser.add_argument('--scale', type=float, default=1.0, help='resize image before processing')
    parser.add_argument('--gap', type=int, default=3, help='frame gap for chaining (optional)')
    args = parser.parse_args()

    run_exhaustive_flow(args)


