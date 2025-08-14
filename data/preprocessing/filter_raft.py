"""
This script filters the raft optical flow using cycle consistency and appearance consistency
checks (using dino features), and produces the following files:

raft masks: h x w x 3 for each pair of flows, first channel stores the mask for cycle consistency,
            second channel stores the mask for occlusion (i.e., regions that detected as occluded
            where the prediction is likely to be reliable using double cycle consistency checks).
count_maps: h x w for each frame (uint16), contains the number of valid correspondences for each pixel
            across all frames.
flow_stats.json: contains the total number of valid correspondences between each pair of frames.
"""

import json, argparse, imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

DEVICE = 'cuda'

def gen_grid(h, w, device, normalize=False, homogeneous=False):
    if normalize:
        lin_y = torch.linspace(-1., 1., steps=h, device=device)
        lin_x = torch.linspace(-1., 1., steps=w, device=device)
    else:
        lin_y = torch.arange(0, h, device=device)
        lin_x = torch.arange(0, w, device=device)
    grid_y, grid_x = torch.meshgrid((lin_y, lin_x))
    grid = torch.stack((grid_x, grid_y), -1)
    if homogeneous:
        grid = torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1)
    return grid  # [h, w, 2 or 3]


def normalize_coords(coords, h, w, no_shift=False):
    assert coords.shape[-1] == 2
    if no_shift:
        return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2
    else:
        return coords / torch.tensor([w-1., h-1.], device=coords.device) * 2 - 1.

@torch.no_grad()
def run_filtering(args):
    feature_name = 'dino'
    scene_dir = Path(args.data_dir)
    max_op_pairs = args.max_pair
    print('flitering raft optical flow for {}....'.format(scene_dir))

    # Loading images
    img_files = sorted(scene_dir.joinpath('color').glob('*'))
    num_imgs = len(img_files)
    #pbar = tqdm(total=num_imgs * (num_imgs - 1))
    pbar = tqdm(total= (2*num_imgs - 1)*max_op_pairs - max_op_pairs**2)

    # Creating output file and directory for stats and masks
    out_flow_stats_file = scene_dir / 'RAFT' / 'flow_stats.json'
    out_dir = scene_dir / 'RAFT' / 'masks'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Creating output directory for count maps
    count_out_dir = scene_dir / 'RAFT' / 'count_maps'
    count_out_dir.mkdir(parents=True, exist_ok=True)

    h, w = imageio.imread(img_files[0]).shape[:2]
    if args.scale < 1.0:
        h = int(h * args.scale)
        w = int(w * args.scale)
    grid = gen_grid(h, w, device=DEVICE).permute(2, 0, 1)[None]
    grid_normed = normalize_coords(grid.squeeze().permute(1, 2, 0), h, w)  # [h, w, 2]

    features = []
    for img_file in img_files:
        file = scene_dir / feature_name / (img_file.stem + '.npy')
        features.append(torch.from_numpy(np.load(file)).float().to(DEVICE))
        #print('loading features for {}'.format(file))

    flow_stats = {}
    count_maps = np.zeros((num_imgs, h, w), dtype=np.uint16)
    for i in range(num_imgs):
        imgname_i = img_files[i].stem
        feature_i = features[i].permute(2, 0, 1)[None]
        feature_i_sampled = F.grid_sample(feature_i, grid_normed[None], align_corners=True)[0].permute(1, 2, 0)

        j_idxs = list(range(i - 1, max(-1, i-1-max_op_pairs*args.gap), -1*args.gap))[::-1]+ \
                 list(range(i + 1, min(num_imgs, i +1+ max_op_pairs*args.gap), args.gap))
        for j in j_idxs:
            if i == j:
                continue
            #print('Pair: ', i, j)
            frame_interval = abs(i - j)
            imgname_j = img_files[j].stem
            out_file = '{}/{}_{}.png'.format(out_dir, imgname_i, imgname_j)
            if Path(out_file).exists():
                out_mask = imageio.imread(out_file) / 255.0
                if not imgname_i in flow_stats.keys():
                    flow_stats[imgname_i] = {}
                flow_stats[imgname_i][imgname_j] = np.sum(out_mask).item()
                count_maps[i] += out_mask.sum(axis=-1).astype(np.uint16)
                pbar.update(1)
                continue

            flow_f = np.load(scene_dir / 'RAFT' / 'flows' / f'{imgname_i}_{imgname_j}.npy')
            flow_f = torch.from_numpy(flow_f).float().permute(2, 0, 1)[None].cuda()
            flow_b = np.load(scene_dir / 'RAFT' / 'flows' / f'{imgname_j}_{imgname_i}.npy')
            flow_b = torch.from_numpy(flow_b).float().permute(2, 0, 1)[None].cuda()

            coord2 = flow_f + grid
            coord2_normed = normalize_coords(coord2.squeeze().permute(1, 2, 0), h, w)  # [h, w, 2]
            flow_21_sampled = F.grid_sample(flow_b, coord2_normed[None], align_corners=True)
            map_i = flow_f + flow_21_sampled
            fb_discrepancy = torch.norm(map_i.squeeze(), dim=0)
            mask_cycle = fb_discrepancy < args.cycle_th # This check that the flows a->b and b->a are consistent

            feature_j = features[j].permute(2, 0, 1)[None]
            feature_j_sampled = F.grid_sample(feature_j, coord2_normed[None], align_corners=True)[0].permute(1, 2, 0)
            feature_sim = torch.cosine_similarity(feature_i_sampled, feature_j_sampled, dim=-1)
            feature_mask = feature_sim > 0.5 # This check that the features are consistent

            mask_cycle = mask_cycle * feature_mask if frame_interval >= 3 else mask_cycle

            # only keep correspondences for occluded pixels if the correspondences are
            # inconsistent in the first cycle but consistent in the second cycle
            # and if the two frames are adjacent enough (interval < 3)
            # Loza: this one rescue the occluded pixels that are not consistent in the first cycle but consistent in the second cycle
            if frame_interval < 3:
                coord_21 = grid + map_i  # [1, 2, h, w]
                coord_21_normed = normalize_coords(coord_21.squeeze().permute(1, 2, 0), h, w)  # [h, w, 2]
                flow_22 = F.grid_sample(flow_f, coord_21_normed[None], align_corners=True)
                fbf_discrepancy = torch.norm((coord_21 + flow_22 - flow_f - grid).squeeze(), dim=0)
                mask_in_range = (coord2_normed.min(dim=-1)[0] >= -1) * (coord2_normed.max(dim=-1)[0] <= 1)
                mask_occluded = (fbf_discrepancy < args.cycle_th) * (fb_discrepancy > args.cycle_th * 1.5)
                mask_occluded *= mask_in_range
            else:
                mask_occluded = torch.zeros_like(mask_cycle)

            out_mask = torch.stack([mask_cycle, mask_occluded, torch.zeros_like(mask_cycle)], dim=-1).cpu().numpy()
            imageio.imwrite(out_file, (255 * out_mask.astype(np.uint8)))

            if not imgname_i in flow_stats.keys():
                flow_stats[imgname_i] = {}
            flow_stats[imgname_i][imgname_j] = np.sum(out_mask).item()
            count_maps[i] += out_mask.sum(axis=-1).astype(np.uint16)
            pbar.update(1)

    pbar.close()
    with open(out_flow_stats_file, 'w') as fp:
        json.dump(flow_stats, fp)

    for i in range(num_imgs):
        img_name = img_files[i].stem + '.png'
        imageio.imwrite(count_out_dir / img_name, count_maps[i])

    print('filtering raft optical flow for {} is done\n'.format(scene_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='', help='dataset dir')
    parser.add_argument('--cycle_th', type=float, default=3., help='threshold for cycle consistency error')
    parser.add_argument('--max_pair', type=int, default=30, help='maximum number of pairs to process (optional)')
    parser.add_argument('--scale', type=float, default=1.0, help='resize image before processing')
    parser.add_argument('--gap', type=int, default=1, help='frame gap for chaining (optional)')
    args = parser.parse_args()

    run_filtering(args)
