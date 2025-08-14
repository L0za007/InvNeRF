import argparse, torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import imageio.v2 as imageio
from tqdm import tqdm
from pathlib import Path
import subprocess

DEVICE = 'cuda'
def load_image(imfile, scale=1.0):
    img = Image.open(imfile)
    if scale<1.0:
        img = img.resize((int(img.width*scale), int(img.height*scale)), Image.LANCZOS)
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float() # HWC to CHW
    return img[None].to(DEVICE)

def get_global_mask(dir, mask_id, scale=1.0):
    masks = []
    for mask_file in dir.glob(f'{mask_id}_*.png'):
        mask = Image.open(mask_file)
        if scale < 1.0:
            mask = mask.resize((int(mask.width*scale), int(mask.height*scale)), Image.LANCZOS)
        mask = np.array(mask)
        mask = mask > 0
        mask = torch.from_numpy(~mask)  # get the inverse mask (places where raft failed)
        _mask = mask[...,0]
        for i in range(1, mask.shape[-1]):
            _mask = _mask & mask[...,i]
        # print(f'loading mask {mask_file.stem} number of missing values {_mask.sum().item()}')
        masks.append(_mask)
    final_mask = masks[0]
    for _mask in masks[1:]:
        final_mask = final_mask | _mask
    # print(f'Final mask {mask_id} number of missing values {final_mask.sum().item()}')
    return final_mask.to(DEVICE)  # Add batch dimension

def load_mask(mask_file):
    mask = np.array(Image.open(mask_file)) > 0
    return torch.from_numpy(mask).to(DEVICE)  # Convert to tensor and move to device

def scale_flow(flow, scale=1.0):
    if scale < 1.0:
        flow = torch.from_numpy(flow).float() if not isinstance(flow, torch.Tensor) else flow
        h, w = flow.shape[:2]
        flow = flow.permute(2, 0, 1).unsqueeze(0).float()
        flow = F.interpolate(flow, scale_factor=scale, mode='bilinear', align_corners=True)
        flow = flow.squeeze().permute(1, 2, 0)  # [h, w, 2]
        flow = flow * scale  # Scale the flow
        flow = flow.numpy()  # Convert back to numpy
    return flow

def run_cotracker(args):
    # Load the raft model
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(DEVICE)
    model.eval()
    print("model interp_shape: ", model.interp_shape)
    # model input format, videos: [batch, time, channel, height, width] with batch=1, queries: [batch, num_quaries, 3]
    # Read the data directory
    data_dir = Path(args.data_dir)
    print('[CoTracker]:Refining all pairwise optical flows for {}...'.format(data_dir))
    # Create output directory
    flow_out_dir = data_dir / 'RAFT' / 'CT_flows'
    flow_out_dir.mkdir(exist_ok=True, parents=True)
    mask_out_dir = data_dir / 'RAFT' / 'CT_masks'
    mask_out_dir.mkdir(exist_ok=True, parents=True)
    # Create raft masks directory
    raft_flow_dir = data_dir / 'RAFT' / 'flows'
    raft_mask_dir = data_dir / 'RAFT' / 'masks'
    # Read all the images
    img_files = sorted(data_dir.joinpath('color').glob('*'))
    num_imgs = len(img_files)
    # Query grid
    img = load_image(img_files[0], scale=args.scale)
    height, width = img.shape[2], img.shape[3]
    print('hight: {}, width: {}'.format(height, width))
    # Compute all pairwise optical flows
    total_pairs = 0
    for i in range(num_imgs - 1):
        for j in range(i + 1, min(num_imgs, i +1+ args.max_pair*args.gap), args.gap):
            total_pairs += 1
    for i in range(num_imgs - 1, 0, -1):
        for j in range(i - 1, max(-1, i-1-args.max_pair*args.gap), -1*args.gap):
            total_pairs += 1

    pbar = tqdm(total=total_pairs)
    with torch.no_grad():
        c = 0
        for i in range(num_imgs - 1): # flow pairs for forward direction
            imfile_i = img_files[i]
            images = [load_image(imfile_i, scale=args.scale)]
            output_files = []
            raft_flows = []
            for j in range(i + 1, min(num_imgs, i +1+ args.max_pair*args.gap), 1):
                imfile_j = img_files[j]
                image_j = load_image(imfile_j, scale=args.scale)
                images.append(image_j)
                if (j-i-1) % args.gap == 0: # Save
                    out_file = flow_out_dir / f'{imfile_i.stem}_{imfile_j.stem}.npy'
                    raft_file = raft_flow_dir / f'{imfile_i.stem}_{imfile_j.stem}.npy'
                    raft_flows.append(np.load(raft_file))
                    output_files.append(out_file)
                
            images = torch.cat(images, dim=0) # [n, 3, h, w]
            queries = gen_grid(h=height, w=width, time=True, device=DEVICE) # [h, w, 3]
            mask = get_global_mask(raft_mask_dir, imfile_i.stem, scale=args.scale ) # [h, w]
            queries = queries[mask].unsqueeze(0) # [1, np, 3]
            flow_up, confidence = [],[]
            for _pts in torch.split(queries, split_size_or_sections=100*100, dim=1): # [1, np, 3]
                _flow_up, _confidence = model(images[None], _pts) # [1, n+1, np, 2], [1, n+1, np] # where 2 is (x,y)
                _flow_up = _flow_up
                _flow_up = _flow_up[0] - _pts[..., 1:] # convert to flow # [n+1, np, 2]
                
                flow_up.append(_flow_up) #
                confidence.append(_confidence[0])
            flow_up = torch.cat(flow_up, dim=1)          # [n+1, np, 2]
            confidence = torch.cat(confidence, dim=1)    # [n+1, np]
            # print(f'flow_up shape: {flow_up.shape}, confidence shape: {confidence.shape}')

            mask_np = mask.cpu().numpy() # [h, w]
            for i, out_file in enumerate(output_files):
                # Update the flow with the previous RAFT flow
                flow = scale_flow(raft_flows[i], scale=args.scale) # [h, w, 2]
                flow[mask_np] = flow_up[i*args.gap+1].cpu().numpy()
                flow = scale_flow(flow, scale=1/args.scale) # [h, w, 2]
                np.save(out_file, flow)
                # Save updated mask with binary confidence values
                out_file = out_file.name.replace('.npy', '.png')
                conf = confidence[i*args.gap+1] > 0.5
                out_mask = load_mask(raft_mask_dir / out_file)  # Load the mask
                out_mask[...,0][mask] = conf # Update mask with confidence
                out_mask = out_mask.cpu().numpy()  # Convert to numpy to save
                imageio.imwrite(mask_out_dir/out_file, (255 * out_mask.astype(np.uint8)))
            pbar.update(len(output_files))

        for i in range(num_imgs - 1, 0, -1): # flow pairs for backward direction
            imfile_i = img_files[i]
            images = [load_image(imfile_i, scale=args.scale)]
            output_files = []
            raft_flows = []
            for j in range(i - 1, max(-1, i-1-args.max_pair*args.gap), -1):
                imfile_j = img_files[j]
                image_j = load_image(imfile_j, scale=args.scale)
                images.append(image_j)
                if abs(j+1-i) % args.gap == 0: # Save
                    out_file = flow_out_dir / f'{imfile_i.stem}_{imfile_j.stem}.npy'
                    raft_file = raft_flow_dir / f'{imfile_i.stem}_{imfile_j.stem}.npy'
                    raft_flows.append(np.load(raft_file))
                    output_files.append(out_file)
                
            images = torch.cat(images, dim=0) # [n, 3, h, w]
            queries = gen_grid(h=height, w=width, time=True, device=DEVICE)
            mask = get_global_mask(raft_mask_dir, imfile_i.stem, scale=args.scale) 
            queries = queries[mask].unsqueeze(0)
            flow_up, confidence = [],[]
            for _pts in torch.split(queries, split_size_or_sections=100*100, dim=1):
                _flow_up, _confidence = model(images[None], _pts) 
                _flow_up = _flow_up[0] - _pts[..., 1:] # convert to flow
                flow_up.append(_flow_up)
                confidence.append(_confidence[0])
            flow_up = torch.cat(flow_up, dim=1)
            confidence = torch.cat(confidence, dim=1)
            mask_np = mask.cpu().numpy() # [h, w]
            for i, out_file in enumerate(output_files):
                # Save the optical flow
                flow = scale_flow(raft_flows[i], scale=args.scale) # [h, w, 2]
                flow[mask_np] = flow_up[i*args.gap+1].cpu().numpy()
                flow = scale_flow(flow, scale=1/args.scale) # [h, w, 2]
                np.save(out_file, flow)
                # Save updated mask with binary confidence values
                out_file = out_file.name.replace('.npy', '.png')
                conf = confidence[i*args.gap+1] > 0.5
                out_mask = load_mask(raft_mask_dir / out_file)  # Load the mask
                out_mask[...,0][mask] = conf # Update mask with confidence
                out_mask = out_mask.cpu().numpy()  # Convert to numpy to save
                imageio.imwrite(mask_out_dir/out_file, (255 * out_mask.astype(np.uint8)))

            pbar.update(len(output_files))

    print('[CoTracker]: Refining all pairwise optical flows for {} is done \n'.format(data_dir))
    subprocess.run(['rm', '-r', str(raft_flow_dir)])
    subprocess.run(['rm', '-r', str(raft_mask_dir)])

def gen_grid(h=None, w=None, time=False, device='cuda'):
        """ Generate a grid with the position of each pixel
        Args:
            h (int): Height of the grid
            w (int): Width of the grid
            time (bool): Add query time
        Returns:
            grid (torch.Tensor): Grid with the position of each pixel [h,w,2 or 3]"""
        lin_y = torch.arange(0, h, device=device)
        lin_x = torch.arange(0, w, device=device)
        grid_y, grid_x = torch.meshgrid((lin_y, lin_x), indexing='ij') # Addressing torch warning regarding indexing
        grid = torch.stack((grid_x, grid_y), dim=-1)
        if time:
            grid = torch.cat([torch.zeros_like(grid[..., :1]), grid], dim=-1)
        return grid.float()  # [h, w, 2 or 3]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='', help='dataset dir')
    parser.add_argument('--max_pair', type=int, default=10, help='maximum number of pairs to process (optional)')
    parser.add_argument('--scale', type=float, default=1.0, help='resize image before processing')
    parser.add_argument('--gap', type=int, default=3, help='frame gap for chaining (optional)')
    args = parser.parse_args()

    run_cotracker(args)


