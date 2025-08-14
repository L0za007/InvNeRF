## Script for finding optical flow between pairs of images in a folder
import os
import subprocess
import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,   required=True, help='Dataset path')
    parser.add_argument('--cycle_th', type=float, default=3.,    help='Threshold for cycle consistency error used with RAFT flow')
    parser.add_argument('--max_pair', type=int,   default=8,     help='Maximum number of pairs per frame to process. default 8')
    parser.add_argument('--gap',      type=int,   default=1,     help='Frame gap, default 1')
    parser.add_argument('--scale',    type=float, default=1.0,   help='Scale to resize images, default 1.0')
    parser.add_argument('--stereo',    action='store_true', help='if Truth, calculate stereo flow')
    parser.add_argument('--CoTracker', action='store_true', help='if Truth, use CoTracker to refine RAFT optical flow')
    parser.add_argument('--MFT',       action='store_true', help='if Truth, use MFT for optical flow')
    args = parser.parse_args()

    # Cancel if cuda is not available
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    
    
    if args.MFT:
        os.chdir('MFT')
        raft_ex = ['python', 'exhaustive_MFT.py', '--data_dir', args.data_dir,
                '--max_pair', str(args.max_pair), '--scale', str(args.scale), '--gap', str(args.gap)]
        if args.stereo:
            raft_ex.append('--stereo')
        subprocess.run(raft_ex)
        os.chdir('../') 
    else: # compute raft optical flows between all pairs. this creates the folder raft_exhaustive
        os.chdir('RAFT')
        raft_ex = ['python', 'exhaustive_raft.py', '--data_dir', args.data_dir,
                '--max_pair', str(args.max_pair), '--scale', str(args.scale), '--gap', str(args.gap)]
        if args.stereo:
            raft_ex.append('--stereo')
        subprocess.run(raft_ex)
        os.chdir('../') 

        # compute dino feature maps it crates the folder features
        os.chdir('dino')
        dino_ex = ['python', 'extract_dino_features.py', '--data_dir', args.data_dir]
        subprocess.run(dino_ex)
        os.chdir('../') 

        # filtering
        os.chdir('RAFT')
        subprocess.run(['python', 'filter_raft.py', '--data_dir', args.data_dir, '--cycle_th', str(args.cycle_th),
                        '--max_pair', str(args.max_pair), '--scale', str(args.scale), '--gap', str(args.gap)])
        os.chdir('../')
        
        # Robust tracker for fast optical flow (Cotracker)
        if args.CoTracker:
            cotracker_ex = ['python', 'dense_cotracker.py', '--data_dir', args.data_dir, 
                            '--max_pair', str(args.max_pair),'--gap', str(args.gap)]
            subprocess.run(cotracker_ex)


