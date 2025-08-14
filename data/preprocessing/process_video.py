from pathlib import Path
from PIL import Image
import cv2, argparse, json, os
import torch
import numpy as np
import imageio.v2 as imageio


def argparser():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory of the video")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale factor for the frames")
    parser.add_argument('--gen_mask', action='store_true', help='Use SAM2 to segment part of the image based on the point prompts')
    args = parser.parse_args()
    return args

def extract_frames_from_video(data_dir:Path, camera='left', scale=1.0):
    """ Extract frames from video"""
    output_folder = Path(data_dir) 
    output_folder = output_folder / 'color' if camera == 'left' else output_folder / 'color_r'
    vid_dir = list(data_dir.glob(f'*{camera}.mp4'))
    assert len(vid_dir) == 1, f"Expected 1 single video for camera {camera}, found {len(vid_dir)}"
    vid_dir = vid_dir[0]
    print(f"Extracting frames from {str(vid_dir)} to {str(output_folder)}")
    output_folder.mkdir(parents=True, exist_ok=True)

    vid = cv2.VideoCapture(str(vid_dir))
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    print(f"Video:{str(output_folder).split('/')[-1]}, Frame count: {frame_count}, FPS: {fps}")
    for i in range(frame_count):
        ret, frame = vid.read()
        if ret: # save frame
            file_name = output_folder / f"{i:04d}.png"
            if file_name.exists(): continue
            np_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np_frame)
            if scale < 1.0: # resize frame
                new_width = int(scale * frame.width)
                new_height = int(scale * frame.height)
                new_width = (new_width // 16) * 16
                new_height = (new_height // 16) * 16
                frame = frame.resize((new_width, new_height), Image.LANCZOS)
            frame.save(file_name)
    vid.release()
    # Return original size of the frames
    file_name = output_folder / f"{0:04d}.png"
    np_frame = cv2.imread(str(file_name))
    return (np_frame.shape[1], np_frame.shape[0])

def verify_camera_params(data_dir:Path, img_size):
    """ Extract camera parameters from video"""
    calib_file = Path(data_dir)/'calib.json'
    # read the calibration file into a dictionary
    calib = json.load(open(str(calib_file)))
    calib["img_size"] = img_size[::-1] # (H, W)
    # Convert the transformation to go from left to right
    if calib['translation'][0]<0:
        calib['translation'] = [-item for item in calib['translation']]
        rot_mat = cv2.Rodrigues(np.array(calib['rotation']).reshape(3,1))[0]
        calib['rotation'] = cv2.Rodrigues(rot_mat.T)[0].ravel().tolist()
    # save the calibration parameters
    with open(calib_file, 'w') as f:
        json.dump(calib, f)

def generate_tools_masks(data_dir:Path, scale=1.0):
    """Generate tool masks for the video frames"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[SAM2]: using device: {device}")

    # read data
    frames_dir = data_dir/'color'
    frame_files = [f for f in (data_dir/ 'color').glob('*.png')]
    frame_files = sorted(frame_files)
    # out directory
    mask_dir = data_dir / 'tool_masks'
    mask_dir.mkdir(parents=True, exist_ok=True)
    # read prompts
    save_path = data_dir.parent / "Tool_mask_prompts.json"
    with open(save_path, "r") as f:
        labels = json.load(f)
    prompt = {}
    for idx, n_pts in enumerate(labels[data_dir.name]):
        n_pts = np.array(n_pts, dtype=np.float32)*3*scale # Scale by 3 first because the prompt were taken at this scale
        prompt[idx] = n_pts, np.ones(len(n_pts), np.int32)
    print(f"[SAM2]: Generating tool masks for the folder {data_dir.name} with the prompt{prompt}")
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(str(frames_dir))
        # Make inference in initial mask
        for key,val in prompt.items():
            frame_idx, object_ids, masks = predictor.add_new_points_or_box(inference_state=state,frame_idx=0,
                                                                           obj_id=key,points=val[0],labels=val[1])
        # Propagate mask
        seg_frames = []
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            tool_mask = [(out_mask_logits[i] > 0.0).cpu().numpy() 
                         for i, out_obj_id in enumerate(out_obj_ids)]
            tool_mask = np.concatenate(tool_mask, axis=0)
            tool_mask = np.any(tool_mask,axis=0)
            _image = imageio.imread(frame_files[out_frame_idx])
            _image[...,1] =  np.clip((_image[...,1].astype(np.int32)+tool_mask*40),a_min=0,a_max=255).astype(np.uint8)
            seg_frames.append(_image)
            # Save the mask
            mask_file = mask_dir / frame_files[out_frame_idx].name
            imageio.imwrite(mask_file, tool_mask.astype(np.uint8) * 255)

        _file = data_dir/'tool_masks.mp4'
        imageio.mimsave(_file, seg_frames, quality=8, fps=20)

if __name__ == '__main__':
    args = argparser()
    data_dir = Path(args.data_dir)
    scale = args.scale
    img_sizeL = extract_frames_from_video(data_dir, scale=scale)
    img_sizeR = extract_frames_from_video(data_dir, scale=scale, camera='right')
    assert img_sizeL == img_sizeR, "Different image sizes between left and right cameras"
    verify_camera_params(data_dir, img_sizeL)
    if args.gen_mask:
        from sam2.sam2_video_predictor import SAM2VideoPredictor
        generate_tools_masks(data_dir, scale=scale)
