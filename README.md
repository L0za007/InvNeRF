# Surg - InvNeRF: Invertible NeRF for 3D tracking and reconstruction in surgical vision

*Full code will be released upon acceptance of the manuscript.*

Invertible Neural Radiance Field (InvNeRF) implementation.

Qualitative results for 3D point tracking.

<image src='media/InvNeRF.gif' width=480 /> |

Qualitative results for our pixel sampling algorithm.

<image src='media/Pixel_Sampling.gif' width=480 /> |
# Installation
We recommend the use of a conda environment. You can use the command below to create an environment with Python 3.10. Then follow the instructions from the [PyTorch documentation](https://pytorch.org/#:~:text=and%20easy%20scaling.-,Install%20PyTorch,-Select%20your%20preferences) to install torch. We ran our code with torch 2.5.1 and CUDA 12.4.
```
conda env create -f enviroment.yml 
```
After installing Torch in your environment, please install the missing requirements using the following command.
```
pip install -r requirements.txt
```
-------
Additionally, if you want to run experiments using the dynamic proposal network in the models folder, you need to install some extra requirements. Please follow the instructions from [NVlabs](https://github.com/NVlabs/tiny-cuda-nn#requirements) to install tiny-cuda-nn and [NeRFAcc](https://github.com/nerfstudio-project/nerfacc#installation) installation process to install NeRFAcc.

# Data
We used a small subset of short videos from the original [STIR dataset](https://ieee-dataport.org/open-access/stir-surgical-tattoos-infrared) to test and make an initial comparison of our methods and other TTO approaches. The list of videos used for this can be extracted from the STIR_Origin labels in `data/labels` (see [`data/labels/STIR_Origin_Tool_mask_prompts.json`](https://github.com/L0za007/InvNeRF/tree/main/data/labels)). We also used the recently released evaluation [STIR Challenge 2024 dataset](https://zenodo.org/records/14803158) to test our approach and make a comparion with feed-forward methods. Finally, the [`SCARED dataset`](https://arxiv.org/pdf/2101.01133) was also used to demonstrate the easy incorporation of kinematic data to inform the camera pose to our model.

Note that only one type of correspondence (RAFT, RAFT+CoTracker or MFT) is needed to train InvNeRF. You can find more details about the preprocessing of the data in [`data/preprocessing`](https://github.com/L0za007/InvNeRF/tree/main/data/preprocessing/README_preprocessing.md).


```
-> video_folder
----> color             # Left RGB frames
----> color_r           # Right RGB frames
----> calib.json        # Camera parameters
----> cam_RT.json       # (Optional) Camera poses
----> tool_masks        # (Optional) Binary masks of the tools
----> dino              # (Optional) Dino features
----> RAFT              # (Optional) RAFT output
-------> count_maps       # Number of corresponcences in the video per pixel
-------> CT_flows         # RAFT correspondences refined by CoTracker
-------> CT_masks         # Occlusion masks CoTracker
-------> flows            # RAFT correspondences
-------> masks            # Cycle consistency/occlusion masks
-------> stereo_flows     # Stereo correspondences
-------> flow_stats.json  # Number of corresponcences in the video per frame
----> MFT               # (Optional) MFT output
-------> count_maps       # Number of corresponcences in the video per pixel
-------> flows            # MFT correspondences
-------> masks            # Confidence MFT masks
-------> stereo_flows     # Stereo correspondences
-------> flow_stats.json  # Number of corresponcences in the video per frame
```

# Optimisation
This section will be released after the work is accepted/published
# Validation
This section will be released soon
# Visualisation
This section will be released soon

# Citation
```bibtex
@article{InvNeRF2024,
   author = {Gerardo Loza and Junlei Hu and Dominic Jones and Pietro Valdastri and Sharib Ali},
   title = {Real-time surgical tool detection with multi-scale positional encoding and contrastive learning},
   keywords = {long-term point tracking, test-time optimisation, invertible NeRF (InvNeRF), consistent motion, geometric consistency},
   year = {2025},
}
```
