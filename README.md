# Surg - InvNeRF: Invertible NeRF for 3D tracking and reconstruction in surgical vision


Invertible Neural Radiance Field (InvNeRF) implementation.

Qualitative results for 3D point tracking.

<image src='media/InvNeRF.gif' width=480 /> |

Qualitative results for our pixel sampling algorithm.

<image src='media/Pixel_Sampling.gif' width=480 /> |

# Optimisation
This section will be released after publishing
# Validation
This section will be released soon
# Visualisation
This section will be released soon
# Installation
We recomend the use of a conda environment, you can use the comand below to create an environment with python 3.10. Then follow the instrucctions from the [PyTorch documentation](https://pytorch.org/#:~:text=and%20easy%20scaling.-,Install%20PyTorch,-Select%20your%20preferences) to install torch. We ran our code with torch 2.5.1 and CUDA 12.4.
```
conda env create -f enviroment.yml 
```
Once you install torch in your environment please intall the missing requirements with the following comand.
```
pip install -r requirements.txt
```
-------
Aditionally, if you want to run experiments using the dynamic proposal network in the models folder, you need to install some extra requirements. Please follow the instructions from [NVlabs](https://github.com/NVlabs/tiny-cuda-nn#requirements) to install tiny-cuda-nn and [NeRFAcc](https://github.com/nerfstudio-project/nerfacc#installation) installation process to install NeRFAcc.
# Citation
```bibtex
@article{InvNeRF2024,
   author = {Gerardo Loza and Junlei Hu and Dominic Jones and Pietro Valdastri and Sharib Ali},
   title = {Real-time surgical tool detection with multi-scale positional encoding and contrastive learning},
   keywords = {long-term point tracking, test-time optimisation, invertible NeRF (InvNeRF), consistent motion, geometric consistency},
   year = {2024},
}
```