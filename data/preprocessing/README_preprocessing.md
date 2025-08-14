# Preprocessing
In this section we use the sample data, [`InvNeRF/data/Sample_data.zip`](https://github.com/L0za007/InvNeRF/tree/main/data), to show how to preprocess the data and get the folder structure required by InvNeRF. Run the following commands to extract the data and move to the main directory. 
```
cd data 
unzip Sample_data.zip
mv Sample_data ../
```
Extract the frames from the videos wiht the following command
```
python preprocessing/process_video.py --data_dir ../Sample_data/case02_seq05
```
Additionally, if you want to obtain the tool mask for each frame, please install SAM2 following the instructions in the official [documentation](https://github.com/facebookresearch/sam2/tree/main). Then run the command below, it will use the point prompts in `Tool_mask_prompts.json` to segment the tools in the first frame and propagate to the rest. You can find all the prompts that we used for masking the tools in the videos from the STIR_Origin and the STIR_challenge datasets in [`InvNeRF/data/labels`](https://github.com/L0za007/InvNeRF/tree/main/data/labels).

```
python preprocessing/process_video.py --data_dir ../Sample_data/case02_seq05 --gen_mask
```
We made some modifications to the functions in Omnimotion that extract correspondences using RAFT. Our version allows us to control the number of frame pairs per time step and the gap between pairs. We also provide the files to refine RAFT correspondences with CoTracker and obtain chained MFT correspondences. Start by running the following commands to clone the needed repositories and move our files to the appropriate location. 
```
cd preprocessing
git clone https://github.com/facebookresearch/dino.git
git clone https://github.com/princeton-vl/RAFT.git
git clone https://github.com/serycjon/MFT.git

mv data/preprocessing/*raft.py data/preprocessing/RAFT/
mv data/preprocessing/extract_dino_features.py data/preprocessing/dino
mv data/preprocessing/*MFT.py data/preprocessing/MFT/
```

Following the experimental setup described in our paper and depending on the correspondences you want to use to train InvNeRF, you can run one the following commands. 

Run the preprocessing pipeline using RAFT
```
python gen_corr.py --data_dir <Path-to-your-data>/Sample_data/case02_seq05 --stereo --gap 2
```

Run the preprocessing pipeline using RAFT+CoTracker
```
python gen_corr.py --data_dir <Path-to-your-data>/Sample_data/case02_seq05 --stereo --CoTracker --gap 2
```

Run the preprocessing pipeline using MFT
```
python gen_corr.py --data_dir <Path-to-your-data>/Sample_data/case02_seq05 --stereo --gap 8 --MFT
```