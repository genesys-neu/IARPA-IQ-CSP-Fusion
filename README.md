# ICARUS: Learning on IQ and Cycle Frequencies for Detecting Anomalous RF Underlay Signals

This github repo releases the codes for different fusion frameworks of a submitted paper on three different dataset containing IQ samples and Cyclostationary Signal Processing (CSP) features. 

The Dataset: 
*The dataset is publicly available in: TBA

# Contents
* [Overview](#overview)
* [Unimodal Networks](#unimodal-networks)
    *  [Radar Model](#radar-model)
    *  [Image Model](#image-model)
* [Fusion Networks](#fusion-networks)
    *  [Aggregated Fusion](#aggregated-fusion)
    *  [Incremental Fusion](#incremental-fusion)


# Overview

This project implements multi-vehicle detection using four (image, radar, acoustic, seismic) modalities on ESCAPE and NuScene dataset. In this repository, we give the codes of our implementation in NuScene dataset which used two of these modalities and two fusion networks. We have multiple other proposed fusion networks on ESCAPE dataset, which are described in details in the aforementioned paper. This repository does not release the datasets, you should download or get access to the datasets seperately. We encourage the community to use the fusion network ideas from this repository or our paper published in IEEE Transactions on Multimedia (citing the mentioned paper).

The overall implementation for NuScene dataset is devided by two folds: (1) first we implement 2 different unimodal neworks for image and radar fined tuned to corresponsing features in NuScene dataset; (2) second we implement fusion between these two modalities. 

Here are the necessary descriptions of the part of the NuScene dataset used here:

**Vehicles** (4): `['construction vehicles', 'motorcycle', 'trailer', 'truck]`

**Number of Scenes**: 850

**Used Sensors**: `['CAM_FRONT', 'RADAR_FRONT']`

**Perforence Metric**: Area under the ROC Curve (`AUC`) and Average Precision (AP)

Paths for required features to run the codes:

Model Path: `E:/ESCAPE_data/Saved_Models/NuScene/` or `/path/to/directory/Multimodal_Fusion_NuScene/NuScene/` (please change as per your setting)

Data Path: `E:/ESCAPE_data/NuScene_data/miniset/` or `/path/to/directory/Multimodal_Fusion_NuScene/NuScene_data/miniset/`  (please change as per your setting - you should download 'v1.0-trainval' folder from NuScene dataset page)


The main file to run the codes is `main_nuscene.py`. The detailed examples are given on how to run each type of unimodal and fusion networks. Using `main_nuscene.py` file you can run: 
1. unimodal radar network
2. unimodal image network
3. aggregated fusion among (radar, image)
4. incremental fusion among (radar, image)


# Unimodal Networks
The unimodal networks for radar, and image  can be trained on the processes features (please notice to give the correct path for the features). Please keep the same folder structure as in the original processed feature folder for each modalities. 

## Radar Model
The radar model can be trained using different learning rates (`lr`, DEFAULT: `0.0001`), batch size (`bs`, DEFAULT: `32`), epochs (`epochs`, DEFAULT: `40`).

One example of training the radar model and getting AUC and AP per vehicle:
```
python main_nuscene.py \
    --data_folder /path/to/directory/Multimodal_Fusion_NuScene/NuScene_data/miniset/ \
    --model_folder /path/to/directory/Multimodal_Fusion_NuScene/Saved_Models/NuScene/ \
    --input radar \
    --lr 0.0001 \
    --bs 32 \
    --epochs 40 \
```
The trained model will be saved in the model folder (`/path/to/directory/Multimodal_Fusion_NuScene/Saved_Models/NuScene/`).

## Image Model
The image model can be trained using different learning rates (`lr`, DEFAULT: `0.0001`), batch size (`bs`, DEFAULT: `32`), epochs (`epochs`, DEFAULT: `40`).

One example of training the image model and getting AUC and AP per vehicle:
```
python main_nuscene.py \
    --data_folder /path/to/directory/Multimodal_Fusion_NuScene/NuScene_data/miniset/ \
    --model_folder /path/to/directory/Multimodal_Fusion_NuScene/Saved_Models/NuScene/ \
    --input image \
    --lr 0.0001 \
    --bs 32 \
    --epochs 40 \
```
The trained model will be saved in the model folder (`/path/to/directory/Multimodal_Fusion_NuScene/Saved_Models/NuScene/`).

# Fusion Networks
After we have trained models for radar, and image, we perform two different types of fusions among them. Different fusion techniques are described bellow. All the fusion networks can be run using the same `main_nuscene.py`. In each run you can either use one of the fusion strategies: (i) `concat`, (ii) `lr_tensor`, (iii)`mi`. These fusion strategies corresponds to concatenation, low rank tensor fusion, and multiplicative interactions. The implementations are taken and modified from https://github.com/pliang279/MultiBench, given in `common_fusions` folder. For the options (3), (4), you can set fusion layers: (i) `ultimate`, (ii) `penultimate`.


## Aggregated Fusion
Aggregated fusion fuses radar and iamge features either at `ultimate` or `penultimate` layer by using either of the fusion strategies. The fusion model can be trained using different learning rates (`lr`, DEFAULT: `0.0001`), batch size (`bs`, DEFAULT: `32`), epochs (`epochs`, DEFAULT: `40`). 

#### Important Parameters: 
* `fusion_layer`: [`penultimate`, `ultimate`] DEFAULT: `ultimate`
* `fusion_techniques`: [`mi`, `lr_tensor`, `concat`] DEFAULT: `concat`
* `restore_models`: [`True`, `False`] DEFAULT: `False` (hence trained for scratch for each modality)
* `retrain`: [`True`, `False`] DEFAULT: `False`
* `state_fusions`: [`True`, `False`] DEFAULT: `False`

One example of training the aggregated fusion model at `penultimate` layer using low rank fusion strategy with restoring the unimodal models and retraining, and getting prediction AUCs and APs per vehicle:

```
python main_nuscene.py \
    --data_folder /path/to/directory/Multimodal_Fusion_NuScene/NuScene_data/miniset/ \
    --model_folder /path/to/directory/Multimodal_Fusion_NuScene/Saved_Models/NuScene/ \
    --input radar \
    --lr 0.0001 \
    --bs 32 \
    --epochs 40 \
    --state_fusions True \ 
    --fusion_layer penultimate \
    --fusion_techniques lr_tensor\
    --restore_models True \
    --retrain True 
```

The trained model will be saved in the model folder (`/path/to/directory/Multimodal_Fusion_NuScene/Saved_Models/NuScene/`).

## Incremental Fusion
Incremental fusion fuses radar, acoustic and seismic features either at `ultimate` or `penultimate` layer successively. It will use one of the pretrained model of `acoustic_radar_incremental_penultimate.pt` or `acoustic_radar_incremental.pt` from the model folder. The fusion model can be trained using different learning rates (`lr`, DEFAULT: `0.0001`), batch size (`bs`, DEFAULT: `32`), epochs (`epochs`, DEFAULT: `40`). 

#### Important Parameters: 
* `incremental_fusion`: [`True`, `False`] DEFAULT: `False` (this will run the aggregated fusion)
* `fusion_layer`: [`penultimate`, `ultimate`] DEFAULT: `ultimate`
* `fusion_techniques`: [`mi`, `lr_tensor`, `concat`] DEFAULT: `concat`
* `state_fusions`: [`True`, `False`] DEFAULT: `False`

One example of training the incremental fusion model at `penultimate` layer using concatenation as fusion strategy, and getting prediction AUCs and APs per vehicle:

```
python non-image-fusion-main.py \
    --data_folder /path/to/directory/Multimodal_Fusion_NuScene/NuScene_data/miniset/ \
    --model_folder /path/to/directory/Multimodal_Fusion_NuScene/Saved_Models/NuScene/ \
    --input radar image \
    --lr 0.0001 \
    --bs 32 \
    --epochs 40 \
    --state_fusions False \ 
    --fusion_layer penultimate \
    --incremental_fusion True \
```
The trained model will be saved in the model folder (`/path/to/directory/Multimodal_Fusion_NuScene/Saved_Models/NuScene/`).


