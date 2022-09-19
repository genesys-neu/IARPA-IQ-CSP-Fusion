# ICARUS: Learning on IQ and Cycle Frequencies for Detecting Anomalous RF Underlay Signals

This github repo releases the codes for different fusion frameworks of a submitted paper on three different dataset containing IQ samples and Cyclostationary Signal Processing (CSP) features. 

The Datasets: Three datasets are used for this paper.
* Synthetic Dataset: Matlab generated dataset with standard compliant waveforms (link: TBA)
* Indoor OTA-PAWR Dataset: OTA dataset collected in POWDER platform (link: TBA)
* Indoor OTA-Cellular Dataset: OTA datset collected in the Wild (link: TBA)

# Contents
* [Overview](#overview)
* [Unimodal Networks](#unimodal-networks)
    *  [Radar Model](#radar-model)
    *  [Image Model](#image-model)
* [Fusion Networks](#fusion-networks)
    *  [Aggregated Fusion](#aggregated-fusion)
    *  [Incremental Fusion](#incremental-fusion)


# Overview
This repository presents a machine learning based framework that offers choices at the physical layer for inference with inputs of (i) in-phase and quadrature (IQ) samples only, (ii) cycle frequency features obtained via cyclostationary signal processing (CSP), and (iii) fusion of both, to detect the underlay DSSS signal and its modulation type within LTE frames. ICARUS chooses the best inference method considering both the expected accuracy and the computational overhead. ICARUS is rigorously validated on multiple real-world datasets that include signals captured in cellular bands in the wild and the NSF POWDER testbed for advanced wireless research (PAWR). We encourage the community to use the fusion network ideas from this repository or our paper published in archive.

The overall implementation for NuScene dataset is devided by two folds: (1) first we implement 2 different unimodal neworks for image and radar fined tuned to corresponsing features in NuScene dataset; (2) second we implement fusion between these two modalities. 

**Signals** (2): `['LTE', 'LTE + DSSS']`

**Number of Samples**: Synthetic: 210 (LTE) + 420 (LTE + DSSS); OTA-Indoor: 500 (LTE) + 2000 (LTE + DSSS); OTA-Cellular: 2430 (LTE) + 2430 (LTE + DSSS)

**Used Modalities**: `['IQ Samples', 'CSP Features']`

**Perforence Metric**: Accuracy in %

Paths for required features to run the codes:

Model Path: `D:/IARPA_DATA/Saved_Models/` or `/path/to/directory/IARPA-IQ-CSP-Fusion/` (please change as per your setting)

Data Path: `D:/IARPA_DATA/` or `/path/to/directory/IARPA-IQ-CSP-Fusion/`  (please change as per your setting)


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


