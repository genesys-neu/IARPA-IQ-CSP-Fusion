# ICARUS: Learning on IQ and Cycle Frequencies for Detecting Anomalous RF Underlay Signals

This github repo releases the codes for different fusion frameworks of a submitted paper on three different dataset containing IQ samples and Cyclostationary Signal Processing (CSP) features. 

The Datasets: Three datasets are used for this paper.
* Synthetic Dataset: Matlab generated dataset with standard compliant waveforms (link: TBA)
* Indoor OTA-PAWR Dataset: OTA dataset collected in POWDER platform (link: TBA)
* Indoor OTA-Cellular Dataset: OTA datset collected in the Wild (link: TBA)

# Contents
* [Overview](#overview)
* [IQ Pipeline](#iq-pipeline)
* [CSP Pipeline](#csp-pipeline)
* [Fusion Pipeline](#fusion-pipeline)



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


The main file to run the codes is `anomaly_detection_main.py`. The detailed examples are given on how to run (a) IQ only, (b) CSP only, and (c) fusion pipelines. 

## Important Parameters: 
* `input`: [`iq`, `nc`, 'c'] DEFAULT: `iq` (for IQ Pipeline: `iq`, CSP Pipeline: `nc`, Fusion Pipeline `iq` `nc`)
* `fusion_layer`: [`penultimate`, `ultimate`] DEFAULT: `ultimate`
* `real_data`: [`True`, `False`] DEFAULT: `False` (hence Matlab dataset will be selected by default, if TRUE OTA Cellular dataset will be selected)
* `powder_data`: [`True`, `False`] DEFAULT: `False` (hence OTA POWDER dataset will not be selected)
* `dsss_type`: [`all`, `real`, `synthetic`] DEFAULT: `all` (hence all type of DSSS signals will be selected, valid when `real_data = True`)
* `iq_slice_len`: [`131072`, `262144`, `524288`] DEFAULT: `131072` (hence the IQ samples of length `131072` and corresponding CSP features will be processed)
* `strategy`: [`0`,`1`, `2`, `3`, `4`] DEFAULT: `4` (Different strategies used for CSP feature processing: naive (0), 2D matrix (1), extract stat (2), 3D matrix (3), extract stat specific max (4))
* `random_test`: [`True`, `False`] DEFAULT: `True` (hence perform train/test on 80/20 of data)
* `random_snr_test`: [`True`, `False`] DEFAULT: `True` (hence train and test on specific SNR values)
* `random_snr`: [`0`, `5`, `10`] DEFAULT: [`0`, `5`, `10`] (LTE signals of all these SNR values will be used)
* `dsss_sir`: [`0`, `5`, `10`] DEFAULT: [`0`, `5`, `10`] (DSSS signals of all these SIR values will be used)
* `restore_models`: [`True`, `False`] DEFAULT: `False` (hence trained for scratch for each modality)
* `retrain`: [`True`, `False`] DEFAULT: `False`

# IQ Pipeline

The IQ pipleine can be trained using different learning rates (`lr`, DEFAULT: `0.0001`), batch size (`bs`, DEFAULT: `8`), epochs (`epochs`, DEFAULT: `50`).

One example of training the IQ pipleine on OTA Cellular dataset with IQ block length of 131072:
```
python anomaly_detection_main.py \
    --data_folder /path/to/directory/IARPA-IQ-CSP-Fusion/ \
    --model_folder /path/to/directory/IARPA-IQ-CSP-Fusion/ \
    --input iq \
    --real_data True \
    --powder_data False \
    --iq_slice_len 131072 \ 
    --random_test True \
    --random_test_blocks 131072 \
    --dsss_type all \
    --input iq \
    --lr 0.0001 \
    --bs 8 \
    --epochs 50 \
```
The trained model will be saved in the model folder (`/path/to/directory/IARPA-IQ-CSP-Fusion/`). This pipeline can be run on any of the three mentioned datasets.


# CSP Pipeline

The CSP pipleine can be trained using different learning rates (`lr`, DEFAULT: `0.0001`), batch size (`bs`, DEFAULT: `8`), epochs (`epochs`, DEFAULT: `50`).

One example of training the IQ pipleine on OTA POWDER dataset with IQ block length of 524288:
```
python anomaly_detection_main.py \
    --data_folder /path/to/directory/IARPA-IQ-CSP-Fusion/ \
    --model_folder /path/to/directory/IARPA-IQ-CSP-Fusion/ \
    --input nc \
    --real_data False \
    --powder_data True \
    --iq_slice_len 524288 \ 
    --random_test True \
    --random_test_blocks 524288 \
    --dsss_type all \
    --input iq \
    --lr 0.0001 \
    --bs 8 \
    --epochs 50 \
```
The trained model will be saved in the model folder (`/path/to/directory/IARPA-IQ-CSP-Fusion/`). This pipeline can be run on any of the three mentioned datasets.




# Fusion Pipeline
The fusion pipleine can be trained using different learning rates (`lr`, DEFAULT: `0.0001`), batch size (`bs`, DEFAULT: `8`), epochs (`epochs`, DEFAULT: `50`).

One example of training the IQ pipleine on Matlab dataset with IQ block length of 262144:
```
python anomaly_detection_main.py \
    --data_folder /path/to/directory/IARPA-IQ-CSP-Fusion/ \
    --model_folder /path/to/directory/IARPA-IQ-CSP-Fusion/ \
    --input iq nc\
    --real_data False \
    --powder_data False \
    --iq_slice_len 262144 \ 
    --random_test True \
    --random_test_blocks 262144 \
    --dsss_type all \
    --input iq \
    --lr 0.0001 \
    --bs 8 \
    --epochs 50 \
```
The trained model will be saved in the model folder (`/path/to/directory/IARPA-IQ-CSP-Fusion/`). This pipeline can be run on any of the three mentioned datasets.


These three mentioned pipleine can also be run to detect the variant of DSSS signal (whether `BPSK` or `QPSK` modulation) using `variant_detection_main.py` in similar manner. 
