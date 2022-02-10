"""
@author: debashri
This file contains different utility fuctions to help the main.py
write_pickle: writes the radar samples in the pickle file
load_nonconjugate_feat: loads non-conjugate features
read_nonconjugate_feat: reads non-conjugate features
load_conjugate_feat: loads conjugate features
read_conjugate_feat: reads conjugate features
"""

import pandas as pd
from sklearn import preprocessing
import pickle
from random import choices
import os
import re
import numpy as np
import scipy
import pandas as pd


def read_and_extract_metadata(metadata_path, meta_filename, feature_shape, num_classes):
    one_label_file_df = pd.read_csv(metadata_path + meta_filename + '.csv', sep=",", header=0).dropna(axis=1,
                                                                                                         how='all')

    one_file_label = np.zeros((feature_shape, num_classes)) # LTE, DSSS-BPSK, DSSS-QPSK, DSSS spreading gain (5) - total 8

    if 'LTE_DSSS' in meta_filename:
        # print("one_label_file_df: ", one_label_file_df)
        # print("Columns: ", list(one_label_file_df.columns))
        # print("DSSS Modulation: ", one_label_file_df['DSSS_Mod'])
        # print("DSSS Spreading Gain: ", one_label_file_df['DSSS_PolyDegree_M'])
        one_file_label[:, 0] = 1
        # if one_label_file_df['DSSS_Mod'].to_numpy() == 'BPSK':
        #     one_file_label[:, 1] = 1
        if one_label_file_df['DSSS_Mod'].to_numpy() == 'QPSK':
            one_file_label[:, 1] = 1
        # considering the spreading gains
        if one_label_file_df['DSSS_SpreadingGain'].to_numpy() == 63:
            one_file_label[:, 2] = 1
        if one_label_file_df['DSSS_SpreadingGain'].to_numpy() == 127:
            one_file_label[:, 3] = 1
        if one_label_file_df['DSSS_SpreadingGain'].to_numpy() == 255:
            one_file_label[:, 4] = 1
        if one_label_file_df['DSSS_SpreadingGain'].to_numpy() == 511:
            one_file_label[:, 5] = 1
        if one_label_file_df['DSSS_SpreadingGain'].to_numpy() == 1023:
            one_file_label[:, 6] = 1

        # SIR tag: 'LTE_DSSS_SIR_dB'


        # lte_dsss_count = lte_dsss_count + 1

    # else: # Only LTE
    #     pass
        # one_file_label[:, 0] = 1
        # only_lte_count = only_lte_count + 1
    # print("For ", meta_filename, " the labels are: ", one_file_label)
    return one_file_label

def extract_snr_from_metadata(metadata_path, meta_filename):
    one_label_file_df = pd.read_csv(metadata_path + meta_filename + '.csv', sep=",", header=0).dropna(axis=1,
                                                                                                         how='all')
    return one_label_file_df['LTE_SNR_dB'].to_numpy()


def extract_sir_from_metadata(metadata_path, meta_filename):
    one_label_file_df = pd.read_csv(metadata_path + meta_filename + '.csv', sep=",", header=0).dropna(axis=1,
                                                                                                         how='all')
    return one_label_file_df['LTE_DSSS_SIR_dB'].to_numpy()

def read_processed_feat(feature_path, feature_type = 'nc', feature_options=[0], blocks = [131072], num_classes=2, snr_list = [0, 5, 10], sir_list=[0, 5, 10]):
    """
    This function is to load non-conjugate features in a specific session at a particular time.
    :param feature_path: str, the file path for the feature
    :param feature_type: str, the feature type: 'nc' or 'c'
    """
    inputs = np.zeros(1)
    labels =np.zeros(1)
    # xtest = np.zeros(1)
    #
    metadata_path = feature_path + 'IQDataSet_LTE_DSSS_v2/Metadata/'
    feature_path = feature_path + 'NEU_LTE_DSSS_Dataset_2_CSP/'
    # feature_path = feature_path + 'test/'

    count = 0
    only_lte_count = 0
    lte_dsss_count = 0

    for filename in os.listdir(feature_path):
        sir_for_this_file = sir_list[0]  # intitlizing the SIR values for only LTE signals
        # print("**********At Starting: ", filename, sir_for_this_file, sir_list)
        file_extension = '.NC'
        if feature_type == 'c':
            file_extension = '.C'
        if file_extension in filename and any(str(ext) in filename for ext in blocks):
            # creating the features
            # print('File with the features:', filename)
            meta_filename = filename[::-1].split('_',2)[-1][::-1]
            snr_for_this_file = extract_snr_from_metadata(metadata_path, meta_filename)
            if 'LTE_DSSS' in filename:
                sir_for_this_file = extract_sir_from_metadata(metadata_path, meta_filename)
            # print("THE SIR VALUES:  ", snr_for_this_file.item(), sir_list)
            # print("**********Before Entering: ", filename, sir_for_this_file, sir_list, snr_for_this_file, snr_list)
            if ((snr_for_this_file in snr_list) and (sir_for_this_file in sir_list)): # the second part of the condition is not being used for 'only lte' signals
                # print("**********Entering: ", filename, sir_for_this_file, sir_list, snr_for_this_file, snr_list)
                try:
                    one_file_df = pd.read_csv(feature_path+filename, sep=" ", header=None).dropna(axis=1, how='all')
                    column_index = list(one_file_df.columns)
                    selected_columns = [column_index[i] for i in feature_options]
                    # print("one_df and input shape:", one_file_df[selected_columns].to_numpy().shape, inputs.shape)
                    one_file_np = one_file_df[selected_columns].to_numpy()

                except pd.errors.EmptyDataError:
                    # print("Note:", filename, "was empty. Skipping.")
                    # count = count - 1 # decreasing the total file count
                    # one_file_df =
                    one_file_np = np.zeros((1, len(feature_options)))
                    # continue

                if count == 0:
                    inputs = one_file_np
                    # print("coming here..", inputs.shape)
                else:
                    inputs = np.concatenate((inputs, one_file_np), axis=0)

                # Creating the labels: 2 Label version - commented and kept aside
                # one_file_label = np.zeros((one_file_np.shape[0], 2))
                # if 'LTE_DSSS' in filename:
                #     one_file_label[:, 1] = 1
                #     lte_dsss_count = lte_dsss_count +1
                #
                # else:
                #     only_lte_count = only_lte_count + 1
                # if count == 0:
                #     labels = one_file_label
                #     # print("coming here too..", labels.shape)
                # else:
                #     labels = np.concatenate((labels, one_file_label), axis=0)
                #
                # 8 label version

                one_file_label = read_and_extract_metadata(metadata_path, meta_filename, one_file_np.shape[0], num_classes)
                if count == 0:
                    labels = one_file_label
                    # print("coming here too..", labels.shape)
                else:

                    labels = np.concatenate((labels, one_file_label), axis=0)

                #Counting the files:
                if 'LTE_DSSS' in filename:
                    lte_dsss_count = lte_dsss_count +1
                else:
                    only_lte_count = only_lte_count + 1
                #total files
                count = count + 1


    # display DataFrame
    print("Input shape: ", inputs.shape)
    print("Label shape: ", labels.shape)
    print("TOTAL READ FILES: ", count)
    print("TOTAL LTE FILES: ", only_lte_count)
    print("TOTAL LTE and DSSS FILES: ", lte_dsss_count)
    return inputs, labels
