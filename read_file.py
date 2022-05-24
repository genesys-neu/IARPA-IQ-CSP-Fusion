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
import glob
from random import random
from collections import defaultdict
from pathlib import Path


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

# Generate the labels from meta data
def generate_labels(metadata_path, meta_filename, num_classes):
    one_file_label = np.zeros((1, num_classes))
    # print("The metadata file name: ", (metadata_path + meta_filename + '.csv'))
    one_label_file_df = pd.read_csv(metadata_path + meta_filename + '.csv', sep=",", header=0).dropna(axis=1, how='all')
    if one_label_file_df['Signal_Type'].to_numpy() == 'LTE':
        one_file_label[0, 0] = 1
    if one_label_file_df['Signal_Type'].to_numpy() == 'LTE_DSSS':
        one_file_label[0, 1] = 1
    return one_file_label

# extract the statistics about the CSP features
def extract_statistic_from_CSP_features(CSP_features_one_file):
    mean = np.max(CSP_features_one_file, axis=0)
    return mean

# extract the statistics about the CSP features
def extract_statistic_from_CSP_nc_features_one_col(CSP_features_one_file):
    # print("coming here")
    row_with_max = np.argmax(CSP_features_one_file[:,1], axis=0) # for non-conjugate features 2nd column, for conjugate features 4th column is important
    # print("important information..", row_with_max, CSP_features_one_file[row_with_max, :], CSP_features_one_file)
    return CSP_features_one_file[row_with_max, :]

# extract the statistics about the CSP features
def extract_statistic_from_CSP_c_features_one_col(CSP_features_one_file):
    # print("coming here")
    row_with_max = np.argmax(CSP_features_one_file[:,3], axis=0) # for non-conjugate features 2nd column, for conjugate features 4th column is important
    # print("important information..", row_with_max, CSP_features_one_file[row_with_max, :], CSP_features_one_file)
    return CSP_features_one_file[row_with_max, :]

# generate one CSP feature in form of 2D matrix
def organize_CSP_features_2D(CSP_features_one_file):
    channels = CSP_features_one_file.shape[1]
    rows_to_be_taken = 100
    rows_to_be_added = 0
    columnIndex = 1 # sorting by the second column
    CSP_features_one_file = CSP_features_one_file[(-CSP_features_one_file[:, columnIndex]).argsort()] # sort the array in decending order by second column
    if CSP_features_one_file.shape[0] <rows_to_be_taken:
        rows_to_be_added = rows_to_be_taken - CSP_features_one_file.shape[0]
    else:
        CSP_features_one_file = CSP_features_one_file[0:rows_to_be_taken, :]

    # print("shape 1: ", CSP_features_one_file.shape)
    CSP_features_one_file = np.pad(CSP_features_one_file, [(0, rows_to_be_added), (0, 0)], mode='constant', constant_values=0)
    # print("******testing*******: ", CSP_features_one_file.shape)

    return CSP_features_one_file

# generate one CSP feature in form of 3D matrix
def organize_CSP_features_3D(CSP_features_one_file):
    channels = CSP_features_one_file.shape[1]
    rows_to_be_taken = 100
    rows_to_be_added = 0
    columnIndex = 1 # sorting by the second column
    # CSP_features_one_file = CSP_features_one_file[np.argsort(CSP_features_one_file)] #[-rows_to_be_taken:]
    # CSP_features_one_file = CSP_features_one_file[CSP_features_one_file[:, columnIndex].argsort()]
    CSP_features_one_file = CSP_features_one_file[(-CSP_features_one_file[:, columnIndex]).argsort()] # sort the array in decending order by second column
    if CSP_features_one_file.shape[0] <rows_to_be_taken:
        rows_to_be_added = rows_to_be_taken - CSP_features_one_file.shape[0]
    else:
        CSP_features_one_file = CSP_features_one_file[0:rows_to_be_taken, :]

    # print("shape 1: ", CSP_features_one_file.shape)
    CSP_features_one_file = np.pad(CSP_features_one_file, [(0, rows_to_be_added), (0, 0)], mode='constant', constant_values=0)
    # print("shape 2: ", CSP_features_one_file.shape)
    columns = CSP_features_one_file.shape[0]
    CSP_features_one_file = np.expand_dims(CSP_features_one_file, axis=1)
    # print("shape 3: ", CSP_features_one_file.shape)
    CSP_features_one_file = np.swapaxes (CSP_features_one_file, 0, 1)
    # print("shape 4: ", CSP_features_one_file.shape)
    # CSP_features_one_file = np.swapaxes(CSP_features_one_file, 0, 1)
    # print("shape 5: ", CSP_features_one_file.shape)
    # CSP_features_one_file = np.reshape(CSP_features_one_file, (1, columns, channels))
    return CSP_features_one_file


# read one IQ file
def read_one_iq(iq_path, filename, slice_len):
    dtype_all = scipy.dtype([('raw-iq', scipy.complex64)])
    with open(iq_path + filename, mode='rb') as file:  # b is important -> binary
        iqdata_one_file = scipy.fromfile(file, dtype=dtype_all)
    iqdata_one_file = np.reshape(iqdata_one_file[:(iqdata_one_file.shape[0] // slice_len) * slice_len], (iqdata_one_file.shape[0] // slice_len, slice_len))  # discard the extra elements for uneven array
    # print("After: ", iqdata_one_file.shape)
    iqdata_one_file = np.expand_dims(iqdata_one_file, axis=2)
    iqdata_one_file = np.concatenate([iqdata_one_file['raw-iq'].real,
                                      iqdata_one_file['raw-iq'].imag], axis=2)
    return iqdata_one_file

#read one CSP (either conjugate/non-conjugate) feature file
def read_one_CSP_feature(filename, feature_options):
    try:
        one_file_df = pd.read_csv(filename, sep=" ", header=None).dropna(axis=1, how='all')
        column_index = list(one_file_df.columns)
        selected_columns = [column_index[i] for i in feature_options]
        # print("one_df and input shape:", one_file_df[selected_columns].to_numpy().shape, inputs.shape)
        one_file_np = one_file_df[selected_columns].to_numpy()

    except pd.errors.EmptyDataError:
        # print("Note:", filename, "was empty. Skipping.")
        # count = count - 1 # decreasing the total file count
        # one_file_df =
        one_file_np = np.zeros((1, len(feature_options)))
    return one_file_np

#genearte the input and labels for all the modalities
def generate_inputs_labels_synthetic_data(feature_path, input_options = ['iq', 'nc', 'c'], feature_options=[0], slice_len = 256, num_classes=2, snr_list = [0, 5, 10], sir_list=[0, 5, 10], strategy = 4, percentage_to_read = 0.1):
    iq_data = np.zeros(1)
    csp_c_features = np.zeros(1)
    csp_nc_features = np.zeros(1)
    labels = np.zeros(1)
    metadata_path = feature_path + 'IQDataSet_LTE_DSSS_v2/Metadata/'

    # ORIGINAL PATH
    iq_path = feature_path + 'IQDataSet_LTE_DSSS_v2/IQ/'
    CSP_path = feature_path + 'NEU_LTE_DSSS_Dataset_2_CSP/'

    # TEST PATH
    # iq_path = feature_path + 'test/IQ/'
    # CSP_path = feature_path + 'test/CSP/'

    count_files = 0
    for filename in os.listdir(iq_path):
        if random() > percentage_to_read:
            # print("Continuing, ", filename)
            continue
        sir_for_this_file = sir_list[0]  # intitlizing the SIR values for only LTE signals
        meta_filename = filename[::-1].split('_', 0)[-1][::-1]
        # print("Meta_filename: ", meta_filename)
        snr_for_this_file = extract_snr_from_metadata(metadata_path, meta_filename)
        if 'LTE_DSSS' in filename:
            sir_for_this_file = extract_sir_from_metadata(metadata_path, meta_filename)
        # print("THE SIR VALUES:  ", sir_for_this_file, sir_list)
        # print("**********Before Entering: ", filename, sir_for_this_file, sir_list, snr_for_this_file, snr_list)
        if ((snr_for_this_file in snr_list) and (sir_for_this_file in sir_list)):
            # read the iq file
            if 'iq' in input_options:
                iqdata_one_file = read_one_iq(iq_path, filename, slice_len)
                print("Length of the iq  for ", filename, " is:", iqdata_one_file.shape)
                # adding to the output array
                if count_files == 0:  iq_data = iqdata_one_file
                else: iq_data = np.concatenate((iq_data, iqdata_one_file), axis=0)

            if 'c' in input_options:
                extrated_stats_c = np.zeros((iqdata_one_file.shape[0], len(feature_options)))
                csp_feature_count = 0
                for file_name in glob.iglob(CSP_path+filename+'_'+str(slice_len)+'*.C', recursive = True):
                    # print("The conjugate feature file name: ", file_name)
                    one_c_csp_file_np = read_one_CSP_feature(file_name, feature_options)
                    ################## (without any strategy implementation (old implementation) #######################
                    # extrated_stats_one_file_c = extract_statistic_from_CSP_features(one_c_csp_file_np)
                    ################## (end of without any strategy implementation (old implementation) #######################

                    ########################### new implementation #########################
                    if strategy == 2:
                        one_c_csp_file_np = extract_statistic_from_CSP_features(one_c_csp_file_np)
                        one_c_csp_file_np = np.expand_dims(one_c_csp_file_np, axis=0)
                    elif strategy == 4:
                        one_c_csp_file_np = extract_statistic_from_CSP_c_features_one_col(one_c_csp_file_np)
                        one_c_csp_file_np = np.expand_dims(one_c_csp_file_np, axis=0)
                    else:
                        print(
                            "Invalid strategy for handling CSP Features while using fusion, please use either 2 or 4.")
                        exit(0)
                    ########################### end of new implementation #########################
                    extrated_stats_c[csp_feature_count,:] = one_c_csp_file_np
                    csp_feature_count +=1
                    print("Shape of conjugate CSP features  for ", file_name, ":", one_c_csp_file_np.shape, one_c_csp_file_np.shape, extrated_stats_c.shape)
                    # adding to the cycle features
                if count_files == 0:
                    csp_c_features = extrated_stats_c
                else:
                    csp_c_features = np.concatenate((csp_c_features, extrated_stats_c), axis=0)

            if 'nc' in input_options:
                extrated_stats_nc = np.zeros((iqdata_one_file.shape[0], len(feature_options)))
                csp_feature_count = 0
                for file_name in glob.iglob(CSP_path+filename+'_'+str(slice_len)+'*.NC', recursive = True):
                    # print("The non conjugate feature file name: ", file_name)
                    one_nc_csp_file_np = read_one_CSP_feature(file_name, feature_options)
                    ################## (without any strategy implementation (old implementation) #######################
                    # extrated_stats_one_file_nc = extract_statistic_from_CSP_features(one_nc_csp_file_np)
                    ################## (end of without any strategy implementation (old implementation) #######################

                    ########################### new implementation #########################
                    if strategy == 2:
                        one_nc_csp_file_np = extract_statistic_from_CSP_features(one_nc_csp_file_np)
                        one_nc_csp_file_np = np.expand_dims(one_nc_csp_file_np, axis=0)
                    elif strategy == 4:
                        one_nc_csp_file_np = extract_statistic_from_CSP_nc_features_one_col(one_nc_csp_file_np)
                        one_nc_csp_file_np = np.expand_dims(one_nc_csp_file_np, axis=0)
                    else:
                        print("Invalid strategy for handling CSP Features while using fusion, please use either 2 or 4.")
                        exit(0)
                    ########################### end of new implementation #########################
                    extrated_stats_nc[csp_feature_count, :] = one_nc_csp_file_np
                    csp_feature_count += 1
                    print("Shape of conjugate CSP features  for ", file_name, ":", one_nc_csp_file_np.shape,
                          one_nc_csp_file_np.shape, extrated_stats_nc.shape)
                    # adding to the cycle features
                if count_files == 0:
                    csp_nc_features = extrated_stats_nc
                else:
                    csp_nc_features = np.concatenate((csp_nc_features, extrated_stats_nc), axis=0)

            # read the corresponding label
            one_label = generate_labels(metadata_path, meta_filename, num_classes)
            one_label = np.repeat(one_label, iqdata_one_file.shape[0], axis=0)
            if count_files == 0:
                labels = one_label
            else:
                labels = np.concatenate((labels, one_label), axis=0)

            # print("the generated label  for ", filename, " is:", one_label)
            # print("Length of the iq and CSP features  for ", filename, " is:", iqdata_one_file.shape, one_c_csp_file_np.shape, one_nc_csp_file_np.shape, one_label.shape)
        count_files +=1
    return iq_data, csp_c_features, csp_nc_features, labels

def generate_inputs_labels_real_data(feature_path, input_options = ['iq', 'nc', 'c'], feature_options=[0], slice_len = 256, num_classes=2, snr_list = [0, 5, 10], sir_list=[0, 5, 10], strategy = 4, dsss_type = 'real', percentage_to_read = 0.1):
    iq_data = np.zeros(1)
    csp_c_features = np.zeros(1)
    csp_nc_features = np.zeros(1)
    labels = np.zeros(1)
    # metadata_path = feature_path + 'IQDataSet_LTE_DSSS_v2/Metadata/'

    # ORIGINAL PATH
    iq_path = feature_path + 'NWRA_data/IQ/'
    CSP_path = feature_path + 'NWRA_data/'+str(slice_len)+'/'

    # TEST PATH
    # iq_path = feature_path + 'test/IQ/'
    # CSP_path = feature_path + 'test/CSP/'

    count_files = 0
    only_lte_count = 0
    lte_dsss_count = 0
    for filename in os.listdir(iq_path):
            if len(filename.split('_')) <= 6:
                continue
            if random() > percentage_to_read:
            # print("Continuing, ", filename)
                continue
            if 'iq' in input_options:
                # with open(iq_path + filename, mode='rb') as file:  # b is important -> binary
                iqdata_one_file = read_one_iq(iq_path, filename, slice_len)
                # iqdata_one_file = iqdata_one_file[2:]  # skipping first two elements (as per Chad's binary.m file)

                # iqdata_one_file = read_one_iq(iq_path, filename, slice_len)
                print("Length of the iq  for ", filename, " is:", iqdata_one_file.shape)
                # adding to the output array
                # if count_files == 0:  iq_data = iqdata_one_file
                # else: iq_data = np.concatenate((iq_data, iqdata_one_file), axis=0)

            # read the corresponding label
            # Creating the labels: 2 Label version
            one_file_label = np.zeros((iqdata_one_file.shape[0], num_classes))
            if num_classes == 2:
                    flag = extract_IQ_metadata_from_NWRA(filename, dsss_type)
                    if 'SKIP' in flag:
                            continue
                    elif 'LTE_DSSS' in flag:
                            one_file_label[:, 1] = 1
                            lte_dsss_count = lte_dsss_count + 1
                    else:
                            one_file_label[:, 0] = 1
                            only_lte_count = only_lte_count + 1

            # adding /Q data and labels
            if count_files == 0:
                    iq_data = iqdata_one_file
                    labels = one_file_label
                            # print("coming here too..", labels.shape)
            else:
                    iq_data = np.concatenate((iq_data, iqdata_one_file), axis=0)
                    labels = np.concatenate((labels, one_file_label), axis=0)

            if 'c' in input_options:
                extrated_stats_c = np.zeros((iqdata_one_file.shape[0], len(feature_options)))
                csp_feature_count = 0

                for file_name in glob.iglob(CSP_path+os.path.splitext(filename)[0]+'_'+str(slice_len)+'*.C', recursive = True):
                    # print("The conjugate feature file name: ", file_name)
                    one_c_csp_file_np = read_one_CSP_feature(file_name, feature_options)
                    ################## (without any strategy implementation (old implementation) #######################
                    # extrated_stats_one_file_c = extract_statistic_from_CSP_features(one_c_csp_file_np)
                    ################## (end of without any strategy implementation (old implementation) #######################

                    ########################### new implementation #########################
                    if strategy == 2:
                        one_c_csp_file_np = extract_statistic_from_CSP_features(one_c_csp_file_np)
                        one_c_csp_file_np = np.expand_dims(one_c_csp_file_np, axis=0)
                    elif strategy == 4:
                        one_c_csp_file_np = extract_statistic_from_CSP_nc_features_one_col(one_c_csp_file_np)
                        one_c_csp_file_np = np.expand_dims(one_c_csp_file_np, axis=0)
                    else:
                        print("Invalid strategy for handling CSP Features while using fusion, please use either 2 or 4.")
                        exit(0)
                    ########################### end of new implementation #########################
                    extrated_stats_c[csp_feature_count,:] = one_c_csp_file_np
                    csp_feature_count +=1
                    print("Shape of conjugate CSP features  for ", file_name, ":", one_c_csp_file_np.shape, one_c_csp_file_np.shape, extrated_stats_c.shape)
                    # adding to the cycle features
                if count_files == 0:
                    csp_c_features = extrated_stats_c
                else:
                    csp_c_features = np.concatenate((csp_c_features, extrated_stats_c), axis=0)

            if 'nc' in input_options:
                extrated_stats_nc = np.zeros((iqdata_one_file.shape[0], len(feature_options)))
                csp_feature_count = 0
                # print("TEST: ", os.path.splitext(filename)[0])
                for file_name in glob.iglob(CSP_path+os.path.splitext(filename)[0]+'_'+str(slice_len)+'*.NC', recursive = True):
                    # print("The non conjugate feature file name: ", file_name)
                    one_nc_csp_file_np = read_one_CSP_feature(file_name, feature_options)
                    ################## (without any strategy implementation (old implementation) #######################
                    # extrated_stats_one_file_nc = extract_statistic_from_CSP_features(one_nc_csp_file_np)
                    ################## (end of without any strategy implementation (old implementation) #######################

                    ########################### new implementation #########################
                    if strategy == 2:
                        one_nc_csp_file_np = extract_statistic_from_CSP_features(one_nc_csp_file_np)
                        one_nc_csp_file_np = np.expand_dims(one_nc_csp_file_np, axis=0)
                    elif strategy == 4:
                        one_nc_csp_file_np = extract_statistic_from_CSP_nc_features_one_col(one_nc_csp_file_np)
                        one_nc_csp_file_np = np.expand_dims(one_nc_csp_file_np, axis=0)
                    else:
                        print("Invalid strategy for handling CSP Features while using fusion, please use either 2 or 4.")
                        exit(0)
                    ########################### end of new implementation #########################
                    extrated_stats_nc[csp_feature_count, :] = one_nc_csp_file_np
                    csp_feature_count += 1
                    print("Shape of conjugate CSP features  for ", file_name, ":", one_nc_csp_file_np.shape,
                          one_nc_csp_file_np.shape, extrated_stats_nc.shape)
                    # adding to the cycle features
                if count_files == 0:
                    csp_nc_features = extrated_stats_nc
                else:
                    csp_nc_features = np.concatenate((csp_nc_features, extrated_stats_nc), axis=0)

            # print("the generated label  for ", filename, " is:", one_label)
            # print("Length of the iq and CSP features  for ", filename, " is:", iqdata_one_file.shape, one_c_csp_file_np.shape, one_nc_csp_file_np.shape, one_label.shape)
            count_files +=1
    return iq_data, csp_c_features, csp_nc_features, labels

def read_iq_files(feature_path, blocks = [131072], slice_len = 256, num_classes=2, snr_list = [0, 5, 10], sir_list=[0, 5, 10], percentage_to_read = 0.1):
    inputs = np.zeros(1)
    labels = np.zeros(1)

    metadata_path = feature_path + 'IQDataSet_LTE_DSSS_v2/Metadata/'
    iq_path = feature_path + 'IQDataSet_LTE_DSSS_v2/IQ/'
    # iq_path = feature_path + 'test/IQ/'

    count = 0
    only_lte_count = 0
    lte_dsss_count = 0
    dtype_all = scipy.dtype([('raw-iq', scipy.complex64)])  # gr_complex is '32fc' --> make any sense?

    for filename in os.listdir(iq_path):
        if random() > percentage_to_read:
            # print("Continuing, ", filename)
            continue
        sir_for_this_file = sir_list[0]  # intitlizing the SIR values for only LTE signals
        meta_filename = filename[::-1].split('_', 0)[-1][::-1]

        snr_for_this_file = extract_snr_from_metadata(metadata_path, meta_filename)
        if 'LTE_DSSS' in filename:
            sir_for_this_file = extract_sir_from_metadata(metadata_path, meta_filename)
        # print("THE SIR VALUES:  ", sir_for_this_file, sir_list)
        # print("**********Before Entering: ", filename, sir_for_this_file, sir_list, snr_for_this_file, snr_list)
        if ((snr_for_this_file in snr_list) and (sir_for_this_file in sir_list)): # the second part of the condition is not being used for 'only lte' signals
            # print("**********Entering: ", filename, sir_for_this_file, sir_list, snr_for_this_file, snr_list)
            with open(iq_path+filename, mode='rb') as file:  # b is important -> binary
                iqdata_one_file = scipy.fromfile(file, dtype=dtype_all)
            iqdata_one_file = np.reshape(iqdata_one_file[:(iqdata_one_file.shape[0]//slice_len)*slice_len], (iqdata_one_file.shape[0]//slice_len,slice_len)) # discard the extra elements for uneven array
            # print("After: ", iqdata_one_file.shape)
            iqdata_one_file = np.expand_dims(iqdata_one_file,axis=2)
            iqdata_one_file = np.concatenate([iqdata_one_file['raw-iq'].real,
                                         iqdata_one_file['raw-iq'].imag], axis=2)

            if count == 0:
                inputs = iqdata_one_file
                # print("coming here..", inputs.shape)
            else:
                inputs = np.concatenate((inputs, iqdata_one_file), axis=0)

            # Creating the labels: 2 Label version
            if num_classes == 2:
                one_file_label = np.zeros((iqdata_one_file.shape[0], num_classes))
                if 'LTE_DSSS' in filename:
                    one_file_label[:, 1] = 1
                    lte_dsss_count = lte_dsss_count + 1

                else:
                    one_file_label[:, 0] = 1
                    only_lte_count = only_lte_count + 1
                if count == 0:
                    labels = one_file_label
                    # print("coming here too..", labels.shape)
                else:
                    labels = np.concatenate((labels, one_file_label), axis=0)



            # 7-label version
            else:
                one_file_label = read_and_extract_metadata(metadata_path, meta_filename, iqdata_one_file.shape[0], num_classes)
                if count == 0:
                    labels = one_file_label
                    # print("coming here too..", labels.shape)
                else:

                    labels = np.concatenate((labels, one_file_label), axis=0)

                # Counting the files:
                if 'LTE_DSSS' in filename:
                    lte_dsss_count = lte_dsss_count + 1
                else:
                    only_lte_count = only_lte_count + 1
                # total files
            count = count + 1

    # display DataFrame
    print("Input shape: ", inputs.shape)
    print("Label shape: ", labels.shape)
    print("TOTAL READ FILES: ", count)
    print("TOTAL LTE FILES: ", only_lte_count)
    print("TOTAL LTE and DSSS FILES: ", lte_dsss_count)
    return inputs, labels

def extract_IQ_metadata_from_NWRA(filename, dsss_type):
    loop_index = int(filename.split('_')[6]) # finding the loop index in the data
    # print("FLAG: ", filename, flag)
    # print("The loop index and dsss_type: ", loop_index, dsss_type)
    if loop_index%2 == 0: # loop index is even - only LTE
        # print("returning the loop index for lte: ", loop_index)
        return 'LTE'
    else: # loop index is odd  - DSSS present
        if dsss_type == 'synthetic' and loop_index > 40:
            # print("returning the loop index for dsss: ", loop_index)
            return 'SKIP'
        if dsss_type == 'real' and loop_index < 40:
            return 'SKIP'
        # print("returning the loop index for dsss: ", loop_index)
        return 'LTE_DSSS'


def read_real_iq_files(feature_path, blocks = [131072], slice_len = 256, num_classes=2, snr_list = [0, 5, 10], sir_list=[0, 5, 10], dsss_type = 'all', percentage_to_read = 0.1):
    inputs = np.zeros(1)
    labels = np.zeros(1)

    # metadata_path = feature_path + 'NWRA_data/Metadata/'
    iq_path = feature_path + 'NWRA_data/IQ/'
    # iq_path = feature_path + 'test/IQ/'

    count = 0
    only_lte_count = 0
    lte_dsss_count = 0
    dtype_all = scipy.dtype([('raw-iq', scipy.complex64)])  # gr_complex is '32fc' --> make any sense?
    # random_number = random.random()
    for filename in os.listdir(iq_path):
            # print("THe length of file", filename, " is : ", len(filename.split('_')))
            if len(filename.split('_')) <= 6:
                continue
            if random() > percentage_to_read:
                # print("Continuing, ", filename)
                continue
            # print("reading file: ", filename)
            with open(iq_path+filename, mode='rb') as file:  # b is important -> binary
                iqdata_one_file = scipy.fromfile(file, dtype=dtype_all)
                iqdata_one_file = iqdata_one_file[2:] # skipping first two elements (as per Chad's binary.m file)
            iqdata_one_file = np.reshape(iqdata_one_file[:(iqdata_one_file.shape[0]//slice_len)*slice_len], (iqdata_one_file.shape[0]//slice_len,slice_len)) # discard the extra elements for uneven array
            # print("After: ", iqdata_one_file.shape)
            iqdata_one_file = np.expand_dims(iqdata_one_file,axis=2)
            iqdata_one_file = np.concatenate([iqdata_one_file['raw-iq'].real,
                                         iqdata_one_file['raw-iq'].imag], axis=2)
            one_file_label = np.zeros((iqdata_one_file.shape[0], num_classes))

            # Creating the labels: 2 Label version
            if num_classes == 2:

                flag = extract_IQ_metadata_from_NWRA(filename, dsss_type)
                if 'SKIP' in flag:
                    continue
                elif 'LTE_DSSS' in flag:
                    one_file_label[:, 1] = 1
                    lte_dsss_count = lte_dsss_count + 1
                else:
                    one_file_label[:, 0] = 1
                    only_lte_count = only_lte_count + 1

                if count == 0:
                    inputs = iqdata_one_file
                    labels = one_file_label
                    # print("coming here too..", labels.shape)
                else:
                    inputs = np.concatenate((inputs, iqdata_one_file), axis=0)
                    labels = np.concatenate((labels, one_file_label), axis=0)
            print("printing the shapes for ", filename, " is: ", iqdata_one_file.shape, one_file_label.shape)
            # if iqdata_one_file.shape[0]!= one_file_label.shape[0]:
            #         print("!!!!MISMATCH!!!!")
            count = count + 1

    # display DataFrame
    print("Input shape: ", inputs.shape)
    print("Label shape: ", labels.shape)
    print("TOTAL READ FILES: ", count)
    print("TOTAL LTE FILES: ", only_lte_count)
    print("TOTAL LTE and DSSS FILES: ", lte_dsss_count)
    return inputs, labels

def read_processed_feat(feature_path, feature_type = 'nc', feature_options=[0], blocks = [131072], num_classes=2, snr_list = [0, 5, 10], sir_list=[0, 5, 10], strategy = 0):
    """
    This function is to load non-conjugate features in a specific session at a particular time.
    :param feature_path: str, the file path for the feature
    :param feature_type: str, the feature type: 'nc' or 'c'
    """
    inputs = np.zeros(1)
    labels =np.zeros(1)
    input_label_dic =  defaultdict(list)
    metadata_path = feature_path + 'IQDataSet_LTE_DSSS_v2/Metadata/'
    feature_path = feature_path + 'NEU_LTE_DSSS_Dataset_2_CSP/'
    # NWRA_data\131072 'test/CSP/'
    # feature_path = feature_path + 'NWRA_data/131072/'

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
            # print("meta file name: ", meta_filename)
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
                if strategy == 1:
                    one_file_np = organize_CSP_features_2D(one_file_np)
                if strategy == 2:
                    one_file_np = extract_statistic_from_CSP_features(one_file_np)
                    one_file_np = np.expand_dims(one_file_np, axis=0)
                if strategy == 3:
                    one_file_np = organize_CSP_features_3D(one_file_np)
                if strategy == 4:
                    one_file_np = extract_statistic_from_CSP_nc_features_one_col(one_file_np)
                    one_file_np = np.expand_dims(one_file_np, axis=0)
                if count == 0:
                    inputs = one_file_np
                    # print("coming here..", inputs.shape)
                else:
                    inputs = np.concatenate((inputs, one_file_np), axis=0)

                one_file_label = np.zeros((one_file_np.shape[0], num_classes))
                # if strategy == 1: one_file_label = np.zeros((1, num_classes))
                # Creating the labels: 2 Label version
                if num_classes == 2:
                    if 'LTE_DSSS' in filename:
                        # if strategy == 1: one_file_label[1] = 1
                        one_file_label[:, 1] = 1
                        lte_dsss_count = lte_dsss_count +1

                    else:
                        one_file_label[:, 0] = 1
                        only_lte_count = only_lte_count + 1
                    if count == 0:
                        labels = one_file_label
                        # print("coming here too..", labels.shape)
                    else:
                        labels = np.concatenate((labels, one_file_label), axis=0)

                # 7-label version
                else:
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

                # adding the input and labels to the dictionary
                input_label_dic['input'].append(one_file_np)
                input_label_dic['label'].append(one_file_label[0])


    # display DataFrame
    print("Input shape: ", inputs.shape)
    print("Label shape: ", labels.shape)
    print("TOTAL READ FILES: ", count)
    print("TOTAL LTE FILES: ", only_lte_count)
    print("TOTAL LTE and DSSS FILES: ", lte_dsss_count)
    print("***** THE SIZE OF THE DICTIONARY INPUT AND LABELS: ",  len(input_label_dic['input']), len(input_label_dic['label']))
    return inputs, labels, input_label_dic

def extract_CSP_metadata_from_NWRA(filename, dsss_type):
    loop_index = int(filename.split('_')[6]) # finding the loop index in the data
    # print("FLAG: ", filename, flag)
    # print("The loop index and dsss_type: ", loop_index, dsss_type)
    if loop_index%2 == 0: # loop index is even - only LTE
        # print("returning the loop index for lte: ", loop_index)
        return 'LTE'
    else: # loop index is odd  - DSSS present
        if dsss_type == 'synthetic' and loop_index > 40:
            # print("returning the loop index for dsss: ", loop_index)
            return 'SKIP'
        if dsss_type == 'real' and loop_index < 40:
            return 'SKIP'
        # print("returning the loop index for dsss: ", loop_index)
        return 'LTE_DSSS'


def read_processed_feat_real_data(feature_path, feature_type = 'nc', feature_options=[0], blocks = [131072], num_classes = 2, strategy = 0, dsss_type = 'all'):
    """
    This function is to load non-conjugate features in a specific session at a particular time.
    :param feature_path: str, the file path for the feature
    :param feature_type: str, the feature type: 'nc' or 'c'
    """
    inputs = np.zeros(1)
    labels =np.zeros(1)
    input_label_dic =  defaultdict(list)
    feature_path = feature_path + 'NWRA_data/'+str(blocks[0])+'/'

    count = 0
    only_lte_count = 0
    lte_dsss_count = 0

    for filename in os.listdir(feature_path):
        file_extension = '.NC'
        if feature_type == 'c':
            file_extension = '.C'
        if file_extension in filename and any(str(ext) in filename for ext in blocks):
                # block_number =
                # print("Reading...", filename)
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

                if strategy == 1:
                    one_file_np = organize_CSP_features_2D(one_file_np)
                if strategy == 2:
                    one_file_np = extract_statistic_from_CSP_features(one_file_np)
                    one_file_np = np.expand_dims(one_file_np, axis=0)
                if strategy == 3:
                    one_file_np = organize_CSP_features_3D(one_file_np)
                if strategy == 4:
                    one_file_np = extract_statistic_from_CSP_nc_features_one_col(one_file_np) # changed for comjugate feautures
                    one_file_np = np.expand_dims(one_file_np, axis=0)

                one_file_label = np.zeros((one_file_np.shape[0], num_classes))
                # Creating the labels: 2 Label version
                if num_classes == 2:
                    flag = extract_CSP_metadata_from_NWRA(filename, dsss_type)
                    if 'SKIP' in flag:
                        continue
                    elif 'LTE_DSSS' in flag:
                        one_file_label[:, 1] = 1
                        lte_dsss_count = lte_dsss_count +1
                    else:
                        one_file_label[:, 0] = 1
                        only_lte_count = only_lte_count + 1

                # gathering the inputs and labels
                if count == 0:
                    inputs = one_file_np
                        # print("coming here..", inputs.shape)
                else:
                    inputs = np.concatenate((inputs, one_file_np), axis=0)

                if count == 0:
                    labels = one_file_label
                        # print("coming here too..", labels.shape)
                else:
                    labels = np.concatenate((labels, one_file_label), axis=0)

                #total files
                count = count + 1
                # adding the input and labels to the dictionary
                input_label_dic['input'].append(one_file_np)
                input_label_dic['label'].append(one_file_label[0])


    # display DataFrame
    print("Input shape: ", inputs.shape)
    print("Label shape: ", labels.shape)
    print("TOTAL READ FILES: ", count)
    print("TOTAL LTE FILES: ", only_lte_count)
    print("TOTAL LTE and DSSS FILES: ", lte_dsss_count)
    print("***** THE SIZE OF THE DICTIONARY INPUT AND LABELS: ", len(input_label_dic['input']), len(input_label_dic['label']))
    return inputs, labels, input_label_dic

