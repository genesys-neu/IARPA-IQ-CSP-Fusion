####################################
## This code is not ready to be released -
# TODO: need to filter out LTE+DSSS signals for inference
#######################################

import torch
import torch.nn as nn
import argparse
import time
import numpy as np
import pandas as pd
import os
import scipy
import random
import glob

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




parser = argparse.ArgumentParser(description='Configure the files before training the net.')
# parser.add_argument('--data_folder', help='Location of the directory where all the conjugate or non-conjugate files are stored', type=str, default= 'D:/IARPA_DATA/NEU_LTE_DSSS_Dataset_2_CSP/')
# parser.add_argument('--model_file', help='Location of the model file (with directory)', type=str, default= 'D:/IARPA_DATA/Saved_Models/best/non_conjugate_all_SNR_all_SIR.pt')
parser.add_argument('--input', nargs='*', default=['iq'],choices = ['iq', 'c'],
help='Which data to use as input. Select from: raw IQ data, non-conjugate features, conjugate features.')
parser.add_argument('--iq_slice_len',default=524288, type=int,choices = [131072, 262144, 524288], help='Slice length for processing IQ files') # 32
parser.add_argument('--data_folder', help='Location of the directory where all the conjugate or non-conjugate files are stored', type=str, default= 'D:/IARPA_DATA/Data_for_Inference_NWRA/')  # D:/IARPA_DATA/NWRA_data/131072/
# parser.add_argument('--model_file', help='Location of the model file (with directory)', type=str, default= 'D:/IARPA_DATA/Saved_Models/block_wise_trained_model_on_sythetic_dataset_strategy5/non_conjugate_131072.pt')
parser.add_argument('--model_folder', help='Location of the model file (with directory)', type=str, default= 'D:/IARPA_DATA/Saved_Models_Variant/NWRA_dataset_models/') # for moludaiton recognition
parser.add_argument('--bs',default=8, type=int,help='Batch size') # 32
parser.add_argument('--id_gpu', default=0, type=int, help='which gpu to use.')
parser.add_argument('--feature_options', nargs='*', default=[0, 1, 2, 3],choices = [0, 1, 2, 3],
help='Which features to use from the conjugate and non-conjugate files.')
parser.add_argument('--strategy',  type=int, default =4, choices = [0, 1, 2, 3, 4], help='Different strategies used for CSP feature processing: naive (0), 2D matrix (1), extract stat (2), 3D matrix (3), extract stat from one column (4).')
# Train and test on the data from Chad
parser.add_argument('--real_data', type=str2bool, help='Perform inference on real data from Chad.', default=False)
parser.add_argument('--air_data', type=str2bool, help='Perform inference on real data from Chad.', default=True)
parser.add_argument('--dsss_type', type=str, default='real', choices = ['all', 'real', 'synthetic'], help='Specify which type of DSSS signals you want to use for training and testing.')
# parser.add_argument('--anom_detection', type=bool, default=False)

args = parser.parse_args()

# CUDA for PyTorch - uncomment these part if you want to use gpu for inference
if args.id_gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)
torch.manual_seed(1234)


fusion = False
if len(args.input) >1: fusion = True


if fusion == False and 'iq' in args.input:
    model_file = args.model_folder + 'Variant_model_iq_'+str(args.iq_slice_len)+'_penultimate.pt'
elif fusion == False and 'c' in args.input:
    model_file = args.model_folder + 'Variant_model_c_'+str(args.iq_slice_len)+'_penultimate.pt'
else:
    model_file = args.model_folder + 'Variant_model_conjugate_iq_'+str(args.iq_slice_len)+'_penultimate.pt'

# extract the statistics about the CSP features
def extract_statistic_from_CSP_features(CSP_features_one_file):
    mean = np.max(CSP_features_one_file, axis=0)
    return mean


# extract the statistics about the CSP features
def extract_statistic_from_CSP_c_features_one_col(CSP_features_one_file):
    # print("coming here")
    row_with_max = np.argmax(CSP_features_one_file[:,3], axis=0) # for non-conjugate features 2nd column, for conjugate features 4th column is important
    # print("important information..", row_with_max, CSP_features_one_file[row_with_max, :], CSP_features_one_file)
    return CSP_features_one_file[row_with_max, :]


def extract_metadata_dsss_variant_from_NWRA_dataset(foldername, filename, no_of_blocks):
    synthetic_dsss_metadata_file_name = foldername+ 'dsss_signal_params.txt'
    captured_dsss_metadata_file_name = foldername + 'dsss_signal_params_2.txt'
    col1 = 'loop_index'
    col2 = 'signal_index'
    col3 = 'M_value'
    col4 = 'chip_rate'
    col5 = 'chipping_seq_index'
    col6 = 'CF_offset'
    col7 = 'EBW'
    col8 = 'power_slicing_factor'
    col9 = 'SIR_db'
    col10 = 'delay'
    col11 = 'upsample'
    col12 = 'downsample'
    col13 = 'noise_spectral_density'
    col14 = 'mod_type' #  1 == BPSK, 2 == QPSK, 3 == SQPSK.
    synthetic_dsss_metadata_df = pd.read_csv(synthetic_dsss_metadata_file_name, sep=" ", header=None,
                     names=[col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14])
    captured_dsss_metadata_df = pd.read_csv(captured_dsss_metadata_file_name, sep=" ", header=None,
                     names=[col1, col2, 'start_sample', col6, col8, col9, col10, col11, col12, col13])

    one_file_label = np.zeros((no_of_blocks, 2))

    loop_index = int(filename.split('_')[6])
    signal_index = int(filename.split('_')[7].split('.')[0])
    print("*******LOOP INDEX AND SIGNAL INDEX: ***********: ", loop_index, signal_index)
    if loop_index % 2 != 0: # DSSS signal, now extract metadata information
        if loop_index > 40: # real
            #for index, row in captured_dsss_metadata_df.iterrows():
                #print("The row: ", row)
                #if loop_index == row['loop_index'] and signal_index == row['signal_index']:
            one_file_label[:, 0] = 1 # all real DSSS are BPSK only
        else: # synthetic
             for index, row in synthetic_dsss_metadata_df.iterrows():
                # print("The row: ", row)
                if loop_index == row['loop_index'] and signal_index == row['signal_index']:
                    if row['mod_type'] == 1:
                        one_file_label[:, 0] = 1
                    else:
                        one_file_label[:, 1] = 1  # two options: BPSK and QPSK

    return one_file_label

# generate one CSP feature in form of 2D matrix
def organize_CSP_features_2D(CSP_features_one_file):
    channels = CSP_features_one_file.shape[1]
    rows_to_be_taken = 100
    rows_to_be_added = 0
    columnIndex = 1 # sorting by the second column
    CSP_features_one_file = CSP_features_one_file[(-CSP_features_one_file[:, columnIndex]).argsort()] # sort the array in decending order by fourth column
    if CSP_features_one_file.shape[0] <rows_to_be_taken:
        rows_to_be_added = rows_to_be_taken - CSP_features_one_file.shape[0]
    else:
        CSP_features_one_file = CSP_features_one_file[0:rows_to_be_taken, :]

    # print("shape 1: ", CSP_features_one_file.shape)
    CSP_features_one_file = np.pad(CSP_features_one_file, [(0, rows_to_be_added), (0, 0)], mode='constant', constant_values=0)
    # print("******testing*******: ", CSP_features_one_file.shape)

    return CSP_features_one_file

#read one CSP (either conjugate/non-conjugate) feature file
def read_one_CSP_feature(filename, feature_options):
    try:
        one_file_df = pd.read_csv(filename, sep=" ", header=None).dropna(axis=1, how='all')
        column_index = list(one_file_df.columns)
        selected_columns = [column_index[i] for i in feature_options]
        # print("one_df and input shape:", one_file_df[selected_columns].to_numpy().shape, inputs.shape)
        one_file_np = one_file_df[selected_columns].to_numpy()

    except pd.errors.EmptyDataError:
        one_file_np = np.zeros((1, len(feature_options)))
    return one_file_np


# read one IQ file
def read_one_iq(iq_path, filename, block_length, slicing = False, slice_length = 256):
    dtype_all = scipy.dtype([('raw-iq', scipy.complex64)])
    with open(iq_path + filename, mode='rb') as file:  # b is important -> binary
        iqdata_one_file = scipy.fromfile(file, dtype=dtype_all)
        iqdata_one_file = np.reshape(iqdata_one_file[:(iqdata_one_file.shape[0] // block_length) * block_length], (iqdata_one_file.shape[0] // block_length, block_length))  # discard the extra elements for uneven array
        # print("Shape before: ", iqdata_one_file.shape)
        if slicing == True:
            slice_index = random.randint(0, iqdata_one_file.shape[1] - slice_length - 1)
            iqdata_one_file = iqdata_one_file[:, slice_index:slice_index + slice_length]


    # print("After: ", iqdata_one_file.shape)
    iqdata_one_file = np.expand_dims(iqdata_one_file, axis=2)
    iqdata_one_file = np.concatenate([iqdata_one_file['raw-iq'].real,
                                      iqdata_one_file['raw-iq'].imag], axis=2)
    return iqdata_one_file


# EVALUATING THE MODEL
print("The model file: ", model_file)
model = torch.load(model_file, map_location=torch.device('cpu'))

# CUDA for PyTorch - uncomment these part if you want to use gpu for inference
use_cuda = torch.cuda.is_available()
if use_cuda and args.id_gpu >= 0:
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    model.cuda()
else:
    device = torch.device("cpu")

# comment this part if you want to use gpu for inference
# device = torch.device("cpu")

total_files = 0
total_matched = 0

print(model)

elapsed_time = []


if fusion == False and 'c' in args.input:
    for filename in os.listdir(args.data_folder+str(args.iq_slice_len)+'/'):
        one_file = read_one_CSP_feature(args.data_folder+str(args.iq_slice_len)+'/'+filename, feature_options=args.feature_options)
        correct = 0
        start_time = time.time()
        # one_file = extract_statistic_from_CSP_features(one_file)
        if args.strategy == 1:
            one_file = organize_CSP_features_2D(one_file)
            one_file = np.expand_dims(one_file, axis=0)
        if args.strategy == 2:
            one_file = extract_statistic_from_CSP_features(one_file)
        if args.strategy == 4:
            one_file = extract_statistic_from_CSP_c_features_one_col(one_file)
        # if strategy == 3:
        #     one_file_np = organize_CSP_features_3D(one_file_np)
        #

        row_output, _ = model(torch.from_numpy(one_file).float().to(device))
        #print("testing111..", row_output)
        row_output = torch.argmax(row_output.squeeze(), dim=0)
        #print("testing222..", row_output)
        row_output = row_output.squeeze().cpu().detach().numpy()
        #print("testing333..", row_output)
        correct = correct+row_output
        end_time = time.time()
        pred_label = 'BPSK'
        if row_output == 1:pred_label = 'QPSK'
        print("The prediction from CSP features in ", filename, " is: ", pred_label)
        total_files +=1
        elapsed_time.append((end_time - start_time))
        print("\n Total time of execution for", filename, " is : ", (end_time - start_time), " seconds.")
elif fusion == False and 'iq' in args.input:
    for filename in os.listdir(args.data_folder+'IQ/'):
        iqdata_one_file = read_one_iq(args.data_folder+'IQ/', filename, args.iq_slice_len)
        start_time = time.time()
        pred_list = []
        # print("Shape of one file", iqdata_one_file.shape)
        iq_sample_counter = 0
        for one_sample in iqdata_one_file:
            # print("Shape of one sample", one_sample.shape)
            one_sample = np.expand_dims(one_sample, axis=0)
            row_output, _ = model(torch.from_numpy(one_sample).float().to(device))
            # print("testing111..", row_output)
            row_output = torch.argmax(row_output.squeeze(), dim=0)
            # print("testing222..", row_output)
            row_output = row_output.squeeze().cpu().detach().numpy()
            # print("Row output in the loop", row_output)
            pred_list.append(row_output)
            iq_sample_counter += 1
        # print("Pred list: ", pred_list, sum(pred_list), len(pred_list))
        row_output = sum(pred_list)//len(pred_list) # calculate which label is being predicted maximum time
        # print("Row output", row_output)
        end_time = time.time()
        pred_label = 'BPSK'
        if row_output == 1: pred_label = 'QPSK'
        print("The prediction from IQ Samples in ", filename, " is: ", pred_label)
        total_files += 1
        elapsed_time.append((end_time - start_time))
        print("\n Total time of execution for", filename, " is : ", (end_time - start_time), " seconds.")
elif fusion == True:
    for filename in os.listdir(args.data_folder + 'IQ/'):

        one_iq_file = read_one_iq(args.data_folder + 'IQ/', filename, args.iq_slice_len)
        print("shape of one iq file: ", one_iq_file.shape)
        pred_list = []
        iq_sample_counter = 0
        start_time = time.time()
        filename = filename.split('.t')[0]
        print("The CSP file name: ", args.data_folder+str(args.iq_slice_len)+"/" + filename + '_' + str(args.iq_slice_len) + '*.C')
        for file_name in glob.iglob(args.data_folder+str(args.iq_slice_len)+"/" + filename + '_' + str(args.iq_slice_len) + '*.C', recursive=True):
            #print("Coming here")
            one_csp_file = read_one_CSP_feature( file_name, feature_options=args.feature_options)
            one_csp_file = extract_statistic_from_CSP_c_features_one_col(one_csp_file)
            #print("The shape of IQ files in loop: ", one_iq_file.shape)
            one_iq_sample = np.expand_dims(one_iq_file[iq_sample_counter, :, :], axis=0)
            one_csp_file = np.expand_dims(one_csp_file, axis=0)
            #print("Shape of one iq flie in loop:", one_iq_sample.shape, one_csp_file.shape)
            row_output, _ = model(torch.from_numpy(one_iq_sample).float().to(device), torch.from_numpy(one_csp_file).float().to(device))
            row_output = torch.argmax(row_output.squeeze(), dim=0)
            row_output = row_output.squeeze().cpu().detach().numpy()
            print("The predicted value: ", row_output)
            pred_list.append(row_output)
            iq_sample_counter +=1
        end_time = time.time()
        row_output = sum(pred_list) // len(pred_list)
        pred_label = 'BPSK'
        if row_output == 1: pred_label = 'QPSK'
        print("The prediction from IQ Samples and CSP feature fusion in ", filename, " is: ", pred_label)
        total_files += 1
        elapsed_time.append((end_time - start_time))
        print("\n Total time of execution for", filename, " is : ", (end_time - start_time), " seconds.")
else:
        # print("Skipping the conjugate features in ", filename)
        pass

############################################
# print("This part is for sanity check with Vini's dataset.")
# print("##################################\nTotal corretly predicred non-conjugate files are ", total_matched, " out of total ", total_files, " files, and the percentage is: ", (total_matched/total_files)*100, "%.")
print("The mean elapsed time: ", sum(elapsed_time)/len(elapsed_time), "seconds")