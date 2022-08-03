
import torch
import torch.nn as nn
import argparse
import time
import numpy as np
import pandas as pd
import os

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

parser.add_argument('--data_folder', help='Location of the directory where all the conjugate or non-conjugate files are stored', type=str, default= 'D:/IARPA_DATA/LTE-recorded-at-AiRANACULUS/LTE/')  # D:/IARPA_DATA/NWRA_data/131072/
# parser.add_argument('--model_file', help='Location of the model file (with directory)', type=str, default= 'D:/IARPA_DATA/Saved_Models/block_wise_trained_model_on_sythetic_dataset_strategy5/non_conjugate_131072.pt')
parser.add_argument('--model_file', help='Location of the model file (with directory)', type=str, default= 'D:/IARPA_DATA/Saved_Models/NWRA_dataset_models/non_conjugate_262144.pt')
parser.add_argument('--bs',default=8, type=int,help='Batch size') # 32
parser.add_argument('--id_gpu', default=0, type=int, help='which gpu to use.')
parser.add_argument('--feature_options', nargs='*', default=[0, 1, 2, 3],choices = [0, 1, 2, 3],
help='Which features to use from the conjugate and non-conjugate files.')
parser.add_argument('--strategy',  type=int, default =4, choices = [0, 1, 2, 3, 4], help='Different strategies used for CSP feature processing: naive (0), 2D matrix (1), extract stat (2), 3D matrix (3), extract stat from one column (4).')
# Train and test on the data from Chad
parser.add_argument('--real_data', type=str2bool, help='Perform inference on real data from Chad.', default=False)
parser.add_argument('--air_data', type=str2bool, help='Perform inference on real data from Chad.', default=True)
parser.add_argument('--dsss_type', type=str, default='real', choices = ['all', 'real', 'synthetic'], help='Specify which type of DSSS signals you want to use for training and testing.')


args = parser.parse_args()

# CUDA for PyTorch - uncomment these part if you want to use gpu for inference
if args.id_gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)
torch.manual_seed(1234)



# extract the statistics about the CSP features
def extract_statistic_from_CSP_features(CSP_features_one_file):
    mean = np.max(CSP_features_one_file, axis=0)
    return mean

# extract the statistics about the CSP features
def extract_statistic_from_CSP_features_one_col(CSP_features_one_file):
    # print("coming here")
    row_with_max = np.argmax(CSP_features_one_file[:,1], axis=0)
    # print("important information..", row_with_max, CSP_features_one_file[row_with_max, :], CSP_features_one_file)
    return CSP_features_one_file[row_with_max, :]

# def extract_metadata_from_chad (filename):
#     flag = filename.split('_')[6]
#     # print("FLAG: ", filename, flag)
#     if int(flag)%2 == 0:
#         return 'LTE'
#     else:
#         return 'LTE_DSSS'

def extract_metadata_from_chad(filename, dsss_type):
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



# EVALUATING THE MODEL
model = torch.load(args.model_file)

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


for filename in os.listdir(args.data_folder):

    if '.NC' in filename:
        one_file = read_one_CSP_feature(args.data_folder+filename, feature_options=args.feature_options)
        correct = 0
        start_time = time.time()
        # one_file = extract_statistic_from_CSP_features(one_file)
        if args.strategy == 1:
            one_file = organize_CSP_features_2D(one_file)
            one_file = np.expand_dims(one_file, axis=0)
        if args.strategy == 2:
            one_file = extract_statistic_from_CSP_features(one_file)
        if args.strategy == 4:
            one_file = extract_statistic_from_CSP_features_one_col(one_file)
        # if strategy == 3:
        #     one_file_np = organize_CSP_features_3D(one_file_np)
        #

        row_output = model(torch.from_numpy(one_file).float().to(device))
        # print("testing111..", row_output)
        row_output = torch.argmax(row_output.squeeze(), dim=0)
        # print("testing222..", row_output)
        row_output = row_output.squeeze().cpu().detach().numpy()
        # print("testing333..", row_output)
        correct = correct+row_output
        end_time = time.time()
        pred_label = 'LTE'
        if row_output == 1:pred_label = 'Combined_LTE_DSSS'
        print("The prediction from CSP features in ", filename, " is: ", pred_label)
        print("\n Total time of execution for", filename, " is : ", (end_time - start_time), " seconds.")
        if args.real_data == True:
            loop_index = int(filename.split('_')[6])
            if loop_index%2 == row_output: #loop index is even - only LTE
                if args.dsss_type == 'synthetic' and loop_index > 40:
                    # print("returning the loop index for dsss: ", loop_index)
                    continue
                if args.dsss_type == 'real' and loop_index < 40:
                    continue
                total_matched += 1

        else:
            if pred_label in filename:
                total_matched +=1
        total_files += 1
    else:
        # print("Skipping the conjugate features in ", filename)
        pass

############################################
print("This part is for sanity check with Vini's dataset.")
print("##################################\nTotal corretly predicred non-conjugate files are ", total_matched, " out of total ", total_files, " files, and the percentage is: ", (total_matched/total_files)*100, "%.")