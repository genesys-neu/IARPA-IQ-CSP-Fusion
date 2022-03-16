
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


parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--data_folder', help='Location of the directory where all the conjugate or non-conjugate files are stored', type=str, default= 'test/CSP/')
parser.add_argument('--model_file', help='Location of the model file (with directory)', type=str, default= 'non_conjugate_all_SNR_all_SIR.pt')
parser.add_argument('--bs',default=8, type=int,help='Batch size') # 32
parser.add_argument('--id_gpu', default=0, type=int, help='which gpu to use.')

args = parser.parse_args()

# CUDA for PyTorch - uncomment these part if you want to use gpu for inference
# if args.id_gpu >= 0:
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     # The GPU id to use
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)
torch.manual_seed(1234)



# extract the statistics about the CSP features
def extract_statistic_from_CSP_features(CSP_features_one_file):
    mean = np.max(CSP_features_one_file, axis=0)
    return mean

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
# use_cuda = torch.cuda.is_available()
# if use_cuda:
#     device = torch.device("cuda:0")
#     torch.backends.cudnn.benchmark = True
#     model.cuda()
# else:
#     device = torch.device("cpu")

# comment this part if you want to use gpu for inference
device = torch.device("cpu")

total_files = 0
total_matched = 0


for filename in os.listdir(args.data_folder):

    if '.NC' in filename:
        one_file = read_one_CSP_feature(args.data_folder+filename, feature_options=[3])
        correct = 0
        start_time = time.time()
        one_file = extract_statistic_from_CSP_features(one_file)
        row_output = model(torch.from_numpy(one_file).float().to(device))
        row_output = torch.argmax(row_output, dim=0)
        row_output = row_output.squeeze().cpu().detach().numpy()
        correct = correct+row_output
        end_time = time.time()
        pred_label = 'OnlyLTE'
        if row_output == 1:pred_label = 'Combined_LTE_DSSS'
        print("The prediction from CSP features in ", filename, " is: ", pred_label)
        print("\n Total time of execution for", filename, " is : ", (end_time - start_time), " seconds.")
        total_files +=1
        if pred_label in filename:
            total_matched +=1
    else:
        print("Skipping the conjugate features in ", filename)

############################################
print("\n\n###############################################\nThis part is for sanity check with Vini's dataset.")
print("Total corretly predicred non-conjugate files are ", total_matched, " out of total ", total_files, " files.")