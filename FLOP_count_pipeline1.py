import argparse
import math
import scipy
import numpy as np
import random
import os

parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--iq_slice_len',default=131072, type=int, choices = [131072, 262144, 524288], help='Slice length for processing IQ files')
# parser.add_argument('--data_folder', help='Location of the directory where all the conjugate or non-conjugate files are stored', type=str, default= 'D:/IARPA_DATA/Data_for_Inference_NWRA/')  # D:/IARPA_DATA/NWRA_data/131072/
parser.add_argument('--percentage_to_read', type=float, default=0.1, help='The percentage of data (from real dataset) you want to read, choose between [0-1].') # by default all DSSS
parser.add_argument('--dataset', type=str, default='nwra',choices = ['neu', 'powder', 'nwra'],
help='Which dataset to use.')

args = parser.parse_args()
print('Argument parser inputs', args)

N = 10 # Total number of processed data samples of length args.iq_slice_len


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



# calculate N

if args.dataset == 'neu': # neu dataset
    iq_folder_path = 'D:/IARPA_DATA/IQDataSet_LTE_DSSS_v2/IQ/'
elif args.dataset == 'nwra': # nwra dataset
    iq_folder_path = 'D:/IARPA_DATA/NWRA_data/IQ/'

else: # powder dataset
    iq_folder_path = 'D:/IARPA_DATA/POWDER_data/Batch1_10MHz/IQ/'




total_files = 0
total_cost = 0
for filename in os.listdir(iq_folder_path):
    if random.random() > args.percentage_to_read:
        # print("Continuing, ", filename)
        continue
    iqdata_one_file = read_one_iq(iq_folder_path, filename, args.iq_slice_len)
    #print("The shape of read dataset ", iqdata_one_file.shape)
    # N = (iqdata_one_file.shape[0]-1)*args.iq_slice_len
    N = iqdata_one_file.shape[0] * args.iq_slice_len
    cost = N* args.iq_slice_len * (math.log2(N) + math.log2(args.iq_slice_len) + 1)
    print("FLOP count for ", filename, " is: ", cost)
    total_cost += cost
    total_files += 1

print("The average required FLOP counts for block length ", str(args.iq_slice_len), " for processing one IQ file in the  ", args.dataset, " dataset is: ", total_cost/total_files)