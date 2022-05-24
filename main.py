"""
@author: debashri
This is the main file to run the code to detect anomalous signal (e.g., DSSS) from baselines (e.g., LTE) using IQ data and features in the IARPA project.
"""

from __future__ import division
import torch

import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import random
import time

import pickle, os

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm
from sklearn.utils import shuffle

import read_file
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

test_acc_anomaly_detection = [] #lISTING ALL THE TEST ACCURACIES
test_acc_anomaly_type = []
test_acc_anomaly_mod= []
test_acc_anomaly_gain= []

############################
# Fix the seed
############################
seed = 0
os.environ['PYTHONHASHSEED']=str(seed)
np.random.seed(seed)
random.seed(seed)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_labels(ylabel):
    ylabel_hot = np.zeros((ylabel.shape[0], 5))
    count  = -1
    for each_label in ylabel:
        # print("each label: ", each_label)
        count = count + 1
        for vehicle in each_label:
            ylabel_hot[count, vehicle-1] = 1
    return ylabel_hot


# IQ DATA PATH: D:\IARPA_DATA\IQDataSet_LTE_DSSS_v2\IQ

# Passing different arguments
parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--data_folder', help='Location of the data directory', type=str, default= 'D:/IARPA_DATA/')
parser.add_argument('--input', nargs='*', default=['nc', 'iq'],choices = ['iq', 'c', 'nc'],
help='Which data to use as input. Select from: raw IQ data, non-conjugate features, conjugate features.')
parser.add_argument('--feature_options', nargs='*', default=[0, 1, 2, 3],choices = [0, 1, 2, 3],
help='Which features to use from the conjugate and non-conjugate files.')
parser.add_argument('--iq_slice_len',default=131072, type=int,help='Slice length for processing IQ files') # 32
parser.add_argument('--num_classes', default=2, type=int, help='Number of classes for classification.')
parser.add_argument('--strategy',  type=int, default =4, choices = [0, 1, 2, 3, 4], help='Different strategies used for CSP feature processing: naive (0), 2D matrix (1), extract stat (2), 3D matrix (3), extract stat specific max (4).')

#Arguments for train/test on 80/20 percentage of overall data randomly
parser.add_argument('--random_test', type=str2bool, help='Perform train/test on 80/20 of data.', default=True)
parser.add_argument('--random_test_blocks', nargs='*', default=[131072],choices = [131072, 262144, 524288],
help='The block lengths to use for random train/test.')

#Arguments for training and testing on seperate block lengths
parser.add_argument('--training_blocks', nargs='*', default=[131072],choices = [131072, 262144, 524288],
help='Which block length to use for training.') # use this argument only if you set the 'random_test' argument as False
parser.add_argument('--testing_blocks', nargs='*', default=[524288],choices = [131072, 262144, 524288],
help='Which block length to use for training.') # use this argument only if you set the 'random_test' argument as False

#Arguments for training and testing on data from all block length, but test on a specific block length (exclusive from training)
parser.add_argument('--optimal_test', type=str2bool, help='Perform training on data from all block length, but test on a specific block length (exclusive from training).', default=False)
parser.add_argument('--optimal_test_blocks', nargs='*', default=[131072],choices = [131072, 262144, 524288],
help='The block length of which 80/20 data is used for train/test. We use 100 percent data for training for rest of the two block lengths.')

# Train and test on specific SNR values
parser.add_argument('--random_snr_test', type=str2bool, help='Perform training on features from one SNR level and other.', default=True)
parser.add_argument('--random_snr', nargs='*', default=[0, 5, 10],choices = [0, 5, 10],
help='The SNR to use for random train/test.')
parser.add_argument('--training_snr', nargs='*', default=[5],choices = [0, 5, 10],
help='Which SNR to use for training.') # use this argument only if you set the 'random_snr_test' argument as False
parser.add_argument('--testing_snr', nargs='*', default=[0],choices = [0, 5, 10],
help='Which SNR to use for training.') # use this argument only if you set the 'random_snr_test' argument as False
parser.add_argument('--dsss_sir', nargs='*', default=[0, 5, 10],choices = [0, 5, 10],
help='Which SIRs of DSSS to use for train/test.') # this argument will be used always #0: less challenging, 10: more challenging


#Neural Network Parameters
parser.add_argument('--classifier', type=str, default='nn', help='Options: lr (for logistic regression), nv (naive bayes), svm (support vector machine), nn (neural network).')
parser.add_argument('--lr', default=0.0001, type=float,help='learning rate for Adam optimizer',)
parser.add_argument('--bs',default=8, type=int,help='Batch size') # 32 # 8
parser.add_argument('--epochs', default=50, type = int, help='Specify the epochs to train')
parser.add_argument('--normalize', help='normalize or not', type=str2bool, default =True)
parser.add_argument('--id_gpu', default=-1, type=int, help='which gpu to use.')
parser.add_argument('--model_folder', help='Location of the data directory', type=str, default= 'D:/IARPA_DATA/Saved_Models/')

# Train and test on the data from Chad
parser.add_argument('--real_data', type=str2bool, default=False, help='Perform training and testing on real LTE and all (synthetic+real) DSSS data from Chad.') # by default all DSSS
parser.add_argument('--percentage_to_read', type=float, default=0.1, help='The percentage of data (from real dataset) you want to read, choose between [0-1].') # by default all DSSS
parser.add_argument('--dsss_type', type=str, default='real', choices = ['all', 'real', 'synthetic'], help='Specify which type of DSSS signals you want to use for training and testing.')
parser.add_argument('--meta_learning', type=str2bool, default=False, help='Perform meta learning from trained model on synthetic data to real data.')
parser.add_argument('--no_of_layers_to_freeze',  type=int, default =1, choices = [0, 1, 2, 3, 4], help="Specify the number of layers you want to freeze from the starting of the model.")
parser.add_argument('--retrain_ratio',  type=float, default =0.1, choices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], help="Specify the percentage of retraining would be done for meta learning.")
parser.add_argument('--model_file', help='Location of the model file (with directory)', type=str, default= 'D:/IARPA_DATA/Saved_Models/block_wise_trained_model_on_sythetic_dataset_strategy5/non_conjugate_131072.pt')

# Not being used so far
parser.add_argument('--restore_models', type=str2bool, help='Load single modality trained weights', default=False)
parser.add_argument('--retrain', type=str2bool, help='Retrain the model on loaded weights', default=True)
parser.add_argument('--fusion_layer', type=str, default='ultimate', help='Assign the layer name where the fusion to be performed.')
parser.add_argument('--incremental_fusion', type=str2bool, help='Perform the incremental fusion or not. By default it will be aggregative fusion.', default=False)
parser.add_argument('--state_fusions', type=str2bool, help='Perform the state-of-the-art fusions.', default=True)
parser.add_argument('--fusion_techniques', nargs='*', default=['mi'],choices = ['mi', 'lr_tensor', 'concat'],
help='Specify if you want to use any of the state-of-the-art fusion techniques such as: multiplicative interactions, low rank tensor.')

args = parser.parse_args()
print('Argument parser inputs', args)

if args.id_gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)
torch.manual_seed(1234)
fusion = False
if len(args.input) >1: fusion = True
all_block_lengths = [131072, 262144, 524288]


#############################################################################################################
# GENERATING inputs and labels
############################################################################################################

# # Conjugate/Non-conjugate FEATURES
if len(args.input) >=2:
    # train/test on a specific block length for CSP features and train/test on random SNR
    if args.real_data == True:
        inputs_iq, inputs_c, inputs_nc, labels = read_file.generate_inputs_labels_real_data(args.data_folder, args.input, args.feature_options, args.iq_slice_len, args.num_classes, args.random_snr, args.dsss_sir, args.strategy, args.dsss_type, args.percentage_to_read)
    else:
        inputs_iq, inputs_c, inputs_nc, labels = read_file.generate_inputs_labels_synthetic_data(args.data_folder,
                                                                                                 args.input,
                                                                                                 args.feature_options,
                                                                                                 args.iq_slice_len,
                                                                                                 args.num_classes,
                                                                                                 args.random_snr,
                                                                                                 args.dsss_sir,
                                                                                                 args.strategy,
                                                                                                 args.percentage_to_read)
    print("Shaped of the inputs and labels: ", inputs_iq.shape, inputs_c.shape, inputs_nc.shape, labels.shape)
else: # reading each CSP features and IQ files seperately (old implementation)
    print("**********************************************************")
    if 'nc' in args.input:
        if args.real_data == True:
            inputs_nc, labels_nc, input_label_dic = read_file.read_processed_feat_real_data(args.data_folder, 'nc', args.feature_options,
                                                                 args.random_test_blocks, args.num_classes, args.strategy, args.dsss_type)
        else:
            if args.random_test == True: # 80/20 train/test on specific block lengths
                if args.random_snr_test == True:
                    inputs_nc, labels_nc, input_label_dic = read_file.read_processed_feat(args.data_folder, 'nc', args.feature_options, args.random_test_blocks, args.num_classes, args.random_snr, args.dsss_sir, args.strategy)
                    print("Shapes of the non-conjugate features and labels: ", inputs_nc.shape, labels_nc.shape)
                else: # if we train on one SNR and test on other snrs
                    xtrain_nc, ytrain_nc, input_label_train_dic = read_file.read_processed_feat(args.data_folder, 'nc', args.feature_options,
                                                                         args.random_test_blocks, args.num_classes,
                                                                         args.training_snr, args.dsss_sir, args.strategy)
                    xtest_nc, ytest_nc, input_label_test_dic = read_file.read_processed_feat(args.data_folder, 'nc',
                                                                                      args.feature_options,
                                                                                      args.random_test_blocks, args.num_classes,
                                                                                      args.testing_snr, args.dsss_sir, args.strategy)
                    print("Shapes of the non-conjugate features and labels: ", xtrain_nc.shape, ytrain_nc.shape, xtest_nc.shape, ytest_nc.shape)
            elif args.optimal_test==True: # test on 50% data of specific block length and train on rest of 50% of that bloack length, along with rest two block length
                rest_block_lengths = [x for x in all_block_lengths if x not in args.optimal_test_blocks]
                    # all_block_lengths - args.optimal_test_blocks
                # print(rest_block_lengths)
                if args.random_snr_test == True:
                    inputs_nc, labels_nc, input_label_dic = read_file.read_processed_feat(args.data_folder, 'nc', args.feature_options,
                                                                         rest_block_lengths, args.num_classes, args.random_snr, args.dsss_sir, args.strategy)
                    # print("Shapes*************: ", inputs_nc.shape)
                    input_test_nc, labels_test_nc, input_label_test_dic = read_file.read_processed_feat(args.data_folder, 'nc', args.feature_options,
                                                                       args.optimal_test_blocks, args.num_classes,  args.random_snr, args.dsss_sir, args.strategy)
                    xhalf_train_nc, xtest_nc, yhalf_train_nc, ytest_nc = train_test_split(input_test_nc, labels_test_nc, test_size=0.2, random_state=42)
                else: # if we train on one SNR and test on other snrs
                    inputs_nc, labels_nc, input_label_dic = read_file.read_processed_feat(args.data_folder, 'nc', args.feature_options,
                                                                         rest_block_lengths, args.num_classes, args.training_snr, args.dsss_sir, args.strategy)
                    xhalf_train_nc, yhalf_train_nc, input_label_train_dic = read_file.read_processed_feat(args.data_folder, 'nc', args.feature_options,
                                                                                  args.optimal_test_blocks, args.num_classes,
                                                                                  args.training_snr, args.dsss_sir, args.strategy) # training the
                    xtest_nc, ytest_nc, input_label_test_dic = read_file.read_processed_feat(args.data_folder, 'nc', args.feature_options,
                                                                         args.optimal_test_blocks, args.num_classes, args.testing_snr, args.dsss_sir, args.strategy)
                    # xhalf_train_nc, xtest_nc, yhalf_train_nc, ytest_nc = train_test_split(input_test_nc, labels_test_nc, test_size=0.5,
                    #                                                 random_state=42)
                # print("Shapes: ", inputs_nc.shape, xhalf_train_nc.shape )
                xtrain_nc = np.concatenate((inputs_nc, xhalf_train_nc), axis=0)
                ytrain_nc = np.concatenate((labels_nc, yhalf_train_nc), axis=0)
                print("Shapes of the non-conjugate train, test features and labels: ", xtrain_nc.shape, xtest_nc.shape,
                      ytrain_nc.shape, ytest_nc.shape)
                # pass
            else: # train on specific block lengths, test on specific block lengths
                if args.random_snr_test == True:
                    training_snrs = args.random_snr
                    testing_snrs = args.random_snr
                else:
                    training_snrs = args.training_snr
                    testing_snrs = args.testing_snr
                xtrain_nc, ytrain_nc, input_label_train_dic = read_file.read_processed_feat(args.data_folder, 'nc', args.feature_options,args.training_blocks, args.num_classes, training_snrs, args.dsss_sir, args.strategy)
                xtest_nc, ytest_nc, input_label_test_dic = read_file.read_processed_feat(args.data_folder, 'nc', args.feature_options,args.testing_blocks, args.num_classes,  testing_snrs, args.dsss_sir, args.strategy)

                print("Shapes of the non-conjugate train, test features and labels: ", xtrain_nc.shape, xtest_nc.shape, ytrain_nc.shape, ytest_nc.shape)
        saved_file_name = 'non_conjugate'


    if 'c' in args.input:
        if args.real_data == True:
            inputs_c, labels_c, input_label_dic = read_file.read_processed_feat_real_data(args.data_folder, 'c', args.feature_options,
                                                                 args.random_test_blocks, args.num_classes, args.strategy, args.dsss_type)
        else:
            if args.random_test == True:  # 80/20 train/test on specific block lengths
                if args.random_snr_test == True:
                    inputs_c, labels_c, input_label_dic = read_file.read_processed_feat(args.data_folder, 'c', args.feature_options,
                                                                         args.random_test_blocks, args.num_classes,
                                                                         args.random_snr, args.dsss_sir, args.strategy)
                    print("Shapes of the conjugate features and labels: ", inputs_c.shape, labels_c.shape)
                else:  # if we train on one SNR and test on other snrs
                    xtrain_c, ytrain_c, input_label_train_dic = read_file.read_processed_feat(args.data_folder, 'c',
                                                                                     args.feature_options,
                                                                                     args.random_test_blocks, args.num_classes,
                                                                                     args.training_snr, args.dsss_sir, args.strategy)
                    xtest_c, ytest_c, input_label_test_dic = read_file.read_processed_feat(args.data_folder, 'c',
                                                                                   args.feature_options,
                                                                                   args.random_test_blocks, args.num_classes,
                                                                                   args.testing_snr, args.dsss_sir, args.strategy)
                    print("Shapes of the conjugate features and labels: ", xtrain_c.shape, ytrain_c.shape,
                          xtest_c.shape, ytest_c.shape)
            elif args.optimal_test == True:  # test on 50% data of specific block length and train on rest of 50% of that bloack length, along with rest two block length
                rest_block_lengths = [x for x in all_block_lengths if x not in args.optimal_test_blocks]
                # all_block_lengths - args.optimal_test_blocks
                # print(rest_block_lengths)
                if args.random_snr_test == True:
                    inputs_c, labels_c, input_label_dic = read_file.read_processed_feat(args.data_folder, 'c', args.feature_options,
                                                                         rest_block_lengths, args.num_classes, args.random_snr, args.dsss_sir, args.strategy)
                    input_test_c, labels_test_c, input_label_test_dic = read_file.read_processed_feat(args.data_folder, 'c', args.feature_options,
                                                                                  args.optimal_test_blocks, args.num_classes,
                                                                                  args.random_snr, args.dsss_sir, args.strategy)
                    xhalf_train_c, xtest_c, yhalf_train_c, ytest_c = train_test_split(input_test_c, labels_test_c,
                                                                                          test_size=0.2, random_state=42)
                else:
                    inputs_c, labels_c, input_label_dic = read_file.read_processed_feat(args.data_folder, 'c', args.feature_options,
                                                                         rest_block_lengths, args.num_classes,
                                                                         args.training_snr, args.dsss_sir, args.strategy)
                    xhalf_train_c, yhalf_train_c, input_label_train_dic = read_file.read_processed_feat(args.data_folder, 'c', args.feature_options,
                                                                                   args.optimal_test_blocks, args.num_classes,
                                                                                   args.training_snr, args.dsss_sir, args.strategy) # training the
                    xtest_c, ytest_c = read_file.read_processed_feat(args.data_folder, 'c', args.feature_options,
                                                                       args.optimal_test_blocks, args.num_classes,
                                                                       args.testing_snr, args.dsss_sir, args.strategy)
                xtrain_c = np.concatenate((inputs_c, xhalf_train_c), axis=0)
                ytrain_c = np.concatenate((labels_c, yhalf_train_c), axis=0)
                print("Shapes of the conjugate train, test features and labels: ", xtrain_c.shape, xtest_c.shape,
                      ytrain_c.shape, ytest_c.shape)
                # pass
            else:  # train on specific block lengths, test on specific block lengths
                if args.random_snr_test == True:
                    training_snrs = args.random_snr
                    testing_snrs = args.random_snr
                else:
                    training_snrs = args.training_snr
                    testing_snrs = args.testing_snr
                xtrain_c, ytrain_c, input_label_train_dic = read_file.read_processed_feat(args.data_folder, 'c', args.feature_options,
                                                                     args.training_blocks, args.num_classes, training_snrs, args.dsss_sir, args.strategy)
                xtest_c, ytest_c, input_label_test_dic = read_file.read_processed_feat(args.data_folder, 'c', args.feature_options,
                                                                   args.testing_blocks, args.num_classes, testing_snrs, args.dsss_sir, args.strategy)

                print("Shapes of the conjugate train, test features and labels: ", xtrain_c.shape, xtest_c.shape,
                      ytrain_c.shape, ytest_c.shape)
        saved_file_name = 'conjugate'

    # reading IQ features
    if 'iq' in args.input:
        if args.real_data == True:
            inputs_iq, labels_iq = read_file.read_real_iq_files(args.data_folder, args.random_test_blocks, args.iq_slice_len, args.num_classes, args.random_snr, args.dsss_sir, args.dsss_type, args.percentage_to_read)
            print("Shapes of the IQ files and labels: ", inputs_iq.shape, labels_iq.shape)
        else:
            if args.random_snr_test == True:
                inputs_iq, labels_iq = read_file.read_iq_files(args.data_folder, args.random_test_blocks, args.iq_slice_len, args.num_classes, args.random_snr, args.dsss_sir, args.percentage_to_read)
                print("Shapes of the IQ files and labels: ", inputs_iq.shape, labels_iq.shape)
            else:  # if we train on one SNR and test on other snrs
                xtrain_iq, ytrain_iq = read_file.read_iq_files(args.data_folder, args.random_test_blocks, args.iq_slice_len, args.num_classes, args.training_snrs, args.dsss_sir, args.percentage_to_read)
                xtest_iq, ytest_iq = read_file.read_iq_files(args.data_folder, args.random_test_blocks, args.iq_slice_len, args.num_classes, args.testing_snrs, args.dsss_sir, args.percentage_to_read)
                print("Shapes of the IQ files and labels: ", xtrain_iq.shape, ytrain_iq.shape,
                          xtest_iq.shape, ytest_iq.shape)

        saved_file_name = 'IQ'
##############################################################################
# Model configuration
##############################################################################

start_time = time.time()
fusion = False
if len(args.input) >1: fusion = True
# num_classes = args.num_classes # DETECT WHETHER DSSS IS PRESENT IN THE LTE SIGNAL OR NOT


################################################################
# CONVERTING NUMPY ARRAY TO TORCH #
################################################################
# Implementing in pytorch
import torch
from ModelHandler import NonConjugateNet, ConjugateNet, FeatureNet, AlexNet1D, FeatureFusion, RFNet, CSPNet, FeatureFusionThree, FeatureNetCNN



#############################################################################################################################
# WORKING ON EACH INDIVIDUAL MODALITIES
#############################################################################################################################
def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

def input_label_for_strategy2(input_label_dic):
    inputs = input_label_dic['input']
    labels = input_label_dic['label']
    return inputs, labels

def train_test_input_label_for_strategy2(input_label_train_dic, input_label_test_dic):
    xtrain = input_label_train_dic['input']
    ytrain = input_label_train_dic['label']
    xtest = input_label_test_dic['input']
    ytest = input_label_test_dic['label']
    return xtrain, xtest, ytrain, ytest

if fusion == False:
    if 'nc' in args.input:
        # if args.strategy ==1:
        #     inputs = input_label_dic['input']
        #     labels = input_label_dic['label']
        #     xtrain, xtest, ytrain, ytest = train_test_split(inputs, labels, test_size=0.2,
        #                                                     random_state=42)  # 80/20 is train/test size

        if args.real_data == True:
            inputs = inputs_nc
            labels = labels_nc
            if args.strategy == 1: inputs, labels = input_label_for_strategy2(input_label_dic) # this will be triggered only for strategy 2 for CSP features
            if args.meta_learning == True:
                xtrain, xtest, ytrain, ytest = train_test_split(inputs, labels, test_size=(1-args.retrain_ratio), random_state=42)  # 80/20 is train/test size
            else:
                xtrain, xtest, ytrain, ytest = train_test_split(inputs, labels, test_size=0.2, random_state=42)  # 80/20 is train/test size
        else:
            if args.random_test == True: # This will be triggered when doing train/test split randomly on whole data
                if args.random_snr_test == True:
                    inputs = inputs_nc
                    labels = labels_nc
                    if args.strategy == 1: inputs, labels = input_label_for_strategy2(input_label_dic)  # this will be triggered only for strategy 2 for CSP features
                    xtrain, xtest, ytrain, ytest = train_test_split(inputs, labels, test_size=0.2, random_state=42)  # 80/20 is train/test size
                else:
                    xtrain = xtrain_nc
                    ytrain = ytrain_nc
                    xtest = xtest_nc
                    ytest = ytest_nc
                    if args.strategy == 1: xtrain, xtest, ytrain, ytest = train_test_input_label_for_strategy2(input_label_train_dic, input_label_test_dic)
            else: # This will be triggered when doing train/test on separate blocks and optimal testing
                xtrain = xtrain_nc
                ytrain = ytrain_nc
                xtest = xtest_nc
                ytest = ytest_nc
                if args.strategy == 1: xtrain, xtest, ytrain, ytest = train_test_input_label_for_strategy2(input_label_train_dic, input_label_test_dic)

                # xtrain = np.swapaxes(xtrain, 1, 2)
            # ytrain = np.swapaxes(ytrain, 1, 2)
            # xtest = np.swapaxes(xtest, 1, 2)
            # ytest = np.swapaxes(ytest, 1, 2)
        if args.strategy == 1: model = FeatureNetCNN(input_dim=xtrain[0].shape[0], output_dim=args.num_classes)
        else: model = FeatureNet(input_dim=xtrain.shape[1], output_dim=args.num_classes)
        if args.meta_learning == True:
            print("coming here")
            model = torch.load(args.model_file)
            print("before ", model)
            # model = torch.nn.Sequential(*(list(mA.children())[:-1]))
            # layers_to_freeze = 1
            count = 0
            for c in model.children():
                # print(c, layers_to_freeze)
                if count <= args.no_of_layers_to_freeze:
                    for param in c.parameters():
                        # print(param)
                        param.requires_grad = False
                count += 1
            # print("after:", model)



    if 'c' in args.input:
        if args.real_data == True:
            inputs = inputs_c
            labels = labels_c
            xtrain, xtest, ytrain, ytest = train_test_split(inputs, labels, test_size=0.2,
                                                            random_state=42)  # 80/20 is train/test size
        else:
            if args.random_test == True: # This will be triggered when doing train/test split randomly on whole data
                if args.random_snr_test == True:
                    inputs = inputs_c
                    labels = labels_c
                    xtrain, xtest, ytrain, ytest = train_test_split(inputs, labels, test_size=0.2,
                                                                    random_state=42)  # 80/20 is train/test size
                else:
                    xtrain = xtrain_c
                    ytrain = ytrain_c
                    xtest = xtest_c
                    ytest = ytest_c
            else: # This will be triggered when doing train/test on separate blocks and optimal testing
                xtrain = xtrain_c
                ytrain = ytrain_c
                xtest = xtest_c
                ytest = ytest_c
        model = FeatureNet(input_dim=xtrain.shape[1], output_dim=args.num_classes)
        # model = NonConjugateNet(input_dim=xtrain.shape[1], output_dim=args.num_classes)

    if 'iq' in args.input:
        # This will be triggered when doing train/test split randomly on whole data
        if args.random_snr_test == True:
            inputs = inputs_iq
            labels = labels_iq
            xtrain, xtest, ytrain, ytest = train_test_split(inputs, labels, test_size=0.2,
                                                                random_state=42)  # 80/20 is train/test size
        else:
            xtrain = xtrain_iq
            ytrain = ytrain_iq
            xtest = xtest_iq
            ytest = ytest_iq
        model = AlexNet1D(input_dim=xtrain.shape[2], output_dim=args.num_classes)
        # model = RFNet(input_dim=xtrain.shape[2], output_dim=args.num_classes)

        #################### NORMALIZE THE X DATA #######################
        if args.normalize == True:
            xtrain_all = np.concatenate((xtrain, xtest))
              # Normalize the data with zero mean and unit variance for each column
            for i in range(xtrain_all.shape[1]):
                standard_iq = preprocessing.StandardScaler().fit(xtrain_all[:, i, :])
                xtrain[:, i, :] = standard_iq.transform(xtrain[:, i, :])
                xtest[:, i, :] = standard_iq.transform(xtest[:, i, :])
        ############### END OF NORMAIZATION ################


    # INITIALIZING THE WEIGHT AND BIAS
    model.apply(weights_init)
else: # incase of fusion
    # considering both conjugate and non-conjugate CSP features
    if 'c' in args.input and 'nc' in args.input and 'iq' not in args.input:
        if args.random_test == True or args.real_data == True: # This will be triggered when doing train/test split randomly on whole data
            inputs = np.concatenate((inputs_nc, inputs_c), axis=0)
            labels = np.concatenate((labels_nc, labels_c), axis=0)
            xtrain, xtest, ytrain, ytest = train_test_split(inputs, labels, test_size=0.2,
                                                            random_state=42)  # 80/20 is train/test size
        else: # This will be triggered when doing train/test on separate blocks and optimal testing
            xtrain = np.concatenate((xtrain_nc, xtrain_c), axis=0)
            ytrain = np.concatenate((ytrain_nc, ytrain_c), axis=0)
            xtest = np.concatenate((xtest_nc, xtest_c), axis=0)
            ytest = np.concatenate((ytest_nc, ytest_c), axis=0)

        model = FeatureNet(input_dim=xtrain.shape[1], output_dim=args.num_classes)
        # model = NonConjugateNet(input_dim=xtrain.shape[1], output_dim=args.num_classes)
        saved_file_name = 'conjugate_non_conjugate'

    # considering IQ, conjugate CSP features
    if 'iq' in args.input and 'c' in args.input and 'nc' not in args.input:
        xtrain_iq, xtest_iq, xtrain_c, xtest_c, ytrain, ytest = train_test_split(inputs_iq, inputs_c, labels, test_size=0.2, random_state=42)
        modelA = AlexNet1D(input_dim=xtrain_iq.shape[2], output_dim=args.num_classes, fusion=args.fusion_layer)
        # modelB = CSPNet(input_dim=xtrain_c.shape[1])
        modelB = FeatureNet(input_dim=xtrain_c.shape[1], output_dim=args.num_classes)
        model = FeatureFusion(modelA, modelB, nb_classes=args.num_classes, fusion=args.fusion_layer)
        saved_file_name = 'conjugate_iq'

        #################### NORMALIZE THE X DATA #######################
        if args.normalize == True:
            xtrain_iq_all = np.concatenate((xtrain_iq, xtest_iq))
              # Normalize the data with zero mean and unit variance for each column
            for i in range(xtrain_iq_all.shape[1]):
                standard_iq = preprocessing.StandardScaler().fit(xtrain_iq_all[:, i, :])
                xtrain_iq[:, i, :] = standard_iq.transform(xtrain_iq[:, i, :])
                xtest_iq[:, i, :] = standard_iq.transform(xtest_iq[:, i, :])

            xtrain_c_all = np.concatenate((xtrain_c, xtest_c))
            standard_c = preprocessing.StandardScaler().fit(xtrain_c_all)  # Normalize the data with zero mean and unit variance for each column
            xtrain_c = standard_c.transform(xtrain_c)
            xtest_c = standard_c.transform(xtest_c)

        ############### END OF NORMAIZATION ################

        print("XTRAIN (IQ, C) AND XTEST SHAPE:", xtrain_iq.shape, xtest_iq.shape, xtrain_c.shape, xtest_c.shape)
        print("YTRAIN AND YTEST SHAPE:", ytrain.shape, ytest.shape)

    # considering IQ, non-conjugate CSP features
    if 'iq' in args.input and 'nc' in args.input and 'c' not in args.input:
        xtrain_iq, xtest_iq, xtrain_nc, xtest_nc, ytrain, ytest = train_test_split(
                inputs_iq, inputs_nc, labels, test_size=0.2, random_state=42)
        modelA = AlexNet1D(input_dim=xtrain_iq.shape[2], output_dim=args.num_classes, fusion=args.fusion_layer)
        # modelB = CSPNet(input_dim=xtrain_nc.shape[1])
        modelB = FeatureNet(input_dim=xtrain_nc.shape[1], output_dim=args.num_classes)
        model = FeatureFusion(modelA, modelB, nb_classes=args.num_classes, fusion=args.fusion_layer)
        saved_file_name = 'non_conjugate_iq'

        #################### NORMALIZE THE X DATA #######################
        if args.normalize == True:
            xtrain_iq_all = np.concatenate((xtrain_iq, xtest_iq))
              # Normalize the data with zero mean and unit variance for each column
            for i in range(xtrain_iq_all.shape[1]):
                standard_iq = preprocessing.StandardScaler().fit(xtrain_iq_all[:, i, :])
                xtrain_iq[:, i, :] = standard_iq.transform(xtrain_iq[:, i, :])
                xtest_iq[:, i, :] = standard_iq.transform(xtest_iq[:, i, :])

            xtrain_nc_all = np.concatenate((xtrain_nc, xtest_nc))
            standard_nc = preprocessing.StandardScaler().fit(xtrain_nc_all)  # Normalize the data with zero mean and unit variance for each column
            xtrain_nc = standard_nc.transform(xtrain_nc)
            xtest_nc = standard_nc.transform(xtest_nc)
        ############### END OF NORMAIZATION ################

        print("XTRAIN (IQ, NC) AND XTEST SHAPE:", xtrain_iq.shape, xtest_iq.shape, xtrain_nc.shape, xtest_nc.shape)
        print("YTRAIN AND YTEST SHAPE:", ytrain.shape, ytest.shape)

    # considering IQ, conjugate and non-conjugate CSP features
    if 'c' in args.input and 'nc' in args.input and 'iq' in args.input:
        xtrain_iq, xtest_iq, xtrain_c, xtest_c, xtrain_nc, xtest_nc, ytrain, ytest = train_test_split(inputs_iq, inputs_c, inputs_nc, labels, test_size = 0.2, random_state = 42)
        modelA = AlexNet1D(input_dim=xtrain_iq.shape[2], output_dim=args.num_classes, fusion=args.fusion_layer)
        # modelB = CSPNet(input_dim=xtrain_c.shape[1])
        # modelC = CSPNet(input_dim=xtrain_nc.shape[1])
        modelB = FeatureNet(input_dim=xtrain_c.shape[1], output_dim=args.num_classes)
        modelC = FeatureNet(input_dim=xtrain_nc.shape[1], output_dim=args.num_classes)
        model = FeatureFusionThree(modelA, modelB, modelC, nb_classes=args.num_classes, fusion=args.fusion_layer)
        saved_file_name = 'conjugate_non_conjugate_iq'

        #################### NORMALIZE THE X DATA #######################
        if args.normalize == True:
            xtrain_iq_all = np.concatenate((xtrain_iq, xtest_iq))
              # Normalize the data with zero mean and unit variance for each column
            for i in range(xtrain_iq_all.shape[1]):
                standard_iq = preprocessing.StandardScaler().fit(xtrain_iq_all[:, i, :])
                xtrain_iq[:, i, :] = standard_iq.transform(xtrain_iq[:, i, :])
                xtest_iq[:, i, :] = standard_iq.transform(xtest_iq[:, i, :])

            xtrain_c_all = np.concatenate((xtrain_c, xtest_c))
            standard_c = preprocessing.StandardScaler().fit(xtrain_c_all)  # Normalize the data with zero mean and unit variance for each column
            xtrain_c = standard_c.transform(xtrain_c)
            xtest_c = standard_c.transform(xtest_c)

            xtrain_nc_all = np.concatenate((xtrain_nc, xtest_nc))
            standard_nc = preprocessing.StandardScaler().fit(xtrain_nc_all)  # Normalize the data with zero mean and unit variance for each column
            xtrain_nc = standard_nc.transform(xtrain_nc)
            xtest_nc = standard_nc.transform(xtest_nc)
        ############### END OF NORMAIZATION ################

        print("XTRAIN (IQ, C, NC) AND XTEST SHAPE:", xtrain_iq.shape, xtest_iq.shape, xtrain_c.shape, xtest_c.shape, xtrain_nc.shape, xtest_nc.shape)
        print("YTRAIN AND YTEST SHAPE:", ytrain.shape, ytest.shape)

        ############### Commented: Old Implementation ##########################
        # if args.random_test == True: # This will be triggered when doing train/test split randomly on whole data
        #     #handling csp features
        #     inputs = np.concatenate((inputs_nc, inputs_c), axis=0)
        #     labels = np.concatenate((labels_nc, labels_c), axis=0)
        #     xtrain_csp, xtest_csp, ytrain_csp, ytest_csp = train_test_split(inputs, labels, test_size=0.2,
        #                                                     random_state=42)  # 80/20 is train/test size
        #
        #
        #     xtrain_iq, xtest_iq, ytrain_iq, ytest_iq = train_test_split(inputs_iq, labels_iq, test_size=0.2,
        #                                                                     random_state=42)  # 80/20 is train/test size
        #
        # else: # This will be triggered when doing train/test on separate blocks and optimal testing
        #     xtrain_csp = np.concatenate((xtrain_nc, xtrain_c), axis=0)
        #     ytrain_csp = np.concatenate((ytrain_nc, ytrain_c), axis=0)
        #     xtest_csp = np.concatenate((xtest_nc, xtest_c), axis=0)
        #     ytest_csp = np.concatenate((ytest_nc, ytest_c), axis=0)
        # modelA = FeatureNet(input_dim=xtrain_csp.shape[1], output_dim=args.num_classes)
        # modelB = RFNet(input_dim=xtrain_iq.shape[1], output_dim=args.num_classes)
        # model = FeatureFusion(modelA, modelB, nb_classes=args.num_classes, fusion=args.fusion_layer)
        # saved_file_name = 'conjugate_non_conjugate_iq'
        ############### Commented: Old Implementation ##########################




#################### NORMALIZE THE X DATA #######################
# if args.normalize == True:
#     xtrain_all = np.concatenate((xtrain, xtest))
#     standard = preprocessing.StandardScaler().fit(xtrain_all)  # Normalize the data with zero mean and unit variance for each column
#     xtrain = standard.transform(xtrain)
#     xtest = standard.transform(xtest)
############### END OF NORMAIZATION ################


######################### NEED TO PERFORM NORMALIZATION ##########


# print("All types: ", type(xtrain), type(xtest), type(ytrain), type(ytest))
# print("XTRAIN AND XTEST SHAPE:", xtrain.shape, xtest.shape)
# print("YTRAIN AND YTEST SHAPE:", ytrain.shape, ytest.shape)

if args.strategy != 1:
    print("The total number of occurances for different classes and total rows are: ")
    print("In train set: ", ytrain.sum(axis=0))
    print("In train set total number of anomalous (LTE+DSSS) and clean (LTE) signals are: ", ytrain.sum(axis=0)[0], (ytrain.shape[0]-ytrain.sum(axis=0)[0]))
    print("In test set: ", ytest.sum(axis=0))
    print("In test set total number of anomalous (LTE+DSSS) and clean (LTE)  signals are: ", ytest.sum(axis=0)[0], (ytest.shape[0]-ytest.sum(axis=0)[0]))



## WHEN SAVING WEIGHTS FOR PENULTIMATE LAYER
if args.fusion_layer == 'penultimate':  saved_file_name = saved_file_name + "_penultimate"

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
if use_cuda and args.id_gpu>=0:
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    model.cuda()
else:
    device = torch.device("cpu")


# DATALOADER FOR THREE MODALITY FUSION
class fusion_three_data_loader(object):
    def __init__(self, ds1, ds2, ds3, label):
        self.ds1 = ds1
        self.ds2 = ds2
        self.ds3 = ds3
        self.label = label

    def __getitem__(self, index):
        x1, x2, x3 = self.ds1[index], self.ds2[index],  self.ds3[index]
        label = self.label[index]
        return torch.from_numpy(x1), torch.from_numpy(x2),  torch.from_numpy(x3), torch.from_numpy(label)

    def __len__(self):
        return self.ds1.shape[0]  # assume both datasets have same length


# DATALOADER FOR DUAL FUSION
class fusion_two_data_loader(object):
    def __init__(self, ds1, ds2, label):
        self.ds1 = ds1
        self.ds2 = ds2
        self.label = label

    def __getitem__(self, index):
        x1, x2 = self.ds1[index], self.ds2[index]
        label = self.label[index]
        return torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(label)

    def __len__(self):
        return self.ds1.shape[0]  # assume both datasets have same length

# DATA LOADER FOR SINGLE MODALITY
class data_loader(object):
    def __init__(self, train_test):
        if train_test == 'train':
            self.feat = xtrain
            self.label = ytrain
            # print("types in train: ", type(xtrain), type(ytrain))
        elif train_test == 'test':
            self.feat = xtest
            self.label = ytest
            # print("types in test: ", type(xtest), type(ytest))
        print(train_test)

    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, index):
        # print("Types: ", type(self.feat), type(self.label))
        feat = self.feat[index]
        label = self.label[index] # change
        # print("Types: ", type(feat), type(label))
        return torch.from_numpy(feat), torch.from_numpy(label)

# DATA LOADER FOR SINGLE MODALITY
class data_loader_strategy2(object):
    def __init__(self, train_test):
        if train_test == 'train':
            self.feat = xtrain
            self.label = ytrain
            print("types in train: ", type(xtrain), type(ytrain))
        elif train_test == 'test':
            self.feat = xtest
            self.label = ytest
            print("types in test: ", type(xtest), type(ytest))
        print(train_test)

    def __len__(self):
        return len(self.feat)

    def __getitem__(self, index):
        # print("Types: ", type(self.feat), type(self.label))
        feat = self.feat[index]
        label = self.label[index] # change
        # print("Types: ", type(feat), type(label))
        return torch.from_numpy(feat), torch.from_numpy(label)


############################################################################################################
#############################    WORKING ON SINGLE MODAL ARCHITECTURES #####################################
############################################################################################################
def single_modal_training(saved_file_name):

    # loading the data
    # Parameters
    params = {'batch_size': args.bs,
              'shuffle': True,
              'num_workers': 0,
              'drop_last': True}

    if args.strategy == 1: training_set = data_loader_strategy2('train')
    else: training_set = data_loader('train')
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    if args.strategy == 1: test_set = data_loader_strategy2('test')
    else: test_set = data_loader('test')
    test_generator = torch.utils.data.DataLoader(test_set, **params)

    # setting up the loss function
    if args.num_classes ==2: criterion = torch.nn.CrossEntropyLoss()  # THIS IS USED WHEN THE LABELS ARE IN ONE HOT ENCODING
    else: criterion = torch.nn.BCEWithLogitsLoss() # THIS IS USED WHEN WE ARE DOING MULTI-LABEL CLASSIFICATION

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

    for epoch in range(int(args.epochs)):
        train_correct_anomaly = 0 # Acc is calculated per epoch for training data
        train_correct_mod = 0
        train_correct_gain = 0
        train_correct_all = 0
        train_total = 0  # Acc is calculated per epoch for training data
        test_correct_anomaly = 0
        test_correct_mod = 0
        test_correct_gain = 0
        test_correct_all = 0
        test_total = 0
        # print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        # print('-' * 10)
        train_start_time = time.time()
        model.train()
        # print("Working on epoch ", epoch)
        for train_batch, train_labels in training_generator:
            # print("test in training: ", train_batch.shape, train_labels.shape)
            train_batch, train_labels = train_batch.float().to(device), train_labels.float().to(device)

            outputs = model(train_batch)
            loss = criterion(outputs, torch.max(train_labels, 1)[1])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Getting the Acc
            outputs = outputs.cpu().detach().numpy()
            labels = train_labels.cpu().detach().numpy()
            train_total += labels.shape[0]
            if args.num_classes == 2:
                train_correct_anomaly += (np.argmax(labels, axis=1) == np.argmax(outputs, axis=1)).sum().item()
            else:
                train_correct_anomaly += (labels[:, 0] == outputs[:, 0]).sum()
            # train_correct_all += (all(labels[:, 1:] == np.round(outputs)[:, 1:])).sum()
                for i in range(labels.shape[0]):
                    train_correct_mod += int(all(labels[i, 1:3] == outputs[i, 1:3]))
                for i in range(labels.shape[0]):
                    train_correct_gain += int(all(labels[i, 3:8] == outputs[i, 3:8]))
                for i in range(labels.shape[0]):
                    train_correct_all += int(all(labels[i, :] == outputs[i, :]))
        train_end_time = time.time()
        model.eval()

        test_start_time = time.time()
        # Test
        for test_batch, test_labels in test_generator:
            test_batch, test_labels = test_batch.float().to(device), test_labels.float().to(device)
            test_output = model(test_batch)

            test_output = test_output.cpu().detach().numpy()
            test_labels = test_labels.squeeze().cpu().detach().numpy()

            # CALCULATING THE TEST ACCURACY
            test_total += test_labels.shape[0]
            # print("********* Sanity check: ", np.argmax(test_output, axis=1))
            if args.num_classes == 2:
                test_correct_anomaly += (np.argmax(test_labels, axis=1) == np.argmax(test_output, axis=1)).sum().item()
            else:
                test_correct_anomaly += (test_labels[:, 0] == test_output[:, 0]).sum()
                for i in range(test_labels.shape[0]):
                    test_correct_mod += int(all(test_labels[i, 1:3] == test_output[i, 1:3]))
                for i in range(test_labels.shape[0]):
                    test_correct_gain += int(all(test_labels[i, 3:8] == test_output[i, 3:8]))
                for i in range(test_labels.shape[0]):
                    test_correct_all += int(all(test_labels[i, :] == test_output[i, :]))

        test_acc_anomaly_detection.append(100 * test_correct_anomaly / test_total)
        if args.num_classes > 2:
            test_acc_anomaly_mod.append(100 * test_correct_mod / test_total)
            test_acc_anomaly_gain.append(100 * test_correct_gain / test_total)
            test_acc_anomaly_type.append(100 * test_correct_all / test_total)
        test_end_time = time.time()
        print("Time to train one epoch: ", (train_end_time - train_start_time), " seconds")
        print("Time to test one sample: ", (test_end_time - test_start_time)/args.bs, " seconds" )

        # print loss and accuracies
        if (epoch % 1 == 0): print('epoch {}, loss {} train acc of anomaly detection {} test acc of anomaly detection {}'.format(epoch, loss.data, (100 * train_correct_anomaly / train_total), (100 * test_correct_anomaly / test_total)))
        if args.num_classes > 2:
            if (epoch % 1 == 0): print('epoch {}, loss {} train acc of anomaly mod {} test acc of anomaly mod {}'.format(epoch, loss.data, (100 * train_correct_mod / train_total), (100 * test_correct_mod / test_total)))
            if (epoch % 1 == 0): print('epoch {}, loss {} train acc of anomaly gain {} test acc of anomaly gain {}'.format(epoch, loss.data, (100 * train_correct_gain / train_total), (100 * test_correct_gain / test_total)))
            if (epoch % 1 == 0): print('epoch {}, loss {} train acc of anomaly type {} test acc of anomaly type {}'.format(epoch,loss.data, (100 * train_correct_all / train_total),(100 * test_correct_all / test_total)))

    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, args.model_folder +"/"+saved_file_name+'.pth')

    torch.save(model, args.model_folder + '/' + saved_file_name + '.pt') # saving the whole model


# TRAINING ON ONLY NON-CONJUGATE CSP FEATURES
if fusion is False and "nc" in args.input:
    if args.classifier == 'nn':
        single_modal_training(saved_file_name)
    else:
        ytrain = np.argmax(ytrain, axis=1)
        ytest = np.argmax(ytest, axis=1)
        if args.classifier == 'lr':
            clf = LogisticRegression(random_state=0).fit(xtrain, ytrain)
        if args.classifier == 'nv':
            clf = GaussianNB().fit(xtrain, ytrain)
        if args.classifier == 'svm':
            clf = svm.SVC().fit(xtrain, ytrain)
            # score = clf.score(xtest, ytest)
        ypred = clf.predict(xtest)
        train_acc = clf.score(xtrain, ytrain)
        test_acc = metrics.accuracy_score(ytest, ypred)
        print("Test Accuracies for LTE and DSSS Detection using", args.classifier, " is :  train acc (", train_acc, ") and test acc (", test_acc, ").")

    print("Test Accuracies for LTE and DSSS Detection: ", test_acc_anomaly_detection)
    if args.num_classes > 2:
        print("Test Accuracies for DSSS (BPSK, QPSK) Detection: ", test_acc_anomaly_mod)
        print("Test Accuracies for DSSS Spreading Gain Detection: ", test_acc_anomaly_gain)
        print("Test Accuracies for DSSS Type Detection: ", test_acc_anomaly_type)
    print("Final test accuracy for LTE and DSSS Detection: ", test_acc_anomaly_detection[int(args.epochs) - 1])
    if args.num_classes > 2:
        print("Final test accuracy for DSSS (BPSK, QPSK) Detection: ", test_acc_anomaly_mod[int(args.epochs) - 1])
        print("Final test accuracy for DSSS Spreading Gain Detection: ", test_acc_anomaly_gain[int(args.epochs) - 1])
        print("Final test accuracy for DSSS Type Detection: ", test_acc_anomaly_type[int(args.epochs) - 1])
    print("End of Non-Conjugate")

# TRAINING ON ONLY CONJUGATE CSP FEATURES
if fusion is False and "c" in args.input:
    if args.classifier == 'nn':
        single_modal_training(saved_file_name)
    else:
        ytrain = np.argmax(ytrain, axis=1)
        ytest = np.argmax(ytest, axis=1)
        if args.classifier == 'lr':
            clf = LogisticRegression(random_state=0).fit(xtrain, ytrain)
        if args.classifier == 'nv':
            clf = GaussianNB().fit(xtrain, ytrain)
        if args.classifier == 'svm':
            clf = svm.SVC().fit(xtrain, ytrain)
            # score = clf.score(xtest, ytest)
        ypred = clf.predict(xtest)
        train_acc = clf.score(xtrain, ytrain)
        test_acc = metrics.accuracy_score(ytest, ypred)
        print("Test Accuracies for LTE and DSSS Detection using", args.classifier, " is :  train acc (", train_acc, ") and test acc (", test_acc, ").")

    print("Test Accuracies for LTE and DSSS Detection: ", test_acc_anomaly_detection)
    if args.num_classes > 2:
        print("Test Accuracies for DSSS (BPSK, QPSK) Detection: ", test_acc_anomaly_mod)
        print("Test Accuracies for DSSS Spreading Gain Detection: ", test_acc_anomaly_gain)
        print("Test Accuracies for DSSS Type Detection: ", test_acc_anomaly_type)
    print("Final test accuracy for LTE and DSSS Detection: ", test_acc_anomaly_detection[int(args.epochs) - 1])
    if args.num_classes > 2:
        print("Final test accuracy for DSSS (BPSK, QPSK) Detection: ", test_acc_anomaly_mod[int(args.epochs) - 1])
        print("Final test accuracy for DSSS Spreading Gain Detection: ", test_acc_anomaly_gain[int(args.epochs) - 1])
        print("Final test accuracy for DSSS Type Detection: ", test_acc_anomaly_type[int(args.epochs) - 1])
    print("End of Conjugate")


# TRAINING ON IQ DATA
# input size of DL framework (batch size, slice size,I/Q) = (256, 256, 2)
if fusion is False and "iq" in args.input:
    single_modal_training(saved_file_name)
    print("Test Accuracies for LTE and DSSS Detection: ", test_acc_anomaly_detection)
    print("Final test accuracy for LTE and DSSS Detection: ", test_acc_anomaly_detection[int(args.epochs) - 1])
    print("End of I/Q")


# TRAINING ON BOTH CONJUGATE AND NON-CONJUGATE CSP FEATURES
if fusion is True and len(args.input) == 2 and 'nc' in args.input and 'c' in args.input:
    if args.classifier == 'nn':
        single_modal_training(saved_file_name)
    else:
        ytrain = np.argmax(ytrain, axis=1)
        ytest = np.argmax(ytest, axis=1)
        if args.classifier == 'lr':
            clf = LogisticRegression(random_state=0).fit(xtrain, ytrain)
        if args.classifier == 'nv':
            clf = GaussianNB().fit(xtrain, ytrain)
        if args.classifier == 'svm':
            clf = svm.SVC().fit(xtrain, ytrain)
            # score = clf.score(xtest, ytest)
        ypred = clf.predict(xtest)
        train_acc = clf.score(xtrain, ytrain)
        test_acc = metrics.accuracy_score(ytest, ypred)
        print("Test Accuracies for LTE and DSSS Detection using", args.classifier, " is :  train acc (", train_acc, ") and test acc (", test_acc, ").")

    print("Test Accuracies for LTE and DSSS Detection: ", test_acc_anomaly_detection)
    if args.num_classes > 2:
        print("Test Accuracies for DSSS (BPSK, QPSK) Detection: ", test_acc_anomaly_mod)
        print("Test Accuracies for DSSS Spreading Gain Detection: ", test_acc_anomaly_gain)
        print("Test Accuracies for DSSS Type Detection: ", test_acc_anomaly_type)
    print("Final test accuracy for LTE and DSSS Detection: ", test_acc_anomaly_detection[int(args.epochs) - 1])
    if args.num_classes > 2:
        print("Final test accuracy for DSSS (BPSK, QPSK) Detection: ", test_acc_anomaly_mod[int(args.epochs) - 1])
        print("Final test accuracy for DSSS Spreading Gain Detection: ", test_acc_anomaly_gain[int(args.epochs) - 1])
        print("Final test accuracy for DSSS Type Detection: ", test_acc_anomaly_type[int(args.epochs) - 1])
    print("End of Non-Conjugate and Conjugate")

############################################################################################################
#############################    WORKING ON TWO MODALITY FUSION ARCHITECTURES #####################################
############################################################################################################
def two_modality_training(saved_file_name, xtrain_mod1, xtrain_mod2, ytrain, xtest_mod1, xtest_mod2, ytest):
    params = {'batch_size': args.bs,
              'shuffle': True,
              'num_workers': 0,
              'drop_last': True}

    training_set = fusion_two_data_loader(xtrain_mod1, xtrain_mod2, ytrain)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    testing_set = fusion_two_data_loader(xtest_mod1, xtest_mod2, ytest)
    test_generator = torch.utils.data.DataLoader(testing_set, **params)
    #model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    # Loss and optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)


    for epoch in range(int(args.epochs)):
        train_correct = 0 # Acc is calculated per epoch for training data
        train_total = 0  # Acc is calculated per epoch for training data
        test_correct = 0
        test_total = 0
        for i, (batch1, batch2, train_labels) in enumerate(training_generator):
                batch1, batch2, train_labels = batch1.float().to(device), batch2.float().to(device), train_labels.float().to(device)

                # Forward pass
                outputs = model(batch1, batch2)
                loss = criterion(outputs, torch.max(train_labels, 1)[1])

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Getting Acc
                outputs = outputs.cpu().detach().numpy()
                labels = train_labels.cpu().detach().numpy()
                train_total += labels.shape[0]
                train_correct += (np.argmax(labels, axis=1) == np.argmax(outputs, axis=1)).sum().item()
        model.eval()

        # Test

        for test_batch1, test_batch2, test_labels in test_generator:
            test_batch1, test_batch2, test_labels = test_batch1.float().to(device), test_batch2.float().to(
                device), test_labels.float().to(device)
            test_output = model(test_batch1, test_batch2)

            test_output = test_output.cpu().detach().numpy()
            test_labels = test_labels.squeeze().cpu().detach().numpy()

            # CALCULATING THE TEST ACCURACY
            test_total += test_labels.shape[0]
            test_correct += (np.argmax(test_labels, axis=1) == np.argmax(test_output, axis=1)).sum().item()
        test_acc_anomaly_detection.append(100 * test_correct / test_total)


        # print loss and accuracies
        if (epoch % 1 == 0): print('epoch {}, loss {} train acc {} test acc {}'.format(epoch, loss.data, (100 * train_correct / train_total), (100 * test_correct / test_total)))


    torch.save(model, args.model_folder + '/' + saved_file_name + '.pt')
    if args.retrain: torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, args.model_folder + '/' + saved_file_name + '.pth')



# TRAINING ON CONJUGATE CSP FEATURES AND IQ DATA
if fusion is True and len(args.input) == 2 and 'c' in args.input and 'iq' in args.input:
    two_modality_training(saved_file_name, xtrain_iq, xtrain_c, ytrain, xtest_iq, xtest_c, ytest)
    print("Test Accuracies for LTE and DSSS Detection: ", test_acc_anomaly_detection)
    print("Final test accuracy for LTE and DSSS Detection: ", test_acc_anomaly_detection[int(args.epochs) - 1])
    print("End of I/Q")


# TRAINING ON NON-CONJUGATE CSP FEATURES AND IQ DATA
if fusion is True and len(args.input) == 2 and 'nc' in args.input and 'iq' in args.input:
    two_modality_training(saved_file_name, xtrain_iq, xtrain_nc, ytrain, xtest_iq, xtest_nc, ytest)
    print("Test Accuracies for LTE and DSSS Detection: ", test_acc_anomaly_detection)
    print("Final test accuracy for LTE and DSSS Detection: ", test_acc_anomaly_detection[int(args.epochs) - 1])
    print("End of I/Q")


############################################################################################################
#############################    WORKING ON THREE MODALITY FUSION ARCHITECTURES #############################
############################################################################################################
def three_modality_training(saved_file_name, xtrain_mod1, xtrain_mod2, xtrain_mod3, ytrain, xtest_mod1, xtest_mod2, xtest_mod3, ytest):
    params = {'batch_size': args.bs,
              'shuffle': True,
              'num_workers': 0,
              'drop_last': True}

    training_set = fusion_three_data_loader(xtrain_mod1, xtrain_mod2, xtrain_mod3, ytrain)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    testing_set = fusion_three_data_loader(xtest_mod1, xtest_mod2, xtest_mod3, ytest)
    test_generator = torch.utils.data.DataLoader(testing_set, **params)
    #model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    # Loss and optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)


    for epoch in range(int(args.epochs)):
        train_correct = 0 # Acc is calculated per epoch for training data
        train_total = 0  # Acc is calculated per epoch for training data
        test_correct = 0
        test_total = 0
        for i, (batch1, batch2, batch3, train_labels) in enumerate(training_generator):
                batch1, batch2, batch3, train_labels = batch1.float().to(device), batch2.float().to(device), batch3.float().to(device), train_labels.float().to(device)

                # Forward pass
                outputs= model(batch1, batch2, batch3)
                loss = criterion(outputs, torch.max(train_labels, 1)[1])

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Getting Acc
                outputs = outputs.cpu().detach().numpy()
                labels = train_labels.cpu().detach().numpy()
                train_total += labels.shape[0]
                train_correct += (np.argmax(labels, axis=1) == np.argmax(outputs, axis=1)).sum().item()
        model.eval()

        # Test

        for test_batch1, test_batch2, test_batch3, test_labels in test_generator:
            test_batch1, test_batch2, test_batch3, test_labels = test_batch1.float().to(device), test_batch2.float().to(device), test_batch3.float().to(device), test_labels.float().to(device)
            test_output = model(test_batch1, test_batch2, test_batch3)

            test_output = test_output.cpu().detach().numpy()
            test_labels = test_labels.squeeze().cpu().detach().numpy()

            # CALCULATING THE TEST ACCURACY
            test_total += test_labels.shape[0]
            test_correct += (np.argmax(test_labels, axis=1) == np.argmax(test_output, axis=1)).sum().item()
        test_acc_anomaly_detection.append(100 * test_correct / test_total)


        # print loss and accuracies
        if (epoch % 1 == 0): print('epoch {}, loss {} train acc {} test acc {}'.format(epoch, loss.data, (100 * train_correct / train_total), (100 * test_correct / test_total)))


    torch.save(model, args.model_folder + '/' + saved_file_name + '.pt')
    if args.retrain: torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, args.model_folder + '/' + saved_file_name + '.pth')



# TRAINING ON CONJUGATE CSP FEATURES AND IQ DATA
if fusion is True and len(args.input) == 3 and 'c' in args.input and 'nc' in args.input and 'iq' in args.input:
    three_modality_training(saved_file_name, xtrain_iq, xtrain_c, xtrain_nc, ytrain, xtest_iq, xtest_c, xtest_nc, ytest)
    print("Test Accuracies for LTE and DSSS Detection: ", test_acc_anomaly_detection)
    print("Final test accuracy for LTE and DSSS Detection: ", test_acc_anomaly_detection[int(args.epochs) - 1])
    print("End of I/Q, conjugate and non-conjugate")


# Calculating total execution time
end_time = time.time()  # Taking end time to calculate overall execution time
print("\n Total Execution Time (Minutes): ")
print(((end_time - start_time) / 60))
