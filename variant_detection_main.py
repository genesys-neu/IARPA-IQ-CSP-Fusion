"""
@author: debashri
This is the main file to run the code to detect anomalous signal (e.g., DSSS) from baselines (e.g., LTE) using IQ data and features in the IARPA project.
"""

# The absence of cyclic features is important. Just looking at BPSK and QPSK, they both have a single NC feature. QPSK has no C features, BPSK has three.

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
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from pthflops import count_ops # not working

from ptflops import get_model_complexity_info

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
torch.manual_seed(seed)


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
parser.add_argument('--data_folder', help='Location of the data directory', type=str, default= '/home/royd/IARPA/')
parser.add_argument('--input', nargs='*', default=['iq'],choices = ['iq', 'c', 'nc'],
help='Which data to use as input. Select from: raw IQ data, non-conjugate features, conjugate features.')
parser.add_argument('--feature_options', type = int, nargs='+', default=[0, 1, 2, 3],choices = [0, 1, 2, 3],
help='Which features to use from the conjugate and non-conjugate files.')
parser.add_argument('--iq_slice_len',default=131072, type=int,help='Slice length for processing IQ files') # 32
parser.add_argument('--num_classes', default=2, type=int, help='Number of classes for classification.')
parser.add_argument('--strategy',  type=int, default =4, choices = [0, 2, 3, 4], help='Different strategies used for CSP feature processing: naive (0),  extract stat (2), 3D matrix (3), extract stat specific max (4). 2D matrix (1) is not handled here')

#Arguments for train/test on 80/20 percentage of overall data randomly
parser.add_argument('--random_test', type=str2bool, help='Perform train/test on 80/20 of data.', default=True)
parser.add_argument('--random_test_blocks', type = int, nargs='+', default=[131072],choices = [131072, 262144, 524288],
help='The block lengths to use for random train/test.')

# Train and test on specific SNR values
parser.add_argument('--random_snr_test', type=str2bool, help='Perform training on features from one SNR level and other.', default=True)
parser.add_argument('--random_snr', type = int, nargs='+', default=[0, 5, 10],choices = [0, 5, 10],
help='The SNR to use for random train/test.')
# parser.add_argument('--training_snr', nargs='*', default=[5],choices = [0, 5, 10],
# help='Which SNR to use for training.') # use this argument only if you set the 'random_snr_test' argument as False
# parser.add_argument('--testing_snr', nargs='*', default=[0],choices = [0, 5, 10],
# help='Which SNR to use for training.') # use this argument only if you set the 'random_snr_test' argument as False
parser.add_argument('--dsss_sir', type = int, nargs='+', default=[0, 5, 10],choices = [0, 5, 10],
help='Which SIRs of DSSS to use for train/test.') # this argument will be used always #0: less challenging, 10: more challenging


#Neural Network Parameters
parser.add_argument('--classifier', type=str, default='nn', help='Options: lr (for logistic regression), nv (naive bayes), svm (support vector machine), nn (neural network).')
parser.add_argument('--lr', default=0.0001, type=float,help='learning rate for Adam optimizer',)
parser.add_argument('--bs',default=8, type=int,help='Batch size') # 32 # 8
parser.add_argument('--epochs', default=50, type = int, help='Specify the epochs to train')
parser.add_argument('--normalize', help='normalize or not', type=str2bool, default =True)
parser.add_argument('--id_gpu', default=1, type=int, help='which gpu to use.')
parser.add_argument('--model_folder', help='Location of the data directory', type=str, default= '/home/royd/IARPA/Saved_Models_Variant/')

# Train and test on the data from Chad
parser.add_argument('--powder_data', type=str2bool, default=False, help='Perform training and testing on real dataset collected in POWDER.') # by default all DSSS
parser.add_argument('--real_data', type=str2bool, default=False, help='Perform training and testing on real LTE and all (synthetic+real) DSSS data from Chad.') # by default all DSSS
parser.add_argument('--percentage_to_read', type=float, default=1, help='The percentage of data (from real dataset) you want to read, choose between [0-1].') # by default all DSSS
parser.add_argument('--dsss_type', type=str, default='all', choices = ['all', 'real', 'synthetic'], help='Specify which type of DSSS signals you want to use for training and testing.')
parser.add_argument('--meta_learning', type=str2bool, default=False, help='Perform meta learning from trained model on synthetic data to real data.')
parser.add_argument('--no_of_layers_to_freeze',  type=int, default =1, choices = [0, 1, 2, 3, 4], help="Specify the number of layers you want to freeze from the starting of the model.")
parser.add_argument('--retrain_ratio',  type=float, default =0.1, choices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], help="Specify the percentage of retraining would be done for meta learning.")
parser.add_argument('--model_file', help='Location of the model file (with directory)', type=str, default= '/home/royd/IARPA/Saved_Models/block_wise_trained_model_on_sythetic_dataset_strategy5/non_conjugate_131072.pt')
parser.add_argument('--fusion_layer', type=str, default='penultimate', help='Assign the layer name where the fusion to be performed.')
parser.add_argument('--restore_models', type=str2bool, help='Load single modality trained weights', default=False)
parser.add_argument('--retrain', type=str2bool, help='Retrain the model on loaded weights', default=True)
parser.add_argument('--slicing', type=str2bool, help='Perform slicing of the I/Q signals for accelerated training', default=False)
parser.add_argument('--slice_length', type=int, help='The length of the slices.', default=256)

# Not being used so far
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
features_tsne = np.zeros(1)


#############################################################################################################
# GENERATING inputs and labels
############################################################################################################

# # Conjugate/Non-conjugate FEATURES
# if len(args.input) >=2:
    # train/test on a specific block length for CSP features and train/test on random SNR
if args.real_data == True:
        inputs_iq, inputs_c, inputs_nc, labels = read_file.generate_inputs_labels_variants_IQ_NC_C_for_NWRA_dataset(args.data_folder, args.input, args.feature_options, args.iq_slice_len, args.num_classes, args.random_snr, args.dsss_sir, args.strategy, args.dsss_type, args.percentage_to_read, args.slicing, args.slice_length)
elif args.powder_data == True:
        inputs_iq, inputs_c, inputs_nc, labels = read_file.generate_inputs_labels_variants_IQ_NC_C_POWDER_dataset(args.data_folder, args.input, args.feature_options, args.iq_slice_len, args.strategy, args.percentage_to_read, args.slicing, args.slice_length)
        
else:
        inputs_iq, inputs_c, inputs_nc, labels = read_file.generate_inputs_labels_variants_IQ_NC_C_NEU_dataset(args.data_folder,
                                                                                                      args.input,
                                                                                                      args.feature_options,
                                                                                                      args.iq_slice_len,
                                                                                                      args.num_classes,
                                                                                                      args.random_snr,
                                                                                                      args.dsss_sir,
                                                                                                      args.strategy,
                                                                                                      args.percentage_to_read, args.slicing, args.slice_length)
print("Shaped of the inputs and labels: ", inputs_iq.shape, inputs_c.shape, inputs_nc.shape, labels.shape)


saved_file_name = 'Variant_model'
for x in  args.input:
    saved_file_name = saved_file_name + "_" + x
                  # 'IQ_' + str(args.iq_slice_len)
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
from ModelHandler_Variant import NonConjugateNet, ConjugateNet, FeatureNet, AlexNet1D, FeatureFusion, RFNet, CSPNet, FeatureFusionThree, FeatureNetCNN



#############################################################################################################################
# WORKING ON EACH INDIVIDUAL MODALITIES
#############################################################################################################################
def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


if fusion == False:
    if 'nc' in args.input:
        if args.real_data == True  or args.powder_data == True:
            inputs = inputs_nc
            if args.meta_learning == True:
                xtrain, xtest, ytrain, ytest = train_test_split(inputs, labels, test_size=(1-args.retrain_ratio), random_state=42)  # 80/20 is train/test size
            else:
                xtrain, xtest, ytrain, ytest = train_test_split(inputs, labels, test_size=0.2, random_state=42)  # 80/20 is train/test size
        else:
            if args.random_test == True: # This will be triggered when doing train/test split randomly on whole data
                if args.random_snr_test == True:
                    inputs = inputs_nc
                    xtrain, xtest, ytrain, ytest = train_test_split(inputs, labels, test_size=0.2, random_state=42)  # 80/20 is train/test size


        # if args.strategy == 1: model = FeatureNetCNN(input_dim=xtrain[0].shape[0], output_dim=args.num_classes)
        model = FeatureNet(input_dim=xtrain.shape[1], output_dim=args.num_classes)



    if 'c' in args.input:
        if args.real_data == True  or args.powder_data == True:
            inputs = inputs_c
            # labels = labels_c
            xtrain, xtest, ytrain, ytest = train_test_split(inputs, labels, test_size=0.2,
                                                            random_state=42)  # 80/20 is train/test size
        else:
            if args.random_test == True: # This will be triggered when doing train/test split randomly on whole data
                if args.random_snr_test == True:
                    inputs = inputs_c
                    # labels = labels_c
                    xtrain, xtest, ytrain, ytest = train_test_split(inputs, labels, test_size=0.2,
                                                                    random_state=42)  # 80/20 is train/test size

        model = FeatureNet(input_dim=xtrain.shape[1], output_dim=args.num_classes)
        # model = NonConjugateNet(input_dim=xtrain.shape[1], output_dim=args.num_classes)

    if 'iq' in args.input:
        # This will be triggered when doing train/test split randomly on whole data
        if args.random_snr_test == True:
            inputs = inputs_iq
            # labels = labels_iq
            xtrain, xtest, ytrain, ytest = train_test_split(inputs, labels, test_size=0.2,
                                                                random_state=42)  # 80/20 is train/test size
        model = RFNet(input_dim=xtrain.shape[2], output_dim=args.num_classes)
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
        if args.random_test == True or args.real_data == True  or args.powder_data == True: # This will be triggered when doing train/test split randomly on whole data
            inputs = np.concatenate((inputs_nc, inputs_c), axis=0)
            labels = np.concatenate((labels, labels), axis=0)
            xtrain, xtest, ytrain, ytest = train_test_split(inputs, labels, test_size=0.2,
                                                            random_state=42)  # 80/20 is train/test size

        model = FeatureNet(input_dim=xtrain.shape[1], output_dim=args.num_classes)
        # model = NonConjugateNet(input_dim=xtrain.shape[1], output_dim=args.num_classes)
        saved_file_name = 'Variant_model_conjugate_non_conjugate'

    # considering IQ, conjugate CSP features
    if 'iq' in args.input and 'c' in args.input and 'nc' not in args.input:
        xtrain_iq, xtest_iq, xtrain_c, xtest_c, ytrain, ytest = train_test_split(inputs_iq, inputs_c, labels, test_size=0.2, random_state=42)
        modelA = RFNet(input_dim=xtrain_iq.shape[2], output_dim=args.num_classes, fusion=args.fusion_layer)
        # modelB = CSPNet(input_dim=xtrain_c.shape[1])
        modelB = FeatureNet(input_dim=xtrain_c.shape[1], output_dim=args.num_classes, fusion=args.fusion_layer)

        ############# LOADING THE MODELS ##########################
        if args.restore_models == True:
            # print("Entering in restore model..")
            if args.slicing == True:
                iq_file_name = args.model_folder + 'NWRA_dataset_models/Slice_256/IQ_' + str(args.iq_slice_len) + '.pt'
            else:
                iq_file_name = args.model_folder + 'NWRA_dataset_models/IQ_' + str(args.iq_slice_len) + '.pt'
            c_file_name = args.model_folder + 'NWRA_dataset_models/conjugate_' + str(args.iq_slice_len) + '.pt'

            if args.fusion_layer == 'penultimate':
                mA = torch.load(iq_file_name)
                modelA = torch.nn.Sequential(*(list(mA.children())[:-1]))
                mB = torch.load(c_file_name)
                modelB = torch.nn.Sequential(*(list(mB.children())[:-1]))
            else:
                modelA = torch.load(iq_file_name)
                modelB = torch.load(c_file_name)
            print("LOADED THE MODELS FOR I/Q and Conjugate")

        # FREEZING THE WEIGHTS BEFORE THE FUSION LAYERS
        if args.retrain == False:
            print("FREEZING THE WEIGHTS BEFORE FUSION LAYERS")
            for c in modelA.children():
                for param in c.parameters():
                    param.requires_grad = False
            for c in modelB.children():
                for param in c.parameters():
                    param.requires_grad = False

        ###########################################################


        model = FeatureFusion(modelA, modelB, nb_classes=args.num_classes, fusion=args.fusion_layer)
        saved_file_name = 'Variant_model_conjugate_iq'

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
        modelA = RFNet(input_dim=xtrain_iq.shape[2], output_dim=args.num_classes, fusion=args.fusion_layer)
        # modelB = CSPNet(input_dim=xtrain_nc.shape[1])
        modelB = FeatureNet(input_dim=xtrain_nc.shape[1], output_dim=args.num_classes, fusion=args.fusion_layer)

        ############# LOADING THE MODELS ##########################
        if args.restore_models == True:
            # print("Entering in restore model..")
            if args.slicing == True: iq_file_name = args.model_folder + 'NWRA_dataset_models/Slice_256/IQ_'+str(args.iq_slice_len)+'.pt'
            else: iq_file_name = args.model_folder + 'NWRA_dataset_models/IQ_' + str(args.iq_slice_len) + '.pt'
            nc_file_name = args.model_folder + 'NWRA_dataset_models/non_conjugate_'+str(args.iq_slice_len)+'.pt'

            if args.fusion_layer == 'penultimate':
                mA = torch.load(iq_file_name)
                modelA = torch.nn.Sequential(*(list(mA.children())[:-1]))
                mB = torch.load(nc_file_name)
                modelB = torch.nn.Sequential(*(list(mB.children())[:-1]))
            else:
                modelA = torch.load(iq_file_name)
                modelB = torch.load(nc_file_name)
            print("LOADED THE MODELS FOR I/Q and Non-conjugate")

        # FREEZING THE WEIGHTS BEFORE THE FUSION LAYERS
        if args.retrain == False:
            print("FREEZING THE WEIGHTS BEFORE FUSION LAYERS")
            for c in modelA.children():
                for param in c.parameters():
                    param.requires_grad = False
            for c in modelB.children():
                for param in c.parameters():
                    param.requires_grad = False

        ###########################################################


        model = FeatureFusion(modelA, modelB, nb_classes=args.num_classes, fusion=args.fusion_layer)
        saved_file_name = 'Variant_model_non_conjugate_iq'

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
        modelA = RFNet(input_dim=xtrain_iq.shape[2], output_dim=args.num_classes, fusion=args.fusion_layer)
        modelB = FeatureNet(input_dim=xtrain_c.shape[1], output_dim=args.num_classes, fusion=args.fusion_layer)
        modelC = FeatureNet(input_dim=xtrain_nc.shape[1], output_dim=args.num_classes, fusion=args.fusion_layer)
        model = FeatureFusionThree(modelA, modelB, modelC, nb_classes=args.num_classes, fusion=args.fusion_layer)
        saved_file_name = 'Variant_model_conjugate_non_conjugate_iq'

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




#################### NORMALIZE THE X DATA #######################
# if args.normalize == True:
#     xtrain_all = np.concatenate((xtrain, xtest))
#     standard = preprocessing.StandardScaler().fit(xtrain_all)  # Normalize the data with zero mean and unit variance for each column
#     xtrain = standard.transform(xtrain)
#     xtest = standard.transform(xtest)
############### END OF NORMAIZATION ################


# print("All types: ", type(xtrain), type(xtest), type(ytrain), type(ytest))
# print("XTRAIN AND XTEST SHAPE:", xtrain.shape, xtest.shape)
# print("YTRAIN AND YTEST SHAPE:", ytrain.shape, ytest.shape)

if args.strategy != 1:
    print("The total number of occurances for different classes and total rows are: ")
    print("In train set: ", ytrain.sum(axis=0))
    print("In train set total number of DSSS BPSK and DSSS QPSK signals are: ", ytrain.sum(axis=0)[0], ytrain.sum(axis=0)[1])
    print("In test set: ", ytest.sum(axis=0))
    print("In test set total number of DSSS BPSK and DSSS QPSK signals are: ", ytest.sum(axis=0)[0], ytest.sum(axis=0)[1])

saved_file_name = saved_file_name +  "_" + str(args.iq_slice_len)

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


######################### TSNE PLOT GENERATOR ##############

def tsne_plot_generator(features_tsne, y_value, saved_file_name):
    # Fit and transform with a TSNE
    tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=300,random_state=0)


    # Project the data in 2D
    tsne_results = tsne.fit_transform(features_tsne)
    df = pd.DataFrame()
    df['y'] = y_value
    df['Dim1'] = tsne_results[:,0]
    df['Dim2'] = tsne_results[:,1]

    print(df.shape)

    df_subset = df.loc[df["y"]>-1]
    # print(data.shape)
    # plt.figure()
    sns.set(font_scale = 2)
    ax = sns.scatterplot(
      x="Dim1", y="Dim2",
      hue="y",
      palette=sns.color_palette("hls", len(np.unique(y_value))),
      data=df_subset,
      legend="full"
    )
    handles, labels  =  ax.get_legend_handles_labels()

    ax.legend(handles, ['BPSK', 'QPSK'])

    plt.savefig('t-SNE/'+saved_file_name+'_tsne.png',bbox_inches='tight',dpi=400)



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
        
        
# training function
def fit(model, train_dataloader, train_dataset, optimizer, criterion):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    total = 0
    prog_bar = tqdm(enumerate(train_dataloader), total=int(len(train_dataset)/train_dataloader.batch_size))
    for i, data in prog_bar:
        counter += 1
        data, target = data[0].to(device), data[1].to(device)
        total += target.size(0)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
        
    train_loss = train_running_loss / counter
    train_accuracy = 100. * train_running_correct / total
    return train_loss, train_accuracy


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
    criterion = torch.nn.CrossEntropyLoss()  # THIS IS USED WHEN THE LABELS ARE IN ONE HOT ENCODING
    # else: criterion = torch.nn.BCEWithLogitsLoss() # THIS IS USED WHEN WE ARE DOING MULTI-LABEL CLASSIFICATION

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5)
    #scheduler = torch.optim.lr_scheduler.StepLR( optimizer, step_size = 50, gamma = 0.1)

    for epoch in range(int(args.epochs)):
        train_correct_anomaly = 0 # Acc is calculated per epoch for training data
        train_correct_all = 0
        train_total = 0  # Acc is calculated per epoch for training data
        test_correct_anomaly = 0
        test_correct_all = 0
        test_total = 0
        # print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        # print('-' * 10)
        train_start_time = time.time()
        model.train()
        # print("Working on epoch ", epoch)
        for train_batch, train_labels in training_generator:
            # print("test in training: ", train_batch.shape, train_labels.shape)
            #if 'iq' in args.input: train_batch = torch.reshape(train_batch, (train_batch.shape[0], train_batch.shape[2], train_batch.shape[1]))# adding this to comment out the reshape from the modelHandler to calcualte the FLOPS
            train_batch, train_labels = train_batch.float().to(device), train_labels.float().to(device)

            outputs, _ = model(train_batch)
            loss = criterion(outputs, torch.max(train_labels, 1)[1])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step(loss) # decreasing LR on plateau

            # Getting the Acc
            outputs = outputs.cpu().detach().numpy()
            labels = train_labels.cpu().detach().numpy()
            train_total += labels.shape[0]
            # if args.num_classes == 2:
            train_correct_anomaly += (np.argmax(labels, axis=1) == np.argmax(outputs, axis=1)).sum().item()
        #print("The learning rate is: ", optimizer.param_groups[0]['lr'])
        print("The learning rate is: ", optimizer.param_groups[0]['lr'])
        train_end_time = time.time()
        model.eval()
        
        

        test_start_time = time.time()
        # Test
        # To create the t-SNE plots
        features_tsne = np.zeros(1)
        y_tsne = np.zeros(1)
        test_batch_count = 0
        
        for test_batch, test_labels in test_generator:
            #if 'iq' in args.input: test_batch = torch.reshape(test_batch, (test_batch.shape[0], test_batch.shape[2], test_batch.shape[1]))# adding this to comment out the reshape from the modelHandler to calcualte the FLOPS
            test_batch, test_labels = test_batch.float().to(device), test_labels.float().to(device)
            test_output, latent_features = model(test_batch)
            
            

            test_output = test_output.cpu().detach().numpy()
            test_labels = test_labels.squeeze().cpu().detach().numpy()
            test_batch = test_batch.squeeze().cpu().detach().numpy()
            np.set_printoptions(precision=3)
#            print('Sanity check: input is {:.4f}, actual labels {}, predicted labels {}'.format(test_batch, np.argmax(test_labels, axis=1),  np.argmax(test_output, axis=1)))
            #print("Sanity check:", test_batch, np.argmax(test_labels, axis=1),  np.argmax(test_output, axis=1))
            print("Sanity check:", np.argmax(test_labels, axis=1),  np.argmax(test_output, axis=1))

            # CALCULATING THE TEST ACCURACY
            test_total += test_labels.shape[0]
            # print("********* Sanity check: ", np.argmax(test_output, axis=1))
            # if args.num_classes == 2:
            test_correct_anomaly += (np.argmax(test_labels, axis=1) == np.argmax(test_output, axis=1)).sum().item()
            
            ############################ t-SNE PLOT ##############################
            if test_batch_count == 0:
              features_tsne = latent_features.cpu().detach().numpy()
              y_tsne = np.argmax(test_output, axis=1)
            else:
              features_tsne = np.concatenate ((features_tsne, latent_features.cpu().detach().numpy()), axis = 0)
              y_tsne = np.concatenate ((y_tsne, np.argmax(test_output, axis=1)), axis = 0)
            
            test_batch_count += 1
            ##############################################################
            
            ################## Calculate FLOPS ###############################
            #with torch.cuda.device(args.id_gpu):
              #net = models.densenet161()
            #print("The number of FLOPS:",  count_ops(model, test_batch.cpu().detach().numpy().shape))
            #if 'iq' in args.input:  
            #  test_batch =  test_batch.cpu().detach().numpy()
            #  inp_shp =  (test_batch.shape[2], test_batch.shape[1])
            #else: inp_shp =  test_batch.cpu().detach().numpy().shape
            #macs, params = get_model_complexity_info(model, inp_shp, as_strings=True,
            #                              print_per_layer_stat=True, verbose=True)
            #print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            #print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            ###########################################################

        test_acc_anomaly_detection.append(100 * test_correct_anomaly / test_total)
        test_end_time = time.time()
        print("Time to train one epoch: ", (train_end_time - train_start_time), " seconds")
        print("Time to test one sample: ", (test_end_time - test_start_time)/args.bs, " seconds" )

        # print loss and accuracies
        if (epoch % 1 == 0): print('epoch {}, loss {} train acc of variant detection {} test acc of variant detection {}'.format(epoch, loss.data, (100 * train_correct_anomaly / train_total), (100 * test_correct_anomaly / test_total)))
        
    # Plotting the features in t-SNE format 
    tsne_plot_generator(features_tsne, y_tsne, saved_file_name)
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
        print("Test Accuracies for DSSS Variant Detection: ", test_acc_anomaly_detection)
        print("Final test accuracy for DSSS Variant Detection: ", test_acc_anomaly_detection[int(args.epochs) - 1])
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
        print("Test Accuracies for DSSS Variant Detection using ", args.classifier, " is :  train acc (", train_acc, ") and test acc (", test_acc, ").")
        # Plotting the features in t-SNE format 
        tsne_plot_generator(xtest, ypred, saved_file_name)
    
    
    print("End of Non-Conjugate")

# TRAINING ON ONLY CONJUGATE CSP FEATURES
if fusion is False and "c" in args.input:

    if args.classifier == 'nn':
        single_modal_training(saved_file_name)
        print("Test Accuracies for DSSS Variant Detection: ", test_acc_anomaly_detection)
        print("Final test accuracy for DSSS Variant Detection: ", test_acc_anomaly_detection[int(args.epochs) - 1])
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
        print("Test Accuracies for DSSS Variant Detection using", args.classifier, " is :  train acc (", train_acc, ") and test acc (", test_acc, ").")
        # Plotting the features in t-SNE format 
        tsne_plot_generator(xtest, ypred, saved_file_name)

    
    print("End of Conjugate")


# TRAINING ON IQ DATA
# input size of DL framework (batch size, slice size,I/Q) = (256, 256, 2)
if fusion is False and "iq" in args.input:
    single_modal_training(saved_file_name)
    print("Test Accuracies for DSSS Variant Detection: ", test_acc_anomaly_detection)
    print("Final test accuracy for DSSS Variant Detection: ", test_acc_anomaly_detection[int(args.epochs) - 1])
    

    print("End of I/Q")
    
    


# TRAINING ON BOTH CONJUGATE AND NON-CONJUGATE CSP FEATURES
if fusion is True and len(args.input) == 2 and 'nc' in args.input and 'c' in args.input:
    # if args.classifier == 'nn':
    single_modal_training(saved_file_name)
    print("Test Accuracies for DSSS Variant  Detection: ", test_acc_anomaly_detection)
    print("Final test accuracy for DSSS Variant Detection: ", test_acc_anomaly_detection[int(args.epochs) - 1])
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5)


    for epoch in range(int(args.epochs)):
        train_correct = 0 # Acc is calculated per epoch for training data
        train_total = 0  # Acc is calculated per epoch for training data
        test_correct = 0
        test_total = 0
        train_start_time = time.time()
        model.train()
        for i, (batch1, batch2, train_labels) in enumerate(training_generator):
                batch1, batch2, train_labels = batch1.float().to(device), batch2.float().to(device), train_labels.float().to(device)

                # Forward pass
                outputs, _ = model(batch1, batch2)
                loss = criterion(outputs, torch.max(train_labels, 1)[1])

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #scheduler.step(loss)

                # Getting Acc
                outputs = outputs.cpu().detach().numpy()
                labels = train_labels.cpu().detach().numpy()
                train_total += labels.shape[0]
                train_correct += (np.argmax(labels, axis=1) == np.argmax(outputs, axis=1)).sum().item()
        train_end_time = time.time()
        print("The learning rate is: ", optimizer.param_groups[0]['lr'])
        model.eval()

        
        # To create the t-SNE plots
        features_tsne = np.zeros(1)
        y_tsne = np.zeros(1)
        test_batch_count = 0
        
        # Test
        test_start_time = time.time()
        for test_batch1, test_batch2, test_labels in test_generator:
            test_batch1, test_batch2, test_labels = test_batch1.float().to(device), test_batch2.float().to(
                device), test_labels.float().to(device)
            test_output, latent_features = model(test_batch1, test_batch2)

            test_output = test_output.cpu().detach().numpy()
            test_labels = test_labels.squeeze().cpu().detach().numpy()
            
            print("Sanity check: ", np.argmax(test_labels, axis=1),  np.argmax(test_output, axis=1))

            # CALCULATING THE TEST ACCURACY
            test_total += test_labels.shape[0]
            test_correct += (np.argmax(test_labels, axis=1) == np.argmax(test_output, axis=1)).sum().item()
            
        
             ############################ t-SNE PLOT ##############################
            if test_batch_count == 0:
              features_tsne = latent_features.cpu().detach().numpy()
              y_tsne = np.argmax(test_output, axis=1)
            else:
              features_tsne = np.concatenate ((features_tsne, latent_features.cpu().detach().numpy()), axis = 0)
              y_tsne = np.concatenate ((y_tsne, np.argmax(test_output, axis=1)), axis = 0)
            
            test_batch_count += 1
            ##############################################################
        test_acc_anomaly_detection.append(100 * test_correct / test_total)
        test_end_time = time.time()
        print("Time to train one epoch: ", (train_end_time - train_start_time), " seconds")
        print("Time to test one sample: ", (test_end_time - test_start_time)/args.bs, " seconds" )
        # print loss and accuracies
        if (epoch % 1 == 0): print('epoch {}, loss {} train acc {} test acc {}'.format(epoch, loss.data, (100 * train_correct / train_total), (100 * test_correct / test_total)))

    # Plotting the features in t-SNE format 
    tsne_plot_generator(features_tsne, y_tsne, saved_file_name)
    torch.save(model, args.model_folder + '/' + saved_file_name + '.pt')
    if args.retrain: torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, args.model_folder + '/' + saved_file_name + '.pth')



# TRAINING ON CONJUGATE CSP FEATURES AND IQ DATA
if fusion is True and len(args.input) == 2 and 'c' in args.input and 'iq' in args.input:
    two_modality_training(saved_file_name, xtrain_iq, xtrain_c, ytrain, xtest_iq, xtest_c, ytest)
    print("Test Accuracies for DSSS Variant  Detection: ", test_acc_anomaly_detection)
    print("Final test accuracy for DSSS Variant  Detection: ", test_acc_anomaly_detection[int(args.epochs) - 1])
    print("End of I/Q and conjugate")


# TRAINING ON NON-CONJUGATE CSP FEATURES AND IQ DATA
if fusion is True and len(args.input) == 2 and 'nc' in args.input and 'iq' in args.input:
    two_modality_training(saved_file_name, xtrain_iq, xtrain_nc, ytrain, xtest_iq, xtest_nc, ytest)
    print("Test Accuracies for  DSSS Variant Detection: ", test_acc_anomaly_detection)
    print("Final test accuracy for DSSS Variant Detection: ", test_acc_anomaly_detection[int(args.epochs) - 1])
    print("End of I/Q and non-conjugate")




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
    #criterion = nn.NLLLoss()
    # Loss and optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


    for epoch in range(int(args.epochs)):
        train_correct = 0 # Acc is calculated per epoch for training data
        train_total = 0  # Acc is calculated per epoch for training data
        test_correct = 0
        test_total = 0
        train_start_time = time.time()
        model.train()
        for i, (batch1, batch2, batch3, train_labels) in enumerate(training_generator):
                batch1, batch2, batch3, train_labels = batch1.float().to(device), batch2.float().to(device), batch3.float().to(device), train_labels.float().to(device)

                # Forward pass
                outputs, _= model(batch1, batch2, batch3)
                loss = criterion(outputs, torch.max(train_labels, 1)[1])

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(loss)

                # Getting Acc
                outputs = outputs.cpu().detach().numpy()
                labels = train_labels.cpu().detach().numpy()
                train_total += labels.shape[0]
                train_correct += (np.argmax(labels, axis=1) == np.argmax(outputs, axis=1)).sum().item()
        #scheduler.step() # implementing decrease LR in plateau
        print("The learning rate is: ", optimizer.state_dic('param_groups ')[0]['lr'])
        model.eval()

    
        # To create the t-SNE plots
        features_tsne = np.zeros(1)
        y_tsne = np.zeros(1)
        test_batch_count = 0
        test_start_time = time.time()
        for test_batch1, test_batch2, test_batch3, test_labels in test_generator:
            test_batch1, test_batch2, test_batch3, test_labels = test_batch1.float().to(device), test_batch2.float().to(device), test_batch3.float().to(device), test_labels.float().to(device)
            test_output, latent_features = model(test_batch1, test_batch2, test_batch3)

            test_output = test_output.cpu().detach().numpy()
            test_labels = test_labels.squeeze().cpu().detach().numpy()
            
            #print("**TEST1**:", features_tsne.shape)
            #features_tsne = np.concatenate((test_labels, test_output))
            #print("**TEST2**:", features_tsne.shape)

            print("Sanity check: ", np.argmax(test_labels, axis=1),  np.argmax(test_output, axis=1))

            # CALCULATING THE TEST ACCURACY
            test_total += test_labels.shape[0]
            test_correct += (np.argmax(test_labels, axis=1) == np.argmax(test_output, axis=1)).sum().item()
            
            ############################ t-SNE PLOT ##############################
            if test_batch_count == 0:
              features_tsne = latent_features.cpu().detach().numpy()
              y_tsne = np.argmax(test_output, axis=1)
            else:
              features_tsne = np.concatenate ((features_tsne, latent_features.cpu().detach().numpy()), axis = 0)
              y_tsne = np.concatenate ((y_tsne, np.argmax(test_output, axis=1)), axis = 0)
            
            test_batch_count += 1
            ##############################################################
            
        test_acc_anomaly_detection.append(100 * test_correct / test_total)
        test_end_time = time.time()
        print("Time to train one epoch: ", (train_end_time - train_start_time), " seconds")
        print("Time to test one sample: ", (test_end_time - test_start_time)/args.bs, " seconds" )

        # print loss and accuracies
        if (epoch % 1 == 0): print('epoch {}, loss {} train acc {} test acc {}'.format(epoch, loss.data, (100 * train_correct / train_total), (100 * test_correct / test_total)))

    # Plotting the features in t-SNE format 
    tsne_plot_generator(features_tsne, y_tsne, saved_file_name)
    
    torch.save(model, args.model_folder + '/' + saved_file_name + '.pt')
    if args.retrain: torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, args.model_folder + '/' + saved_file_name + '.pth')



# TRAINING ON CONJUGATE CSP FEATURES AND IQ DATA
if fusion is True and len(args.input) == 3 and 'c' in args.input and 'nc' in args.input and 'iq' in args.input:
    three_modality_training(saved_file_name, xtrain_iq, xtrain_c, xtrain_nc, ytrain, xtest_iq, xtest_c, xtest_nc, ytest)
    print("Test Accuracies for DSSS Variant Detection: ", test_acc_anomaly_detection)
    print("Final test accuracy for DSSS Variant Detection: ", test_acc_anomaly_detection[int(args.epochs) - 1])
    print("End of I/Q, conjugate and non-conjugate")
    






# Calculating total execution time
end_time = time.time()  # Taking end time to calculate overall execution time
print("\n Total Execution Time (Minutes): ")
print(((end_time - start_time) / 60))
