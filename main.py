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


# Passing different arguments
parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--data_folder', help='Location of the data directory', type=str, default= 'D:/IARPA_DATA/')
parser.add_argument('--input', nargs='*', default=['nc', 'c'],choices = ['iq', 'nc', 'c'],
help='Which data to use as input. Select from: raw IQ data, non-conjugate features, conjugate features.')
parser.add_argument('--feature_options', nargs='*', default=[0, 1, 2, 3],choices = [0, 1, 2, 3],
help='Which features to use from the conjugate and non-conjugate files.')
parser.add_argument('--num_classes', default=7, type=int, help='Number of classes for classification.')

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
help='The block length of which 50/50 data is used for train/test. We use 100 percent data for training for rest of the two block lengths.')

# Train and test on specific SNR values
parser.add_argument('--random_snr_test', type=str2bool, help='Perform training on features from one SNR level and other.', default=True)
parser.add_argument('--random_snr', nargs='*', default=[0],choices = [0, 5, 10],
help='The SNR to use for random train/test.')
parser.add_argument('--training_snr', nargs='*', default=[10],choices = [0, 5, 10],
help='Which SNR to use for training.') # use this argument only if you set the 'random_snr_test' argument as False
parser.add_argument('--testing_snr', nargs='*', default=[0],choices = [0, 5, 10],
help='Which SNR to use for training.') # use this argument only if you set the 'random_snr_test' argument as False
parser.add_argument('--dsss_sir', nargs='*', default=[10],choices = [0, 5, 10],
help='Which SIRs of DSSS to use for train/test.') # this argument will be used always #0: less challenging, 10: more challenging


#Neural Network Parameters
parser.add_argument('--lr', default=0.0001, type=float,help='learning rate for Adam optimizer',)
parser.add_argument('--bs',default=32, type=int,help='Batch size') # 32
parser.add_argument('--epochs', default=50, type = int, help='Specify the epochs to train')
parser.add_argument('--shuffle', help='shuffle or not', type=str2bool, default =True)
parser.add_argument('--normalize', help='normalize or not', type=str2bool, default =False)
parser.add_argument('--id_gpu', default=0, type=int, help='which gpu to use.')
parser.add_argument('--model_folder', help='Location of the data directory', type=str, default= 'D:/IARPA_DATA/Saved_Models/')

# Not being used so far
parser.add_argument('--restore_models', type=str2bool, help='Load single modality trained weights', default=False)
parser.add_argument('--retrain', type=str2bool, help='Retrain the model on loaded weights', default=True)
parser.add_argument('--fusion_layer', type=str, help='Assign the layer name where the fusion to be performed.', default='penultimate')
parser.add_argument('--incremental_fusion', type=str2bool, help='Perform the incremental fusion or not. By default it will be aggregative fusion.', default=False)
parser.add_argument('--state_fusions', type=str2bool, help='Perform the state-of-the-art fusions.', default=True)
parser.add_argument('--fusion_techniques', nargs='*', default=['mi'],choices = ['mi', 'lr_tensor', 'concat'],
help='Specify if you want to use any of the state-of-the-art fusion techniques such as: multiplicative interactions, low rank tensor.')
parser.add_argument('--evaluate_only', type=str2bool, help='Only evaluate the model, no training will be done.', default=False)

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
if 'nc' in args.input:
    if args.random_test == True: # 80/20 train/test on specific block lengths
        if args.random_snr_test == True:
            inputs_nc, labels_nc = read_file.read_processed_feat(args.data_folder, 'nc', args.feature_options, args.random_test_blocks, args.num_classes, args.random_snr, args.dsss_sir)
            print("Shapes of the non-conjugate features and labels: ", inputs_nc.shape, labels_nc.shape)
        else: # if we train on one SNR and test on other snrs
            inputs_train_nc, labels_train_nc = read_file.read_processed_feat(args.data_folder, 'nc', args.feature_options,
                                                                 args.random_test_blocks, args.num_classes,
                                                                 args.training_snr, args.dsss_sir)
            inputs_test_nc, labels_test_nc = read_file.read_processed_feat(args.data_folder, 'nc',
                                                                              args.feature_options,
                                                                              args.random_test_blocks, args.num_classes,
                                                                              args.testing_snr, args.dsss_sir)
            print("Shapes of the non-conjugate features and labels: ", inputs_train_nc.shape, labels_train_nc.shape, inputs_test_nc.shape, labels_test_nc.shape)
    elif args.optimal_test==True: # test on 50% data of specific block length and train on rest of 50% of that bloack length, along with rest two block length
        rest_block_lengths = [x for x in all_block_lengths if x not in args.optimal_test_blocks]
            # all_block_lengths - args.optimal_test_blocks
        # print(rest_block_lengths)
        if args.random_snr_test == True:
            inputs_nc, labels_nc = read_file.read_processed_feat(args.data_folder, 'nc', args.feature_options,
                                                                 rest_block_lengths, args.num_classes, args.random_snr, args.dsss_sir)
            input_test_nc, labels_test_nc = read_file.read_processed_feat(args.data_folder, 'nc', args.feature_options,
                                                               args.optimal_test_blocks, args.num_classes,  args.random_snr, args.dsss_sir)
            xhalf_train_nc, xtest_nc, yhalf_train_nc, ytest_nc = train_test_split(input_test_nc, labels_test_nc, test_size=0.5, random_state=42)
        else:
            inputs_nc, labels_nc = read_file.read_processed_feat(args.data_folder, 'nc', args.feature_options,
                                                                 rest_block_lengths, args.num_classes, args.training_snr, args.dsss_sir)
            xhalf_train_nc, yhalf_train_nc = read_file.read_processed_feat(args.data_folder, 'nc', args.feature_options,
                                                                          args.optimal_test_blocks, args.num_classes,
                                                                          args.training_snr, args.dsss_sir) # training the
            xtest_nc, ytest_nc = read_file.read_processed_feat(args.data_folder, 'nc', args.feature_options,
                                                                 args.optimal_test_blocks, args.num_classes, args.testing_snr, args.dsss_sir)
            # xhalf_train_nc, xtest_nc, yhalf_train_nc, ytest_nc = train_test_split(input_test_nc, labels_test_nc, test_size=0.5,
            #                                                 random_state=42)
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
        xtrain_nc, ytrain_nc = read_file.read_processed_feat(args.data_folder, 'nc', args.feature_options,args.training_blocks, args.num_classes, training_snrs, args.dsss_sir)
        xtest_nc, ytest_nc = read_file.read_processed_feat(args.data_folder, 'nc', args.feature_options,args.testing_blocks, args.num_classes,  testing_snrs, args.dsss_sir)

        print("Shapes of the non-conjugate train, test features and labels: ", xtrain_nc.shape, xtest_nc.shape, ytrain_nc.shape, ytest_nc.shape)
    saved_file_name = 'non_conjugate'


if 'c' in args.input:
    if args.random_test == True:  # 80/20 train/test on specific block lengths
        if args.random_snr_test == True:
            inputs_c, labels_c = read_file.read_processed_feat(args.data_folder, 'c', args.feature_options,
                                                                 args.random_test_blocks, args.num_classes,
                                                                 args.random_snr, args.dsss_sir)
            print("Shapes of the conjugate features and labels: ", inputs_c.shape, labels_c.shape)
        else:  # if we train on one SNR and test on other snrs
            inputs_train_c, labels_train_c = read_file.read_processed_feat(args.data_folder, 'c',
                                                                             args.feature_options,
                                                                             args.random_test_blocks, args.num_classes,
                                                                             args.training_snr, args.dsss_sir)
            inputs_test_c, labels_test_c = read_file.read_processed_feat(args.data_folder, 'c',
                                                                           args.feature_options,
                                                                           args.random_test_blocks, args.num_classes,
                                                                           args.testing_snr, args.dsss_sir)
            print("Shapes of the conjugate features and labels: ", inputs_train_c.shape, labels_train_c.shape,
                  inputs_test_c.shape, labels_test_c.shape)
    elif args.optimal_test == True:  # test on 50% data of specific block length and train on rest of 50% of that bloack length, along with rest two block length
        rest_block_lengths = [x for x in all_block_lengths if x not in args.optimal_test_blocks]
        # all_block_lengths - args.optimal_test_blocks
        # print(rest_block_lengths)
        if args.random_snr_test == True:
            inputs_c, labels_c = read_file.read_processed_feat(args.data_folder, 'c', args.feature_options,
                                                                 rest_block_lengths, args.num_classes, args.random_snr, args.dsss_sir)
            input_test_c, labels_test_c = read_file.read_processed_feat(args.data_folder, 'c', args.feature_options,
                                                                          args.optimal_test_blocks, args.num_classes,
                                                                          args.random_snr, args.dsss_sir)
            xhalf_train_c, xtest_c, yhalf_train_c, ytest_c = train_test_split(input_test_c, labels_test_c,
                                                                                  test_size=0.5, random_state=42)
        else:
            inputs_c, labels_c = read_file.read_processed_feat(args.data_folder, 'c', args.feature_options,
                                                                 rest_block_lengths, args.num_classes,
                                                                 args.training_snr, args.dsss_sir)
            xhalf_train_c, yhalf_train_c = read_file.read_processed_feat(args.data_folder, 'c', args.feature_options,
                                                                           args.optimal_test_blocks, args.num_classes,
                                                                           args.training_snr, args.dsss_sir) # training the
            xtest_c, ytest_c = read_file.read_processed_feat(args.data_folder, 'c', args.feature_options,
                                                               args.optimal_test_blocks, args.num_classes,
                                                               args.testing_snr, args.dsss_sir)
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
        xtrain_c, ytrain_c = read_file.read_processed_feat(args.data_folder, 'c', args.feature_options,
                                                             args.training_blocks, args.num_classes, training_snrs, args.dsss_sir)
        xtest_c, ytest_c = read_file.read_processed_feat(args.data_folder, 'c', args.feature_options,
                                                           args.testing_blocks, args.num_classes, testing_snrs, args.dsss_sir)

        print("Shapes of the conjugate train, test features and labels: ", xtrain_c.shape, xtest_c.shape,
              ytrain_c.shape, ytest_c.shape)
    saved_file_name = 'conjugate'

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
from ModelHandler import NonConjugateNet, ConjugateNet, FeatureNet, FeatureFusion



#############################################################################################################################
# WORKING ON EACH INDIVIDUAL MODALITIES
#############################################################################################################################
def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

if fusion == False:
    if 'nc' in args.input:
        if args.random_test == True: # This will be triggered when doing train/test split randomly on whole data
            if args.random_snr_test == True:
                inputs = inputs_nc
                labels = labels_nc
                xtrain, xtest, ytrain, ytest = train_test_split(inputs, labels, test_size=0.2,
                                                                random_state=42)  # 80/20 is train/test size
            else:
                xtrain = xtrain_nc
                ytrain = ytrain_nc
                xtest = xtest_nc
                ytest = ytest_nc
        else: # This will be triggered when doing train/test on separate blocks and optimal testing
            xtrain = xtrain_nc
            ytrain = ytrain_nc
            xtest = xtest_nc
            ytest = ytest_nc
        model = NonConjugateNet(input_dim=xtrain.shape[1], output_dim=args.num_classes)


    if 'c' in args.input:
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
        model = ConjugateNet(input_dim=xtrain.shape[1], output_dim=args.num_classes)

    # INITIALIZING THE WEIGHT AND BIAS
    model.apply(weights_init)
else: # incase of fusion
    if 'c' in args.input and 'nc' in args.input:
        if args.random_test == True: # This will be triggered when doing train/test split randomly on whole data
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
        saved_file_name = 'conjugate_non_conjugate'


#################### NORMALIZE THE X DATA #######################
if args.normalize == True:
    xtrain_all = np.concatenate((xtrain, xtest))
    standard = preprocessing.StandardScaler().fit(xtrain_all)  # Normalize the data with zero mean and unit variance for each column
    xtrain = standard.transform(xtrain)
    xtest = standard.transform(xtest)
############### END OF NORMAIZATION ################


# print("All types: ", type(xtrain), type(xtest), type(ytrain), type(ytest))
print("XTRAIN AND XTEST SHAPE:", xtrain.shape, xtest.shape)
print("YTRAIN AND YTEST SHAPE:", ytrain.shape, ytest.shape)

print("The total number of occurances for different classes and total rows are: ")
print("In train set: ", ytrain.sum(axis=0))
print("In train set total number of anomalous (LTE+DSSS) and clean (LTE) signals are: ", ytrain.sum(axis=0)[0], (ytrain.shape[0]-ytrain.sum(axis=0)[0]))
print("In test set: ", ytest.sum(axis=0))
print("In test set total number of anomalous (LTE+DSSS) and clean (LTE)  signals are: ", ytest.sum(axis=0)[0], (ytest.shape[0]-ytest.sum(axis=0)[0]))



## WHEN SAVING WEIGHTS FOR PENULTIMATE LAYER
if args.fusion_layer == 'penultimate':  saved_file_name = saved_file_name + "_penultimate"

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    model.cuda()
else:
    device = torch.device("cpu")

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


############################################################################################################
#############################    WORKING ON SINGLE MODAL ARCHITECTURES #####################################
############################################################################################################
def single_modal_training(saved_file_name, optimizer_name = 'adam'):

    # loading the data
    # Parameters
    params = {'batch_size': args.bs,
              'shuffle': True,
              'num_workers': 0,
              'drop_last': True}

    training_set = data_loader('train')
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    test_set = data_loader('test')
    test_generator = torch.utils.data.DataLoader(test_set, **params)

    # setting up the loss function
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

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
        model.train()
        # print("Working on epoch ", epoch)
        for train_batch, train_labels in training_generator:
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
            # train_correct += (np.argmax(labels, axis=1) == np.argmax(outputs, axis=1)).sum().item()
            # print("In epoch ", epoch, " Train Labels: ", labels)
            # print("In epoch ", epoch, " Train Outputs: ", outputs)
            train_correct_anomaly += (labels[:, 0] == outputs[:, 0]).sum()
            # train_correct_all += (all(labels[:, 1:] == np.round(outputs)[:, 1:])).sum()
            for i in range(labels.shape[0]):
                train_correct_mod += int(all(labels[i, 1:3] == outputs[i, 1:3]))
            for i in range(labels.shape[0]):
                train_correct_gain += int(all(labels[i, 3:8] == outputs[i, 3:8]))
            for i in range(labels.shape[0]):
                train_correct_all += int(all(labels[i, :] == outputs[i, :]))
            # print("train_correct and train_correct_all: ", train_correct, train_correct_all)
            # print(" Testing: ",labels[0, 1:], outputs[0, 1:], ((labels[0, 1:] == np.round(outputs)[0, 1:]).sum()), (labels[0, 1:] == np.round(outputs)[0, 1:]), int(all(labels[0, 1:] == np.round(outputs)[0, 1:])))

        model.eval()

        test_start_time = time.time()
        # Test
        for test_batch, test_labels in test_generator:
            test_batch, test_labels = test_batch.float().to(device), test_labels.float().to(device)
            test_output = model(test_batch)

            test_output = test_output.cpu().detach().numpy()
            test_labels = test_labels.squeeze().cpu().detach().numpy()

            # CALCULATING THE TEST ACCURACY
            test_total +=test_labels.shape[0]
            test_correct_anomaly += (test_labels[:, 0] == test_output[:, 0]).sum()
            for i in range(test_labels.shape[0]):
                test_correct_mod += int(all(test_labels[i, 1:3] == test_output[i, 1:3]))
            for i in range(test_labels.shape[0]):
                test_correct_gain += int(all(test_labels[i, 3:8] == test_output[i, 3:8]))
            for i in range(test_labels.shape[0]):
                test_correct_all += int(all(test_labels[i, :] == test_output[i, :]))
        test_acc_anomaly_detection.append(100 * test_correct_anomaly / test_total)
        test_acc_anomaly_mod.append(100 * test_correct_mod / test_total)
        test_acc_anomaly_gain.append(100 * test_correct_gain / test_total)
        test_acc_anomaly_type.append(100 * test_correct_all / test_total)
        test_end_time = time.time()
        print("Time to test one sample: ", (test_end_time - test_start_time)/args.bs, " seconds" )

        # print loss and accuracies
        if (epoch % 1 == 0): print('epoch {}, loss {} train acc of anomaly detection {} test acc of anomaly detection {}'.format(epoch, loss.data, (100 * train_correct_anomaly / train_total), (100 * test_correct_anomaly / test_total)))
        if (epoch % 1 == 0): print('epoch {}, loss {} train acc of anomaly mod {} test acc of anomaly mod {}'.format(epoch, loss.data, (100 * train_correct_mod / train_total), (100 * test_correct_mod / test_total)))
        if (epoch % 1 == 0): print('epoch {}, loss {} train acc of anomaly gain {} test acc of anomaly gain {}'.format(epoch, loss.data, (100 * train_correct_gain / train_total), (100 * test_correct_gain / test_total)))
        if (epoch % 1 == 0): print('epoch {}, loss {} train acc of anomaly type {} test acc of anomaly type {}'.format(epoch,loss.data, (100 * train_correct_all / train_total),(100 * test_correct_all / test_total)))



    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, args.model_folder +"/"+saved_file_name+'.pth')

    torch.save(model, args.model_folder + '/' + saved_file_name + '.pt') # saving the whole model


if fusion is False and "nc" in args.input and args.evaluate_only == False:
    single_modal_training(saved_file_name, 'adam')
    print("Test Accuracies for LTE and DSSS Detection: ", test_acc_anomaly_detection)
    print("Test Accuracies for DSSS Type Detection: ", test_acc_anomaly_type)
    print("Final test accuracy for LTE and DSSS Detection: ", test_acc_anomaly_detection[int(args.epochs) - 1])
    print("Final test accuracy for DSSS Type Detection: ", test_acc_anomaly_type[int(args.epochs) - 1])
    print("End of Non-Conjugate")

if fusion is False and "c" in args.input and args.evaluate_only == False:
    single_modal_training(saved_file_name, 'adam')
    print("Test Accuracies for LTE and DSSS Detection: ", test_acc_anomaly_detection)
    print("Test Accuracies for DSSS Type Detection: ", test_acc_anomaly_type)
    print("Final test accuracy for LTE and DSSS Detection: ", test_acc_anomaly_detection[int(args.epochs) - 1])
    print("Final test accuracy for DSSS Type Detection: ", test_acc_anomaly_type[int(args.epochs) - 1])
    print("End of Conjugate")




############################################################################################################
#############################    WORKING ON FUSION ARCHITECTURES #####################################
############################################################################################################
def two_modality_training(saved_file_name, xtrain_mod1, xtrain_mod2, ytrain, xval_mod1, xval_mod2, yval, xtest_mod1, xtest_mod2, ytest):
    params = {'batch_size': args.bs,
              'shuffle': True,
              'num_workers': 0,
              'drop_last': True}

    training_set = fusion_two_data_loader(xtrain_mod1, xtrain_mod2, ytrain)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    validation_set = fusion_two_data_loader(xval_mod1, xval_mod2, yval)
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)

    testing_set = fusion_two_data_loader(xtest_mod1, xtest_mod2, ytest)
    test_generator = torch.utils.data.DataLoader(testing_set, **params)
    #model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    # Loss and optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)


    for epoch in range(int(args.epochs)):
        train_correct = 0 # Acc is calculated per epoch for training data
        train_total = 0  # Acc is calculated per epoch for training data
        test_correct = 0
        test_total = 0
        for i, (batch1, batch2, train_labels) in enumerate(training_generator):
                batch1, batch2, train_labels = batch1.float().to(device), batch2.float().to(device), train_labels.float().to(device)

                # Forward pass
                outputs, hidden_layers = model(batch1, batch2)
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


if fusion is True and len(args.input) == 2 and 'nc' in args.input and 'c' in args.input and args.evaluate_only == False:
    single_modal_training(saved_file_name, 'adam')
    print("Test Accuracies for LTE and DSSS Detection: ", test_acc_anomaly_detection)
    print("Test Accuracies for DSSS (BPSK, QPSK) Detection: ", test_acc_anomaly_mod)
    print("Test Accuracies for DSSS Spreading Gain Detection: ", test_acc_anomaly_gain)
    print("Test Accuracies for DSSS Type Detection: ", test_acc_anomaly_type)
    print("Final test accuracy for LTE and DSSS Detection: ", test_acc_anomaly_detection[int(args.epochs) - 1])
    print("Final test accuracy for DSSS (BPSK, QPSK) Detection: ", test_acc_anomaly_mod[int(args.epochs) - 1])
    print("Final test accuracy for DSSS Spreading Gain Detection: ", test_acc_anomaly_gain[int(args.epochs) - 1])
    print("Final test accuracy for DSSS Type Detection: ", test_acc_anomaly_type[int(args.epochs) - 1])
    print("End of Non-Conjugate and Conjugate")


#########################################################################
######################      EVALUATE THE MODES     ######################
#########################################################################
def evaluation(saved_file_name):

    # loading the data
    # Parameters
    params = {'batch_size': args.bs,
              'shuffle': True,
              'num_workers': 0,
              'drop_last': True}

    test_set = data_loader('test')
    test_generator = torch.utils.data.DataLoader(test_set, **params)
    model = torch.load(saved_file_name)
    model.cuda()

    for epoch in range(int(args.epochs)):
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        print('-' * 10)
        test_correct = [0, 0, 0, 0, 0]
        test_total = 0
        model.eval()
        # Test
        test_epoch_outputs = []
        test_epoch_labels = []
        for test_batch, test_labels in test_generator:
            test_batch, test_labels = test_batch.float().to(device), test_labels.float().to(device)
            test_output = model(test_batch)

            test_output = test_output.cpu().detach().numpy()
            test_labels = test_labels.squeeze().cpu().detach().numpy()

            test_epoch_outputs.append(1 * test_output)
            test_epoch_labels.append(1 * test_labels)
        test_epoch_outputs_matrix = np.concatenate(test_epoch_outputs, axis=0)
        test_epoch_label_matrix = np.concatenate(test_epoch_labels, axis=0)
        acc = 0



if args.evaluate_only == True:
    if 'image' in args.input and fusion == False:
        model_file_name = args.model_folder + '/best/nusc_img_visi.pt'
    if 'radar' in args.input and fusion == False:
        model_file_name = args.model_folder + '/best/radar.pt'
    evaluation(model_file_name)
    pass


# Calculating total execution time
end_time = time.time()  # Taking end time to calculate overall execution time
print("\n Total Execution Time (Minutes): ")
print(((end_time - start_time) / 60))
