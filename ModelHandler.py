"""
@author: debashri
This file contains different neural network models used in main.py
NonConjugateNet: CNN model for non-conjugate features
ConjugateNet: CNN model for conjugate features
FeatureFusion: CNN model to fuse two modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# CNN based model for non-conjugate features as unimodal network
class NonConjugateNet(nn.Module):
    def __init__(self, input_dim, output_dim, fusion ='ultimate'):
            super(NonConjugateNet, self).__init__()
            self.conv1 = nn.Conv1d(input_dim, 20, kernel_size=2, padding="same")
            self.conv2 = nn.Conv1d(20, 20, kernel_size=2, padding="same")
            self.pool = nn.MaxPool1d(2, padding=1)

            self.hidden1 = nn.Linear(20, 1024)
            self.hidden2 = nn.Linear(1024, 512)
            self.hidden3 = nn.Linear(512, 256)
            self.hidden4 = nn.Linear(256, 64)
            self.out = nn.Linear(256, output_dim)  # 128
            #######################
            self.drop = nn.Dropout(0.25)
            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax()
            self.fusion = fusion

    def forward(self, x):
            x = torch.reshape(x, (x.shape[0], x.shape[1], 1))
            # FOR CNN BASED IMPLEMENTATION
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)

            x = self.relu(self.conv2(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            # print("shape", x.shape)
            x = self.relu(self.hidden1(x))
            x = self.drop(x)
            x = self.relu(self.hidden2(x))
            x = self.drop(x)
            x = self.relu(self.hidden3(x))
            x = self.drop(x)
            if self.fusion == 'penultimate':
                x = self.sigmoid(self.hidden4(x))
            else:

                # x = self.softmax(self.out(x))
                # x = self.relu(self.out(x))  # no softmax: CrossEntropyLoss()
                x = self.sigmoid(self.out(x))
            return x


# CNN based model for conjugate features as unimodal network
class ConjugateNet(nn.Module):
    def __init__(self, input_dim, output_dim, fusion='ultimate'):
        super(ConjugateNet, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 20, kernel_size=2, padding="same")
        self.conv2 = nn.Conv1d(20, 20, kernel_size=2, padding="same")
        self.pool = nn.MaxPool1d(2, padding=1)

        self.hidden1 = nn.Linear(20, 1024)
        self.hidden2 = nn.Linear(1024, 512)
        self.hidden3 = nn.Linear(512, 256)
        self.hidden4 = nn.Linear(256, 64)
        self.out = nn.Linear(256, output_dim)  # 128
        #######################
        self.drop = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.fusion = fusion

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], x.shape[1], 1))
        # FOR CNN BASED IMPLEMENTATION
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = self.relu(self.conv2(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        # print("shape", x.shape)
        x = self.relu(self.hidden1(x))
        x = self.drop(x)
        x = self.relu(self.hidden2(x))
        x = self.drop(x)
        x = self.relu(self.hidden3(x))
        x = self.drop(x)
        if self.fusion == 'penultimate':
            x = self.sigmoid(self.hidden4(x))
        else:

            # x = self.softmax(self.out(x))
            # x = self.relu(self.out(x))  # no softmax: CrossEntropyLoss()
            x = self.sigmoid(self.out(x))
        return x


# CNN based model for conjugate features as unimodal network
class FeatureNet(nn.Module):
    def __init__(self, input_dim, output_dim, fusion='ultimate'):
        super(FeatureNet, self).__init__()
        # self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=2, padding="same")
        # self.conv2 = nn.Conv1d(256, 256, kernel_size=2, padding="same")
        # self.pool = nn.MaxPool1d(2, padding=1)

        self.hidden1 = nn.Linear(input_dim, 512) # 256
        self.hidden2 = nn.Linear(2048, 1024)
        self.hidden3 = nn.Linear(1024, 512)
        self.hidden4 = nn.Linear(512, 512)
        self.hidden5 = nn.Linear(512, 512)
        self.hidden6 = nn.Linear(512, 256)
        self.hidden7 = nn.Linear(256, 64)
        self.out = nn.Linear(256, output_dim)  # 128
        #######################
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.fusion = fusion

    def forward(self, x):
        # print("shape", x.shape)
        x = self.hidden1(x)
        x = self.drop(x)
        x = self.hidden6(x)
        x = self.drop(x)
        if self.fusion == 'penultimate':
            x = self.sigmoid(self.hidden7(x))
        else:
            x = self.sigmoid(self.out(x))
        return x

# CNN based model does not work
class FeatureNetCNN(nn.Module):
    def __init__(self, input_dim, output_dim, fusion ='ultimate'):
        super(FeatureNetCNN, self).__init__()

         # BEST ONESS  - GIVING 1 AUC FOR  BIN-BASED (CNN BASED)
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=2, stride = 1, padding='same')  # 256 #32 # padding
        self.conv2 = nn.Conv1d(256, 128, kernel_size=1, padding='same')   # 256 #32
        self.pool = nn.MaxPool1d(2)

        self.hidden1 = nn.Linear(128, 512)  # 32 #512 #input_dim # 86272 scratch #345344 from model
        self.out = nn.Linear(512, output_dim)  # 128
        #######################

        self.drop = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.fusion = fusion


    def forward(self, x):

        # print("shape1:", x.shape)
        x = torch.reshape(x, (x.shape[0], x.shape[1], 1))
        # print("shape2:", x.shape)
        # FOR CNN BASED IMPLEMENTATION
        x = self.conv1(x)
        # x = self.pool(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        # END OF IT

        x = self.sigmoid(self.hidden1(x)) # BEST ONE
        if self.fusion == 'penultimate':
            return x
        else:

            x = self.sigmoid(self.out(x)) # [batch_size, 5] # use sigmoid
            return x

# CNN BASED FUSION CLASS - TWO MODALITIES
class FeatureFusion(nn.Module):
    def __init__(self, modelA, modelB, nb_classes=5, fusion = 'ultimate'):
        super(FeatureFusion, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.relu = nn.ReLU()

        self.drop = nn.Dropout(0.5)

        # for radar: 512; for seismic: 90
        self.hidden1 = nn.Linear(81792, 512) # 14592 WHEN training from scratch and incremental fusion; 81792 when restoring the models
        self.hidden2 = nn.Linear(512, 256)
        self.hidden3 = nn.Linear(256, nb_classes)
        self.classifier = nn.Linear(2*nb_classes, nb_classes)
        self.fusion = fusion

        self.conv1 = nn.Conv1d(1, 256, kernel_size=4, stride=2)  # 256 #32 # padding # 7
        self.conv2 = nn.Conv1d(256, 128, kernel_size=1)  # 256 #32 # 5
        self.pool1 = nn.MaxPool1d(5)  # 2
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=4, stride=2)  # 256 #32 # padding # 7

    def forward(self, x1, x2): # x1:acoustic; x2:radar
        x1 = self.modelA(x1)  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)

        x2 = self.modelB(x2)
        x2 = x2.view(x2.size(0), -1)

        x = torch.cat((x1, x2), dim=1)

        if self.fusion == 'penultimate':
            x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool1(x)

            x = self.relu(self.conv3(x))
            x = self.relu(self.conv2(x))
            x = self.pool2(x)

            x = x.view(x.size(0), -1)
            x = self.hidden1(x)
            # x = self.drop(x)

            x = self.hidden2(x)
            # x = self.drop(x)

            x = self.hidden3(x)
            # x = self.drop(x)

            return x

        x = self.classifier(F.sigmoid(x)) # use relu
        return x
