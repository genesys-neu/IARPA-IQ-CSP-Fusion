"""
@author: debashri
This file contains different neural network models used in main.py
FeatureNet: NN model to conjugate and non-conjugate features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



# This is for 2-class classification
class FeatureNet(nn.Module):
    def __init__(self, input_dim, output_dim, fusion='ultimate'):
        super(FeatureNet, self).__init__()
        # self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=2, padding="same")
        # self.conv2 = nn.Conv1d(256, 256, kernel_size=2, padding="same")
        # self.pool = nn.MaxPool1d(2, padding=1)

        self.hidden1 = nn.Linear(input_dim, 256) # 256
        self.hidden2 = nn.Linear(256, 256)
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
        x = self.sigmoid(self.hidden1(x))
        x = self.sigmoid(self.hidden2(x))
        # x = self.drop(x)
        if self.fusion == 'penultimate':
            x = self.sigmoid(self.hidden2(x))
        else:
            x = self.softmax(self.out(x))
        return x
