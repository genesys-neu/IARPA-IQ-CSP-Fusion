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


#################### USED MODELS #################################
# This is for 2-class classification
class FeatureNet(nn.Module): # USED
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
        latent_features = self.sigmoid(self.hidden2(x))
        # x = self.drop(x)
        if self.fusion == 'penultimate':
            x = self.sigmoid(self.hidden2(latent_features))
        else:
            x = self.sigmoid(self.out(latent_features))
        return x, latent_features




cfg = {
    'AlexNet1D': [128, 'M', 128, 'M', 128, 'M', 128, 'M', 128, 'M'],
    'Baseline': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
}

class AlexNet1D(nn.Module): # USED
    def __init__(self, input_dim, output_dim, fusion ='ultimate'):
        super(AlexNet1D, self).__init__()
        self.features = self._make_layers(cfg['AlexNet1D'], input_dim)
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2097152 , 256), # 2097152 (52 block length) 1048576 (26 block length) 524288  (for 13 block length) 1024 (for slicing) 262144 (65536 block length)
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2097152 , 256), # 2097152 (52 block length) 1048576 (26 block length) 524288  (for 13 block length) 1024 (for slicing) 262144 (65536 block length)
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, output_dim),
        )
        self.fusion = fusion

    def forward(self, x):
        #print("test shape:", x.shape)
        x = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1]))
        # print("Input shape: ", x.shape)
        out = self.features(x)
        # print("X1: ", out.shape)
        out = out.view(out.size(0), -1)
        # print("X2: ", out.shape)
        latent_features = self.fusion_classifier(out) # to be used for t-SNE
        if self.fusion == 'penultimate':
            out = self.fusion_classifier(out)  # was sigmoid
        else:

            out = self.classifier(out)
        # return x

        # print("X3: ", out.shape)
        return out, latent_features

    def _make_layers(self, cfg, input_dim):
        layers = []
        in_channels = input_dim # it was 2
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv1d(in_channels, x, kernel_size=7, padding='same'),
                           nn.ReLU(inplace=True)]
                in_channels = x
                layers += [nn.Conv1d(in_channels, x, kernel_size=5, padding='same'),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)


# CNN based FUSION CLASS - TWO MODALITIES
class FeatureFusion(nn.Module): # USED
    def __init__(self, modelA, modelB, nb_classes=5, fusion = 'ultimate'):
        super(FeatureFusion, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.relu = nn.ReLU()

        self.drop = nn.Dropout(0.5)

        self.hidden1 = nn.Linear(2944, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.hidden3 = nn.Linear(256, nb_classes)
        self.classifier = nn.Linear(2*nb_classes, nb_classes)
        self.fusion = fusion

        self.conv1 = nn.Conv1d(1, 256, kernel_size=4, stride=2)  # 256 #32 # padding # 7
        self.conv2 = nn.Conv1d(256, 128, kernel_size=1)  # kernel_size= 2
        self.pool1 = nn.MaxPool1d(2)  # was 5
        self.pool2 = nn.MaxPool1d(2) # # was 2
        self.conv3 = nn.Conv1d(128, 256, kernel_size=4, stride=2)  # 256 #32 # padding # 7

    def forward(self, x1, x2):
        x1, _ = self.modelA(x1)  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)

        x2, _ = self.modelB(x2)
        x2 = x2.view(x2.size(0), -1)


        x = torch.cat((x1, x2), dim=1)

        if self.fusion == 'penultimate':
            #print("coming here...")
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

            latent_features = self.hidden2(x)
            # x = self.drop(x)

            x = self.hidden3(latent_features)
            # x = self.drop(x)

            return x, latent_features

        x = self.classifier(F.sigmoid(x)) # use relu
        return x, _
####################################################################################################################



