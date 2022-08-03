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
        self.hidden1 = nn.Linear(input_dim, 512, bias=True) # 256
        self.hidden2 = nn.Linear(512, 256, bias=True)
        self.hidden3 = nn.Linear(256, 256, bias=True)
        self.out = nn.Linear(256, output_dim, bias=True)  # 128
        
        self.one_layer = nn.Linear(input_dim, output_dim, bias=True)

        self.two_layer1 = nn.Linear(input_dim, 50, bias=True)
        self.two_layer2 = nn.Linear(50, output_dim, bias=True)
        #######################
        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.fusion = fusion

    def forward(self, input_x):
        #print("shape", input_x.shape)
        
        ### testing 
        x = self.hidden1(input_x)
        x = self.hidden2(x)
        latent_features = self.hidden3(x)
        if self.fusion == 'penultimate':
          out = self.sigmoid(latent_features)
          return out, latent_features
        else:
           x = self.sigmoid(self.out(latent_features))
        return x, latent_features
        ## end testing
        #out = self.relu(self.one_layer (input_x))
        #return out, input_x
        #########################
        #feature = self.two_layer1(input_x)
        #out = self.relu(self.two_layer2(feature))
        #return out, feature

class RFNet(nn.Module):
    def __init__(self, input_dim, output_dim, fusion ='ultimate'):
        super(RFNet, self).__init__()

        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=7,  padding='same')  # 256 #32 # padding
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, padding='same')  # 256 #32
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding='same')  # 256 #32
        self.pool = nn.MaxPool1d(2)

        self.hidden1 = nn.Linear(524288 , 512)
        self.hidden2 = nn.Linear(512, 256)
        self.hidden3 = nn.Linear(256, 256)
        self.hidden4 = nn.Linear(256, 256)
        self.hidden5 = nn.Linear(256, 512)
        self.out = nn.Linear(262144, output_dim)  # 128
        self.one_layer =nn.Linear(512, output_dim)
        self.two_layer1 =nn.Linear(262144, 512, bias=True) # 262144 (13 BL) 524288 (26 BL) 1048576 (52 BL) 512 (slicing)
        self.two_layer2 =nn.Linear(512, 256, bias=True) 
        self.two_layer3 =nn.Linear(256, output_dim, bias=True)
        #######################

        self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.fusion = fusion


    def forward(self, x):

        #x = torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1]))
        # FOR CNN BASED IMPLEMENTATION
            x = x.view(x.size(0), -1)
            #x = self.one_layer(x)
            x = self.two_layer1(x)
            x = self.two_layer2(x)
            x = self.relu(self.two_layer3(x))
            return x, x





# CNN based FUSION CLASS - TWO MODALITIES
class FeatureFusion(nn.Module): # USED
    def __init__(self, modelA, modelB, nb_classes=5, fusion = 'ultimate'):
        super(FeatureFusion, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.relu = nn.ReLU()

        self.drop = nn.Dropout(0.5)

        self.hidden1 = nn.Linear(384, 512) # 2944 for variant 
        self.hidden2 = nn.Linear(512, 512)
        self.hidden3 = nn.Linear(512, 256)
        self.hidden4 = nn.Linear(256, nb_classes)
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
            
            x = self.relu(self.conv3(x))
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            
            #x = self.relu(self.conv3(x))
            #x = self.relu(self.conv2(x))
            #x = self.pool2(x)

            x = x.view(x.size(0), -1)
            x = self.hidden1(x)
            # x = self.drop(x)

            x = self.hidden2(x)
            # x = self.drop(x)

            latent_features = self.hidden3(x)
            # x = self.drop(x)
            
            out = self.hidden4(latent_features)

            return out, latent_features

        x = self.classifier(F.sigmoid(x)) # use relu
        return x, _
####################################################################################################################


#################### MODELS WHICH ARE NOT BEING USED NOW #################################
cfg = {
    'AlexNet1D': [128, 'M', 128, 'M', 128, 'M', 128, 'M', 128, 'M'],
    'Baseline': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
}

class AlexNet1D(nn.Module): # USED
    def __init__(self, input_dim, output_dim, fusion ='ultimate'):
        super(AlexNet1D, self).__init__()
        self.features = self._make_layers(cfg['AlexNet1D'], input_dim)
        self.fusion_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 256), # Alexnet: 2097152 (52 block length) 1048576 (26 block length) 524288  (for 13 block length) 1024 (for slicing) Baseline: 4194304 (13 block length)
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 256), # Alexnet: 2097152 (52 block length) 1048576 (26 block length) 524288  (for 13 block length) 1024 (for slicing) Baseline: 4194304 (13 block length)
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

# CNN based model does not work
class FeatureNetCNN(nn.Module): # USED
    def __init__(self, input_dim, output_dim, fusion ='ultimate'):
        super(FeatureNetCNN, self).__init__()

        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=2, stride = 1, padding='same')  # 256 #32 # padding
        self.conv2 = nn.Conv1d(256, 128, kernel_size=1, padding='same')   # 256 #32
        self.conv3 = nn.Conv1d(128, 128, kernel_size=1, padding='same')
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

        #print("shape1:", x.shape)
        x = torch.reshape(x, (x.shape[0], x.shape[1], 1))
        # print("shape2:", x.shape)
        # FOR CNN BASED IMPLEMENTATION
        x = self.conv1(x)
        # print("shape2:", x.shape)
        # x = self.pool(x)
        x = self.conv2(x)
        #x = self.pool(x)

        # x = self.conv3(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        # END OF IT

        latent_features = self.hidden1(x) # BEST ONE
        if self.fusion == 'penultimate':
            out = self.relu(latent_features)
            return out, latent_features
        else:

            out = self.relu(self.out(latent_features)) # [batch_size, 5] # use sigmoid
            return out, latent_features

# CNN based FUSION CLASS - THREE MODALITIES
class FeatureFusionThree(nn.Module): # USED
    def __init__(self, modelA, modelB, modelC, nb_classes=5, fusion = 'ultimate'):
        super(FeatureFusionThree, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.relu = nn.ReLU()

        self.drop = nn.Dropout(0.5)


        self.hidden1 = nn.Linear(256, 512) #1920 for variant
        self.hidden2 = nn.Linear(512, 256)
        self.hidden3 = nn.Linear(256, nb_classes)
        self.classifier = nn.Linear(3*nb_classes, nb_classes)
        self.fusion = fusion

        self.conv1 = nn.Conv1d(1, 256, kernel_size=4, stride=2)  # 256 #32 # padding # 7
        self.conv2 = nn.Conv1d(256, 128, kernel_size=1)  # 256 #32 # 5
        self.pool1 = nn.MaxPool1d(5)  # 2
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=4, stride=2)  # 256 #32 # padding # 7
        self.softmax = nn.LogSoftmax()

    def forward(self, x1, x2, x3):
        x1, _ = self.modelA(x1)  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)

        x2, _ = self.modelB(x2)
        x2 = x2.view(x2.size(0), -1)

        x3, _ = self.modelC(x3)
        x3 = x3.view(x3.size(0), -1)

        x = torch.cat((x1, x2, x3), dim=1)

        if self.fusion == 'penultimate':
           # print("coming here...")
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



class CSPNet(nn.Module):
    def __init__(self, input_dim):
        super(CSPNet, self).__init__()
        self.hidden1 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = self.hidden1(x)
        return x




# CNN based model for non-conjugate features as unimodal network
class NonConjugateNet(nn.Module):
    def __init__(self, input_dim, output_dim, fusion ='ultimate'):
            super(NonConjugateNet, self).__init__()
            self.conv1 = nn.Conv1d(input_dim, 40, kernel_size=2, padding="same")
            self.conv2 = nn.Conv1d(40, 40, kernel_size=2, padding="same")
            self.pool = nn.MaxPool1d(2, padding=1)

            self.hidden1 = nn.Linear(80, 1024)
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
            # x = torch.reshape(x, (x.shape[0], x.shape[1], 1))
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
        # x = torch.reshape(x, (x.shape[0], x.shape[1], 1))
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


# This is for multi-label multi-class classification
class FeatureNet_7class(nn.Module):
    def __init__(self, input_dim, output_dim, fusion='ultimate'):
        super(FeatureNet_7class, self).__init__()
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


