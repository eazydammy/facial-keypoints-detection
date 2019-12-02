import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # CONVOLUTIONAL LAYERS
        # 1 square input image channel (grayscale) each, [32,64,128,256] 
        # output channels/feature maps, 10x10 square convolution kernel each
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        # DENSE LAYERS
        self.dense1 = nn.Linear(43264, 1024)
        self.dense2 = nn.Linear(1024, 512)
        self.dense3 = nn.Linear(512, 136)
        # outputs 136 values, 2 for each of the 68 keypoint (x, y) pairs
        #self.dense4 = nn.Linear(256, 136)
        
        
        # WEIGHT INITIALIZATION
        #layers = [self.conv1, self.conv2, self.conv3, self.conv3, self.dense1, self.dense2, self.dense3]
        #for layer in layers:
        #    I.uniform_(layer.weight.data, 0, 1)
        #    if layer.bias is not None:
        #        layer.bias.data.fill_(0)
        
        # POOL LAYERS
        # maxpool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # DROPOUT LAYERS
        self.conv1_drop = nn.Dropout(p=0.4)
        self.conv2_drop = nn.Dropout(p=0.4)
        self.conv3_drop = nn.Dropout(p=0.4)
        self.conv4_drop = nn.Dropout(p=0.4)
        self.dense1_drop = nn.Dropout(p=0.5)
        self.dense2_drop = nn.Dropout(p=0.5)
        self.dense3_drop = nn.Dropout(p=0.5)
        
        # BATCH NORMALIZATION
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        x = self.pool(self.conv1_drop(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool(self.conv2_drop(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool(self.conv3_drop(F.relu(self.bn3(self.conv3(x)))))
        x = self.pool(self.conv4_drop(F.relu(self.bn4(self.conv4(x)))))

        # flatten 3D tensor to 1D for dense layers
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = self.dense1_drop(F.relu(self.dense1(x)))
        x = self.dense2_drop(F.relu(self.dense2(x)))
        #x = self.dense3_drop(F.relu(self.dense3(x)))
        #x = self.dense4(x)
        x = self.dense3(x)
        
        return x