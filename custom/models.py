import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, ReLU, MaxPool2d, ConvTranspose2d, Sigmoid

class u_net(nn.Module):
    def __init__(self, nf=64, nc=1, img_h=256, img_w=256):
        super(u_net, self).__init__()
        
        self.nc = nc
        self.nf = nf
        
        self.conv11 = Conv2d(   nc,    nf, 3, 1, 1)
        self.conv12 = Conv2d(   nf,    nf, 3, 1, 1)
        
        self.conv21 = Conv2d(   nf,  2*nf, 3, 1, 1)
        self.conv22 = Conv2d( 2*nf,  2*nf, 3, 1, 1)
        
        self.conv31 = Conv2d( 2*nf,  4*nf, 3, 1, 1)
        self.conv32 = Conv2d( 4*nf,  4*nf, 3, 1, 1)
        
        self.conv41 = Conv2d( 4*nf,  8*nf, 3, 1, 1)
        self.conv42 = Conv2d( 8*nf,  8*nf, 3, 1, 1)
        
        self.conv51 = Conv2d( 8*nf, 16*nf, 3, 1, 1)
        self.conv52 = Conv2d(16*nf, 16*nf, 3, 1, 1)
        
        self.conv61 = Conv2d(16*nf, 8*nf, 3, 1, 1)
        self.conv62 = Conv2d( 8*nf, 8*nf, 3, 1, 1)
        
        self.conv71 = Conv2d( 8*nf, 4*nf, 3, 1, 1)
        self.conv72 = Conv2d( 4*nf, 4*nf, 3, 1, 1)
        
        self.conv81 = Conv2d( 4*nf, 2*nf, 3, 1, 1)
        self.conv82 = Conv2d( 2*nf, 2*nf, 3, 1, 1)
        
        self.conv91 = Conv2d( 2*nf,   nf, 3, 1, 1)
        self.conv92 = Conv2d(   nf,   nf, 3, 1, 1)
        
        self.unconv1 = ConvTranspose2d(16*nf, 8*nf, 4, 2, 1)
        self.unconv2 = ConvTranspose2d( 8*nf, 4*nf, 4, 2, 1)
        self.unconv3 = ConvTranspose2d( 4*nf, 2*nf, 4, 2, 1)
        self.unconv4 = ConvTranspose2d( 2*nf,   nf, 4, 2, 1)
        
        self.conv01 = Conv2d(nf, 1, 1)
        
        self.weights_init()
    
    def forward(self, x):
        nc = self.nc
        nf = self.nf
        
        conv1 = ReLU(True)(self.conv11(x))
        conv1 = ReLU(True)(self.conv12(conv1))
        pool1 = MaxPool2d(2)(conv1) # size (128, 128, 64)
        
        conv2 = ReLU(True)(self.conv21(pool1))
        conv2 = ReLU(True)(self.conv22(conv2))
        pool2 = MaxPool2d(2)(conv2) # size (64, 64, 128)
        
        conv3 = ReLU(True)(self.conv31(pool2))
        conv3 = ReLU(True)(self.conv32(conv3))
        pool3 = MaxPool2d(2)(conv3) # size (32, 32, 256)
        
        conv4 = ReLU(True)(self.conv41(pool3))
        conv4 = ReLU(True)(self.conv42(conv4))
        pool4 = MaxPool2d(2)(conv4) # size (16, 16, 512)
        
        conv5 = ReLU(True)(self.conv51(pool4))
        conv5 = ReLU(True)(self.conv52(conv5))
        
        upconv1 = torch.cat((conv4, ReLU(True)(self.unconv1(conv5))), dim=1)
        conv6 = ReLU(True)(self.conv61(upconv1))
        conv6 = ReLU(True)(self.conv62(conv6))
        
        upconv2 = torch.cat((conv3, ReLU(True)(self.unconv2(conv6))), dim=1)
        conv7 = ReLU(True)(self.conv71(upconv2))
        conv7 = ReLU(True)(self.conv72(conv7))
        
        upconv3 = torch.cat((conv2, ReLU(True)(self.unconv3(conv7))), dim=1)
        conv8 = ReLU(True)(self.conv81(upconv3))
        conv8 = ReLU(True)(self.conv82(conv8))
        
        upconv4 = torch.cat((conv1, ReLU(True)(self.unconv4(conv8))), dim=1)
        conv9 = ReLU(True)(self.conv91(upconv4))
        conv9 = ReLU(True)(self.conv92(conv9))
        
        output = Sigmoid()(self.conv01(conv9))
        return output

    def weights_init(self, init_func=torch.nn.init.kaiming_normal):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_func(m.weight)
#

#
''' keras version

import keras
from keras import backend as K
from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import metrics

class u_net(object):
    def __init__(self):
        pass
    
    def get_model(self, height=256, width=256, channel=1):
        im_input = Input((height, width, channel)) # size (256, 256, 1)
        wt_input = Input((height, width, channel))
        
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(im_input)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D()(conv1) # size (128, 128, 64)
        
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D()(conv2) # size (64, 64, 128)
        
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D()(conv3) # size (32, 32, 256)
        
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        pool4 = MaxPooling2D()(conv4) # size (16, 16, 512)
        
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        
        upconv1 = concatenate([Conv2DTranspose(512, 4, strides = 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5), conv4], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        
'''        
#







