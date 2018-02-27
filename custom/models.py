import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, ReLU, MaxPool2d, ConvTranspose2d

class u_net(nn.Module):
    def __init__(self, nf=64, nc=1, img_h=256, img_w=256):
        super(u_net, self).__init__()
        self.nc = nc
        self.nf = nf
        self.weights_init()
    
    def forward(self, x):
        nc = self.nc
        nf = self.nf
        
        conv1 = ReLU(True)(Conv2d(   nc,    nf, 3, 1, 1)(x))
        conv1 = ReLU(True)(Conv2d(   nf,    nf, 3, 1, 1)(conv1))
        pool1 = MaxPool2D(2)(conv1) # size (128, 128, 64)
        
        conv2 = ReLU(True)(Conv2d(   nf,  2*nf, 3, 1, 1)(pool1))
        conv2 = ReLU(True)(Conv2d(   nf,  2*nf, 3, 1, 1)(conv2))
        pool2 = MaxPool2D(2)(conv2) # size (64, 64, 128)
        
        conv3 = ReLU(True)(Conv2d( 2*nf,  4*nf, 3, 1, 1)(pool2))
        conv3 = ReLU(True)(Conv2d( 2*nf,  4*nf, 3, 1, 1)(conv3))
        pool3 = MaxPool2D(2)(conv3) # size (32, 32, 256)
        
        conv4 = ReLU(True)(Conv2d( 4*nf,  8*nf, 3, 1, 1)(pool3))
        conv4 = ReLU(True)(Conv2d( 4*nf,  8*nf, 3, 1, 1)(conv4))
        pool4 = MaxPool2D(2)(conv4) # size (16, 16, 512)
        
        conv5 = ReLU(True)(Conv2d( 8*nf, 16*nf, 3, 1, 1)(pool4))
        conv5 = ReLU(True)(Conv2d(16*nf, 16*nf, 3, 1, 1)(conv5))
        
        upconv1 = torch.cat(conv4, ReLU(True)(ConvTranspose2d(16*nf, 8*nf, 4, 2, 1)(conv5)))
        conv6 = ReLU(True)(Conv2d(16*nf, 8*nf, 3, 1, 1)(upconv1))
        conv6 = ReLU(True)(Conv2d( 8*nf, 8*nf, 3, 1, 1)(conv6))
        
        upconv2 = torch.cat(conv3, ReLU(True)(ConvTranspose2d( 8*nf, 4*nf, 4, 2, 1)(conv6)))
        conv7 = ReLU(True)(Conv2d( 8*nf, 4*nf, 3, 1, 1)(upconv2))
        conv7 = ReLU(True)(Conv2d( 4*nf, 4*nf, 3, 1, 1)(conv7))
        
        upconv3 = torch.cat(conv2, ReLU(True)(ConvTranspose2d( 4*nf, 2*nf, 4, 2, 1)(conv7)))
        conv8 = ReLU(True)(Conv2d( 4*nf, 2*nf, 3, 1, 1)(upconv3))
        conv8 = ReLU(True)(Conv2d( 2*nf, 2*nf, 3, 1, 1)(conv8))
        
        upconv4 = torch.cat(conv1, ReLU(True)(ConvTranspose2d( 2*nf,   nf, 4, 2, 1)(conv8)))
        conv9 = ReLU(True)(Conv2d( 2*nf,   nf, 3, 1, 1)(upconv4))
        conv9 = ReLU(True)(Conv2d(   nf,   nf, 3, 1, 1)(conv9))
        
        output = Sigmoid()(Conv2D(nf, 2, 1)(conv9))
        return output

    def weights_init(self, init_func=torch.nn.init.kaiming_normal):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_func(m)
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








