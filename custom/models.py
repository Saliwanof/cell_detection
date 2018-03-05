import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, ReLU, MaxPool2d, ConvTranspose2d, Sigmoid
from torch.nn import BatchNorm2d as Norm2d

class seg_net(nn.Module):
    def __init__(self, nf=32, nc=1, img_h=256, img_w=256, bn=False):
        super(seg_net, self).__init__()
        
        self.nf = nf
        self.nc = nc
        
        self.conv1_3 = Conv2d(nc,   nf, 3, 1, 1, bias=True)
        self.conv1_5 = Conv2d(nc,   nf, 5, 1, 2, bias=True)
        self.conv1_7 = Conv2d(nc, 2*nf, 7, 1, 3, bias=True)
        self.norm1 = Norm2d(4*nf+1)
        
        self.conv2_1 = Conv2d(4*nf+1, 1, 1, 1, 0, bias=True)
        self.conv2_3 = Conv2d(4*nf+1, 1, 3, 1, 1, bias=True)
        self.conv2_5 = Conv2d(4*nf+1, 1, 5, 1, 2, bias=True)
        self.conv2_7 = Conv2d(4*nf+1, 1, 7, 1, 3, bias=True)
        self.norm2 = Norm2d(4)
        
        self.conv3 = Conv2d(4, 1, 1, 1, 0, bias=True)
        
        self.weights_init()
        
    def forward(self, x):
        conv1_3 = self.conv1_3(x)
        conv1_5 = self.conv1_5(x)
        conv1_7 = self.conv1_7(x)
        conv1 = torch.cat((x, conv1_3, conv1_5, conv1_7), dim=1)
        conv1 = self.norm1(conv1)
        
        conv2_1 = self.conv2_1(conv1)
        conv2_3 = self.conv2_3(conv1)
        conv2_5 = self.conv2_5(conv1)
        conv2_7 = self.conv2_7(conv1)
        conv2 = torch.cat((conv2_1, conv2_3, conv2_5, conv2_7), dim=1)
        conv2 = self.norm2(conv2)
        
        output = Sigmoid()(self.conv3(conv2))
        
        return output
    
    def weights_init(self, init_func=torch.nn.init.xavier_uniform):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight)

#
class u_net(nn.Module):
    def __init__(self, nf=64, nc=1, img_h=256, img_w=256, bn=False):
        super(u_net, self).__init__()
        
        self.nc = nc
        self.nf = nf
        self.bn = bn
        
        self.conv11 = Conv2d(   nc,    nf, 7, 1, 3, bias=True)
        self.conv12 = Conv2d(   nf,    nf, 3, 1, 1, bias=True)
        
        self.conv21 = Conv2d(   nf,  2*nf, 3, 1, 1, bias=False)
        self.conv22 = Conv2d( 2*nf,  2*nf, 3, 1, 1, bias=True)
        
        self.conv31 = Conv2d( 2*nf,  4*nf, 3, 1, 1, bias=False)
        self.conv32 = Conv2d( 4*nf,  4*nf, 3, 1, 1, bias=True)
        
        self.conv41 = Conv2d( 4*nf,  8*nf, 3, 1, 1, bias=False)
        self.conv42 = Conv2d( 8*nf,  8*nf, 3, 1, 1, bias=True)
        
        self.conv51 = Conv2d( 8*nf, 4*nf, 3, 1, 1, bias=False)
        self.conv52 = Conv2d( 4*nf, 4*nf, 3, 1, 1, bias=True)
        
        self.conv61 = Conv2d( 4*nf, 2*nf, 3, 1, 1, bias=False)
        self.conv62 = Conv2d( 2*nf, 2*nf, 3, 1, 1, bias=True)
        
        self.conv71 = Conv2d( 2*nf,   nf, 3, 1, 1, bias=False)
        self.conv72 = Conv2d(   nf,   nf, 3, 1, 1, bias=True)
        
        self.unconv1 = ConvTranspose2d( 8*nf, 4*nf, 4, 2, 1, bias=False)
        self.unconv2 = ConvTranspose2d( 4*nf, 2*nf, 4, 2, 1, bias=False)
        self.unconv3 = ConvTranspose2d( 2*nf,   nf, 4, 2, 1, bias=False)
        
        self.conv01 = Conv2d(nf, 1, 1, bias=False)
        
        if self.bn:
            self.norm01 = Norm2d(   nf,affine=False)
            self.norm02 = Norm2d( 2*nf,affine=False)
            self.norm03 = Norm2d( 4*nf,affine=False)
            self.norm04 = Norm2d( 8*nf,affine=False)
            self.norm05 = Norm2d( 4*nf,affine=False)
            self.norm06 = Norm2d( 2*nf,affine=False)
            self.norm07 = Norm2d(   nf,affine=False)
        
        self.weights_init()
    
    def forward(self, x):
        nc = self.nc
        nf = self.nf
        bn = self.bn
        
        conv1 = ReLU(True)(self.conv11(x))
        conv1 = ReLU(True)(self.conv12(conv1))
        if bn: conv1 = self.norm01(conv1)
        pool1 = MaxPool2d(2)(conv1) # size (128, 128, 64)
        
        conv2 = ReLU(True)(self.conv21(pool1))
        conv2 = ReLU(True)(self.conv22(conv2))
        if bn: conv2 = self.norm02(conv2)
        pool2 = MaxPool2d(2)(conv2) # size (64, 64, 128)
        
        conv3 = ReLU(True)(self.conv31(pool2))
        conv3 = ReLU(True)(self.conv32(conv3))
        if bn: conv3 = self.norm03(conv3)
        pool3 = MaxPool2d(2)(conv3) # size (32, 32, 256)
        
        conv4 = ReLU(True)(self.conv41(pool3))
        conv4 = ReLU(True)(self.conv42(conv4))
        if bn: conv4 = self.norm04(conv4)
        
        upconv1 = torch.cat((conv3, ReLU(True)(self.unconv1(conv4))), dim=1)
        conv5 = ReLU(True)(self.conv51(upconv1))
        conv5 = ReLU(True)(self.conv52(conv5))
        if bn: conv5 = self.norm05(conv5)
        
        upconv2 = torch.cat((conv2, ReLU(True)(self.unconv2(conv5))), dim=1)
        conv6 = ReLU(True)(self.conv61(upconv2))
        conv6 = ReLU(True)(self.conv62(conv6))
        if bn: conv6 = self.norm06(conv6)
                            
        upconv3 = torch.cat((conv1, ReLU(True)(self.unconv3(conv6))), dim=1)
        conv7 = ReLU(True)(self.conv71(upconv3))
        conv7 = ReLU(True)(self.conv72(conv7))
        if bn: conv7 = self.norm07(conv7)
        
        output = Sigmoid()(self.conv01(conv7))
        return output

    def weights_init(self, init_func=torch.nn.init.kaiming_normal):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight)
#

class u_net_res(nn.Module):
    def __init__(self, nf=64, nc=1, img_h=256, img_w=256):
        super(u_net_res, self).__init__()
        
        self.nc = nc
        self.nf = nf
        
        self.conv11 = Conv2d(   nc,    nf, 3, 1, 1, bias=False)
        self.conv12 = Conv2d(   nf,    nf, 3, 1, 1, bias=False)
        self.conv13 = Conv2d(   nf,    nf, 3, 1, 1, bias=False)
        
        self.conv21 = Conv2d(   nf,  2*nf, 3, 1, 1, bias=False)
        self.conv22 = Conv2d( 2*nf,  2*nf, 3, 1, 1, bias=False)
        self.conv23 = Conv2d( 2*nf,  2*nf, 3, 1, 1, bias=False)
        
        self.conv31 = Conv2d( 2*nf,  4*nf, 3, 1, 1, bias=False)
        self.conv32 = Conv2d( 4*nf,  4*nf, 3, 1, 1, bias=False)
        self.conv33 = Conv2d( 4*nf,  4*nf, 3, 1, 1, bias=False)
        
        self.conv41 = Conv2d( 4*nf,  8*nf, 3, 1, 1, bias=False)
        self.conv42 = Conv2d( 8*nf,  8*nf, 3, 1, 1, bias=False)
        self.conv43 = Conv2d( 8*nf,  8*nf, 3, 1, 1, bias=False)
        
        self.conv51 = Conv2d( 8*nf, 16*nf, 3, 1, 1, bias=False)
        self.conv52 = Conv2d(16*nf, 16*nf, 3, 1, 1, bias=False)
        self.conv53 = Conv2d(16*nf, 16*nf, 3, 1, 1, bias=False)
        
        self.conv61 = Conv2d( 8*nf, 8*nf, 3, 1, 1, bias=False)
        self.conv62 = Conv2d( 8*nf, 8*nf, 3, 1, 1, bias=False)
        
        self.conv71 = Conv2d( 4*nf, 4*nf, 3, 1, 1, bias=False)
        self.conv72 = Conv2d( 4*nf, 4*nf, 3, 1, 1, bias=False)
        
        self.conv81 = Conv2d( 2*nf, 2*nf, 3, 1, 1, bias=False)
        self.conv82 = Conv2d( 2*nf, 2*nf, 3, 1, 1, bias=False)
        
        self.conv91 = Conv2d(   nf,   nf, 3, 1, 1, bias=False)
        self.conv92 = Conv2d(   nf,   nf, 3, 1, 1, bias=False)
        
        self.unconv1 = ConvTranspose2d(16*nf, 8*nf, 4, 2, 1, bias=False)
        self.unconv2 = ConvTranspose2d( 8*nf, 4*nf, 4, 2, 1, bias=False)
        self.unconv3 = ConvTranspose2d( 4*nf, 2*nf, 4, 2, 1, bias=False)
        self.unconv4 = ConvTranspose2d( 2*nf,   nf, 4, 2, 1, bias=False)
        
        self.conv01 = Conv2d(nf, 1, 1, bias=False)
        
        self.weights_init()
    
    def forward(self, x):
        nc = self.nc
        nf = self.nf
        
        conv11 = ReLU(True)(self.conv11(x))
        conv12 = ReLU(True)(self.conv12(conv11))
        conv13 = ReLU(True)(self.conv13(conv12) + conv11)
        pool1 = MaxPool2d(2)(conv13) # size (128, 128, 64)
        
        conv21 = ReLU(True)(self.conv21(pool1))
        conv22 = ReLU(True)(self.conv22(conv21))
        conv23 = ReLU(True)(self.conv23(conv22) + conv21)
        pool2 = MaxPool2d(2)(conv23) # size (64, 64, 128)
        
        conv31 = ReLU(True)(self.conv31(pool2))
        conv32 = ReLU(True)(self.conv32(conv31))
        conv33 = ReLU(True)(self.conv33(conv32) + conv31)
        pool3 = MaxPool2d(2)(conv33) # size (32, 32, 256)
        
        conv41 = ReLU(True)(self.conv41(pool3))
        conv42 = ReLU(True)(self.conv42(conv41))
        conv43 = ReLU(True)(self.conv43(conv42) + conv41)
        pool4 = MaxPool2d(2)(conv43) # size (16, 16, 512)
        
        conv51 = ReLU(True)(self.conv51(pool4))
        conv52 = ReLU(True)(self.conv52(conv51))
        conv53 = ReLU(True)(self.conv53(conv52) + conv51)
        
        upconv1 = conv43 + self.unconv1(conv53)
        conv6 = ReLU(True)(self.conv61(upconv1))
        conv6 = ReLU(True)(self.conv62(conv6) + upconv1)
        
        upconv2 = conv33 + self.unconv2(conv6)
        conv7 = ReLU(True)(self.conv71(upconv2))
        conv7 = ReLU(True)(self.conv72(conv7) + upconv2)
                            
        upconv3 = conv23 + self.unconv3(conv7)
        conv8 = ReLU(True)(self.conv81(upconv3))
        conv8 = ReLU(True)(self.conv82(conv8) + upconv3)
        
        upconv4 = conv13 + self.unconv4(conv8)
        conv9 = ReLU(True)(self.conv91(upconv4))
        conv9 = ReLU(True)(self.conv92(conv9) + upconv4)
        
        output = Sigmoid()(self.conv01(conv9))
        return output

    def weights_init(self, init_func=torch.nn.init.kaiming_uniform):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight)
#







