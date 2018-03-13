import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, ReLU, MaxPool2d, AvgPool2d, ConvTranspose2d, Upsample, Sigmoid
from torch.nn import BatchNorm2d as Norm2d
from torch.nn import Parameter

class seg_net(nn.Module):
    def __init__(self, nf=32, nc=1, img_h=256, img_w=256):
        super(seg_net, self).__init__()
        
        self.nf = nf
        self.nc = nc
        
        self.conv_block_1 = conv357_block(1, 64, True)
        self.conv_block_2 = conv357_block(3*64, 32, True)
        self.conv_block_3 = conv357_block(3*32, 16, True)
        self.conv_block_4 = conv357_block(3*16, 1, True)
        self.conv5 = Conv2d(3, 1, 1, 1, 0, bias=True)
        
        torch.nn.init.xavier_uniform(self.conv5.weight)
        
    def forward(self, x):
        original_size = tuple(x.size())[2:]
        
        conv1 = self.conv_block_1(x)
        pool1 = AvgPool2d(2)(conv1)
        conv2 = self.conv_block_2(pool1)
        unpool1 = Upsample(size=original_size, mode='bilinear')(conv2)
        conv3 = self.conv_block_3(unpool1)
        conv4 = self.conv_block_4(conv3)
        conv5 = self.conv5(conv4)
        
        output = Sigmoid()(conv5)
        
        return output
    
    def weights_init(self, init_func=torch.nn.init.xavier_uniform):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight)

#
class conv_SVM(nn.Module):
    def __init__(self, nf=32, nc=1):
        super(conv_SVM, self).__init__()
        
        self.conv_block = conv1357_block(nc, nf)
        self.SVM_block  = pixelwise_SVM(4*nf+1)
        
    def forward(self, x):
        conv_out = self.conv_block(x)
        SVM_out = self.SVM_block(conv_out)
        label = Sigmoid()(SVM_out)
        
        return label
    
class pixelwise_SVM(nn.Module):
    def __init__(self, nc):
        super(pixelwise_SVM, self).__init__()
        
        self.SVM_scale = Parameter(torch.Tensor(1).float())
        self.SVM_bias  = Parameter(torch.Tensor(1).float())
        self.RBF_scale = Parameter(torch.Tensor(1).float())
        self.RBF_bias  = Parameter(torch.Tensor(nc).float())
        
        self.weights_init()
    
    def forward(self, x):
        # x size BCHW
        x = x.permute(0, 2, 3, 1)
        RBF_output = torch.sum(x.add(self.RBF_bias).pow(2), 3, True).div(self.RBF_scale.pow(2)).mul(-1).exp()
        SVM_output = RBF_output.mul(self.SVM_scale).add(self.SVM_bias)
        SVM_output = SVM_output.permute(0, 3, 1, 2)
        
        return SVM_output
        
    def weights_init(self, init_func=torch.nn.init.normal):
        for para in self.parameters():
            init_func(para)

class conv357_block(nn.Module):
    def __init__(self, nc, nf, norm_flag=True):
        super(conv357_block, self).__init__()
        
        self.conv3 = Conv2d(nc, nf, 3, 1, 1, bias=True)
        self.conv5 = Conv2d(nc, nf, 5, 1, 2, bias=True)
        self.conv7 = Conv2d(nc, nf, 7, 1, 3, bias=True)
        
        if norm_flag:
            self.norm_flag = norm_flag
            self.norm = Norm2d(3*nf)
        
        self.weights_init()
        
    def forward(self, x):
        # x size BCHW
        conv3 = ReLU(True)(self.conv3(x))
        conv5 = ReLU(True)(self.conv5(x))
        conv7 = ReLU(True)(self.conv7(x))
        
        cat = torch.cat((conv3, conv5, conv7), dim=1)
        if self.norm_flag: cat = self.norm(cat)
        
        return cat

    def weights_init(self, init_func=torch.nn.init.kaiming_uniform):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight)


#






