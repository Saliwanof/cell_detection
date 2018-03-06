import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, ReLU, MaxPool2d, ConvTranspose2d, Sigmoid
from torch.nn import BatchNorm2d as Norm2d
from torch.nn import Parameter

class seg_net(nn.Module):
    def __init__(self, nf=32, nc=1, img_h=256, img_w=256):
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
        RBF_output = torch.sum(x.add_(self.RBF_bias).pow_(2), 3, True).div_(self.RBF_scale.pow(2)).mul_(-1).exp_()
        SVM_output = RBF_output.mul_(self.SVM_scale).add(self.SVM_bias)
        SVM_output = SVM_output.permute(0, 3, 1, 2)
        
        return SVM_output
        
    def weights_init(self, init_func=torch.nn.init.normal):
        for para in self.parameters():
            init_func(para)

class conv1357_block(nn.Module):
    def __init__(self, nc, nf):
        super(conv1357_block, self).__init__()
        
        self.conv3 = Conv2d(nc,   nf, 3, 1, 1, bias=False)
        self.conv5 = Conv2d(nc,   nf, 5, 1, 2, bias=False)
        self.conv7 = Conv2d(nc, 2*nf, 7, 1, 3, bias=False)
        
        self.norm = Norm2d(4*nf+1)
        
        self.weights_init()
        
    def forward(self, x):
        # x size BCHW
        conv3 = self.conv3(x)
        conv5 = self.conv5(x)
        conv7 = self.conv7(x)
        
        cat = torch.cat((x, conv3, conv5, conv7), dim=1)
        norm = self.norm(cat)
        
        return norm

    def weights_init(self, init_func=torch.nn.init.xavier_uniform):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init_func(m.weight)


#






