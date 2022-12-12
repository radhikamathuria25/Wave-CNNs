'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from src.DWT_IDWT.downsample import *

class BasicBlock_dwt(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None,downsample_ly=nn.MaxPool2d(kernel_size=1,stride=2),non_linearity_ty='relu'):
        super(BasicBlock_dwt, self).__init__()
        if(stride==1):
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(planes, self.expansion*planes, kernel_size=3, stride=1, padding=1, bias=False),
                                       downsample_ly,)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, self.expansion*planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = downsample
        self.non_linearity_ty = non_linearity_ty
        self.non_linearity = nn.ReLU(inplace=True) if non_linearity_ty == 'relu' else nn.LeakyReLU(inplace=True);
        

    def forward(self, x):
        out = self.non_linearity(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        residual=x;
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual;
        out = self.non_linearity(out)
        return out


class Bottleneck_dwt(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, downsample_ly=nn.MaxPool2d(kernel_size=1,stride=2),non_linearity_ty='relu'):
        super(Bottleneck_dwt, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if(stride==1):
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv2 = nn.Sequential(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                                       downsample_ly,)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        
        self.downsample=downsample;
        self.non_linearity_ty = non_linearity_ty
        self.non_linearity = nn.ReLU(inplace=True) if non_linearity_ty == 'relu' else nn.LeakyReLU(inplace=True);

    def forward(self, x):
        out = self.non_linearity(self.bn1(self.conv1(x)))
        out = self.non_linearity(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        residual=x;
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual;
        out = self.non_linearity(out)
        return out

# ResNet v1.0
class ResNet_dwt(nn.Module):
    def __init__(self, block, num_blocks, width, stride, stem_width=64, num_classes=10,downsample_ly=nn.MaxPool2d(kernel_size=1,stride=2),non_linearity_ty='relu'):
        super(ResNet_dwt, self).__init__()
        self.in_planes = stem_width;
        self.downsample_ly=downsample_ly;
        
        self.conv1 = nn.Conv2d(3, stem_width, kernel_size=7, stride=1,bias=False)
        self.mp0 = downsample_ly;
        self.mp1 = downsample_ly;
        self.bn1 = nn.BatchNorm2d(stem_width)
        
        self.non_linearity_ty = non_linearity_ty
        self.non_linearity = nn.ReLU(inplace=True) if non_linearity_ty == 'relu' else nn.LeakyReLU(inplace=True);
        
        inplanes=stem_width;
        layers = [];
        for i in range(len(num_blocks)):
            layer,inplanes = self._make_layer(block, inplanes, width[i], num_blocks[i], stride=stride[i])
            layers.append(copy.deepcopy(layer))
                
        self.layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width[-1]*block.expansion, num_classes)

    def _make_layer(self, block, inplanes, planes, num_blocks, stride):    
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            conv = nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=1,bias=False)
            dpool = nn.MaxPool2d(kernel_size=1,stride=stride) if stride == 1 else self.downsample_ly;
            dbn = nn.BatchNorm2d(planes * block.expansion)
            if dbn is not None:
                downsample = nn.Sequential(conv, dpool, dbn)
            else:
                downsample = nn.Sequential(conv, dpool)

        layers = []
        for i in range(num_blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    stride=stride if i == 0 else 1,
                    downsample=downsample if i == 0 else None,
                    downsample_ly = self.downsample_ly,
                    non_linearity_ty = self.non_linearity_ty
                )
            )
            inplanes = planes * block.expansion;

        return nn.Sequential(*layers),inplanes
    
    def forward(self, x):
        out = self.non_linearity(self.bn1(self.mp1(self.mp0(self.conv1(x)))))
        
        out = self.layers(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)        
        out = self.fc(out)
        # out = F.softmax(out, dim=-1)
        return out


def ResNet9_dwt(num_classes=10,wavename='haar',dwt_type='LL',non_linearity_ty='relu'):
    return ResNet_dwt(BasicBlock_dwt, [1,1,1], [128, 256, 512], [1,1,1], stem_width=64, num_classes=num_classes,downsample_ly=get_dwt_layer(wavename=wavename,dwt_type=dwt_type),non_linearity_ty=non_linearity_ty)

def ResNet18_dwt(num_classes=10,wavename='haar',dwt_type='LL',non_linearity_ty='relu'):
    return ResNet_dwt(BasicBlock_dwt, [2,2,2,2],downsample_ly=get_dwt_layer(wavename=wavename,dwt_type=dwt_type),non_linearity_ty=non_linearity_ty)

def ResNet34_dwt(wavename='haar',dwt_type='LL',non_linearity_ty='relu'):
    return ResNet_dwt(BasicBlock_dwt, [3,4,6,3],downsample_ly=get_dwt_layer(wavename=wavename,dwt_type=dwt_type),non_linearity_ty=non_linearity_ty)

def ResNet50_dwt(num_classes=10,wavename='haar',dwt_type='LL',non_linearity_ty='relu'):
    return ResNet_dwt(Bottleneck_dwt, [3,4,6,3], [64, 128, 256, 512], [1,2,2,2] ,stem_width=64, num_classes=num_classes,downsample_ly=get_dwt_layer(wavename=wavename,dwt_type=dwt_type),non_linearity_ty=non_linearity_ty)

def ResNet101_dwt(wavename='haar',dwt_type='LL',non_linearity_ty='relu'):
    return ResNet_dwt(Bottleneck_dwt, [3,4,23,3],downsample_ly=get_dwt_layer(wavename=wavename,dwt_type=dwt_type),non_linearity_ty=non_linearity_ty)

def ResNet152_dwt(wavename='haar',dwt_type='LL',non_linearity_ty='relu'):
    return ResNet_dwt(Bottleneck_dwt, [3,8,36,3],downsample_ly=get_dwt_layer(wavename=wavename,dwt_type=dwt_type),non_linearity_ty=non_linearity_ty)


def test():
    net = ResNet18_dwt()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()