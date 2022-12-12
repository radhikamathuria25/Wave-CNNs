# This code is built from the PyTorch examples repository: https://github.com/pytorch/vision/tree/master/torchvision/models.
# Copyright (c) 2017 Torch Contributors.
# The Pytorch examples are available under the BSD 3-Clause License.
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2019 Adobe. All rights reserved.
# Adobe’s modifications are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License (CC-NC-SA-4.0). To view a copy of the license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
#
# ==========================================================================================
#
# BSD-3 License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE

import torch
import torch.nn as nn

from src.DWT_IDWT.downsample import *

class VGG_dwt(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, downsample_ly=nn.MaxPool2d(kernel_size=1,stride=2),non_linearity_ty='relu'):
        super(VGG_dwt, self).__init__()
        self.features = features
        self.downsample_ly = downsample_ly;
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.non_linearity_ty=non_linearity_ty;
        non_linearity = nn.ReLU if non_linearity_ty == 'relu' else nn.LeakyReLU;
                
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            non_linearity(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            non_linearity(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels != m.out_channels or m.out_channels != m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self.non_linearity_ty)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    print('Not initializing')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, downsample_ly=nn.MaxPool2d(kernel_size=1,stride=2),batch_norm=False,non_linearity_ty='relu'):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [downsample_ly]
            # layers += [nn.MaxPool2d(kernel_size=2, stride=1), Downsample(filt_size=filter_size, stride=2, channels=in_channels)]
        else:
            non_linearity = nn.ReLU if non_linearity_ty == 'relu' else nn.LeakyReLU;
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), non_linearity(inplace=True)]
            else:
                layers += [conv2d, non_linearity(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg11_dwt(wavename='haar',dwt_type='LL',non_linearity_ty='relu',**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_dwt(make_layers(cfg['A'],get_dwt_layer(wavename,dwt_type),non_linearity_ty=non_linearity_ty), non_linearity_ty=non_linearity_ty,**kwargs)
    return model


def vgg11_bn_dwt(wavename='haar',dwt_type='LL',non_linearity_ty='relu',**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_dwt(make_layers(cfg['A'],get_dwt_layer(wavename,dwt_type),non_linearity_ty=non_linearity_ty, batch_norm=True), non_linearity_ty=non_linearity_ty,**kwargs)
    return model


def vgg13_dwt(wavename='haar',dwt_type='LL',non_linearity_ty='relu',**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_dwt(make_layers(cfg['B'],get_dwt_layer(wavename,dwt_type),non_linearity_ty=non_linearity_ty), non_linearity_ty=non_linearity_ty,**kwargs)
    return model


def vgg13_bn_dwt(wavename='haar',dwt_type='LL',non_linearity_ty='relu',**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_dwt(make_layers(cfg['B'],get_dwt_layer(wavename,dwt_type),non_linearity_ty=non_linearity_ty, batch_norm=True), non_linearity_ty=non_linearity_ty,**kwargs)
    return model


def vgg16_dwt(wavename='haar',dwt_type='LL',non_linearity_ty='relu',**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_dwt(make_layers(cfg['D'],get_dwt_layer(wavename,dwt_type),non_linearity_ty=non_linearity_ty), non_linearity_ty=non_linearity_ty,**kwargs)
    return model


def vgg16_bn_dwt(wavename='haar',dwt_type='LL',non_linearity_ty='relu',**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_dwt(make_layers(cfg['D'],get_dwt_layer(wavename,dwt_type),non_linearity_ty=non_linearity_ty, batch_norm=True), non_linearity_ty=non_linearity_ty,**kwargs)
    return model


def vgg19_dwt(wavename='haar',dwt_type='LL',non_linearity_ty='relu',**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_dwt(make_layers(cfg['E'],get_dwt_layer(wavename,dwt_type),non_linearity_ty=non_linearity_ty), non_linearity_ty=non_linearity_ty,**kwargs)
    return model


def vgg19_bn_dwt(wavename='haar',dwt_type='LL',non_linearity_ty='relu',**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_dwt(make_layers(cfg['E'],get_dwt_layer(wavename,dwt_type),non_linearity_ty=non_linearity_ty, batch_norm=True), non_linearity_ty=non_linearity_ty,**kwargs)
    return model

