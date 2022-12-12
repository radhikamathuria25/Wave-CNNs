import torch
import torch.nn as nn
import torch.nn.functional as F

from src.DWT_IDWT.downsample import *

class LeNet5_dwt(nn.Module):
    def __init__(self, in_chan,input_dim, hidden_dims, output_dim=10,downsample_ly=nn.MaxPool2d(kernel_size=1,stride=2),non_linearity_ty='relu'):
        super(LeNet5_dwt, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, 6, 5)
        self.pool1 = downsample_ly;
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = downsample_ly;

        self.non_linearity_ty=non_linearity_ty;
        self.non_linearity = nn.ReLU(inplace=True) if non_linearity_ty == 'relu' else nn.LeakyReLU(inplace=True); 
        
        self.input_dim = input_dim;
        self.hidden_dims = hidden_dims;
        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool1(self.non_linearity(self.conv1(x)))
        x = self.pool2(self.non_linearity(self.conv2(x)))
        x = x.view(-1, self.input_dim)

        x = self.non_linearity(self.fc1(x))
        x = self.non_linearity(self.fc2(x))
        x = self.fc3(x)
        return x
    

def lenet5_dwt(num_classes=10,wavename='haar',dwt_type='LL',non_linearity_ty='relu',**kwargs):
    return LeNet5_dwt(in_chan=3,input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=num_classes, downsample_ly=get_dwt_layer(wavename=wavename,dwt_type=dwt_type),non_linearity_ty=non_linearity_ty,**kwargs);