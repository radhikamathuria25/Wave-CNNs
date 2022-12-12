import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, in_chan,input_dim, hidden_dims, output_dim=10,non_linearity_ty='relu'):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

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
        x = self.pool(self.non_linearity(self.conv1(x)))
        x = self.pool(self.non_linearity(self.conv2(x)))
        x = x.view(-1, self.input_dim)

        x = self.non_linearity(self.fc1(x))
        x = self.non_linearity(self.fc2(x))
        x = self.fc3(x)
        return x
    

def lenet5(num_classes=10,non_linearity_ty='relu',**kwargs):
    return LeNet5(in_chan=3,input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=num_classes,non_linearity_ty=non_linearity_ty,**kwargs);