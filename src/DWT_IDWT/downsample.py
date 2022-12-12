import torch
import torch.nn as nn

from src.DWT_IDWT import DWT_2D, DWT_2D_tiny, IDWT_2D

class Downsample_LL(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(Downsample_LL, self).__init__()
        self.dwt = DWT_2D_tiny(wavename = wavename)

    def forward(self, input):
        LL = self.dwt(input)
        return LL
class Downsample_LH(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(Downsample_LH, self).__init__()
        self.dwt = DWT_2D(wavename = wavename)

    def forward(self, input):
        LL,LH,HL,HH = self.dwt(input)
        return LH
class Downsample_HL(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(Downsample_HL, self).__init__()
        self.dwt = DWT_2D(wavename = wavename)

    def forward(self, input):
        LL,LH,HL,HH = self.dwt(input)
        return HL
class Downsample_HH(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(Downsample_HH, self).__init__()
        self.dwt = DWT_2D(wavename = wavename)

    def forward(self, input):
        LL,LH,HL,HH = self.dwt(input)
        return HH

class Downsample_cat(nn.Module):
    """
        X --> torch.cat(X_ll, X_lh, X_hl, X_hh)
    """
    def __init__(self, wavename = 'haar'):
        super(Downsample_cat, self).__init__()
        self.dwt = DWT_2D(wavename = wavename)

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        return torch.cat((LL, LH, HL, HH), dim = 1)


class Downsample_avg(nn.Module):
    """
        X --> 1/4*(X_ll + X_lh + X_hl + X_hh)
    """
    def __init__(self, wavename = 'haar'):
        super(Downsample_avg, self).__init__()
        self.dwt = DWT_2D(wavename = wavename)

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        return (LL + LH + HL + HH) / 4

class Downsample_LL2(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(Downsample_LL2, self).__init__()
        self.dwt1 = DWT_2D_tiny(wavename = wavename);
        self.dwt2 = DWT_2D_tiny(wavename = wavename);
        self.idwt = IDWT_2D(wavename = wavename);

    def forward(self, x):
        LL= self.dwt1(x)
        LL2= self.dwt2(LL)
        LL_f = self.idwt(LL2,torch.zeros_like(LL2),torch.zeros_like(LL2),torch.zeros_like(LL2));
        return LL_f;

    
def get_dwt_layer(wavename, dwt_type):
    
    if dwt_type == "LL":
        dwt_layer = Downsample_LL;
    elif dwt_type == "LH":
        dwt_layer = Downsample_LH;
    elif dwt_type == "HL":
        dwt_layer = Downsample_HL;
    elif dwt_type == "HH":
        dwt_layer = Downsample_HH;
    elif dwt_type == "Avg":
        dwt_layer = Downsample_avg;
    elif dwt_type == "Cat":
        dwt_layer = Downsample_cat;
    elif dwt_type == "LL2":
        dwt_layer = Downsample_LL2;
    
    return dwt_layer(wavename=wavename);