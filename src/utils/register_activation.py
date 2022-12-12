import torch
import torchvision
import magicattr

from src.DWT_IDWT import DWT_2D
from .plot_activations import *

def get_activation(features,name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook

def register_extract_activation_hook(model, layers, activations):
    for name in layers:        
        magicattr.get(model, name).register_forward_hook(get_activation(activations,name));

def register_activation_hook(activations,model,model_name,dataset):
    
    if model_name in ('resnet50', 'resnet50-lrelu'):    
        layers=[
            'conv1',
            'layers[0][0].downsample[0]',
            'layers[0][0].downsample[2]',
            'layers[1][0].downsample[0]',
            'layers[1][0].downsample[2]',
            'layers[2][0].downsample[0]',
            'layers[2][0].downsample[2]',
            'layers[3][0].downsample[0]',
            'layers[3][0].downsample[2]',
            'layers[3][2].bn3'
        ];
    elif model_name[:12] == 'resnet50_dwt':    
        layers=[
            'conv1',
            'layers[0][0].downsample[0]',
            'layers[0][0].downsample[2]',
            'layers[1][0].downsample[0]',
            'layers[1][0].downsample[2]',
            'layers[2][0].downsample[0]',
            'layers[2][0].downsample[2]',
            'layers[3][0].downsample[0]',
            'layers[3][0].downsample[2]',
            'layers[3][2].bn3'
        ];
    elif model_name in ('vgg16','vgg16-lrelu'):
        layers = [
            'features[2]',  
            'features[7]',  
            'features[14]',  
            'features[21]',  
            'features[28]'  
        ];
    elif model_name[:9] == 'vgg16_dwt':    
        layers = [
            'features[2]',  
            'features[7]',  
            'features[14]',  
            'features[21]',  
            'features[28]'  
        ];
    elif model_name in ('vgg16_bn','vgg16_bn-lrelu'):
        layers = [
            'features[3]',  
            'features[4]',  
            'features[5]',  
            'features[10]',  
            'features[11]',  
            'features[12]',  
            'features[20]',  
            'features[21]',  
            'features[22]',  
            'features[30]',  
            'features[31]',  
            'features[32]',
            'features[40]',
            'features[41]',
            'features[42]',
        ];
    elif model_name[:12] == 'vgg16_bn_dwt':    
        layers = [
            'features[3]',  
            'features[4]',  
            'features[5]',  
            'features[10]',  
            'features[11]',  
            'features[12]',  
            'features[20]',  
            'features[21]',  
            'features[22]',  
            'features[30]',  
            'features[31]',  
            'features[32]',
            'features[40]',
            'features[41]',
            'features[42]',
        ];
    elif model_name in ('lenet5','lenet5-lrelu'):    
        layers = [
            'conv1',
            'conv2'
        ];
    elif model_name[:10] == 'lenet5_dwt':    
        layers = [
            'conv1',
            'conv2'
        ];

    register_extract_activation_hook(model,layers,activations);
    

def define_test_hook(activations,logger,wavelet,device):
    def test_hook(x,step,ids=None):
        for i in range(len(x)):
            if ids:
                if i not in ids:
                    continue;
            inp_act_grid = torchvision.utils.make_grid(x[i].unsqueeze(0),nrow=1);
            logger.add_imagegrid(f'{i}/original',inp_act_grid);

        log_activation_maps(activations,logger,ids,step);

        # Taking DWT
        dwt=DWT_2D(wavelet).to(device);

        # Hacky solution to handle odd sized layers in DWT being used

        ll={};lh={};hl={};hh={};
        for name,val in activations.items():
            with torch.no_grad():
                LL,LH,HL,HH = dwt(activations[name].to(device));
                ll[name]=LL;
                lh[name]=LH;
                hl[name]=HL;
                hh[name]=HH;
                

        log_activation_dwt_maps(ll,hl,lh,hh,logger,wavelet,ids,step);

        log_energy_hist(ll,hl,lh,hh,logger,ids,step)

    return test_hook;    