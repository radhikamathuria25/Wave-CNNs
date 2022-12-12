import torch

from .model import *


def init_net(model,dataset,device=torch.device('cpu')):
    
    if dataset in ('imagenet') or dataset[:11] == 'imagefolder':
        if model == 'resnet50':
            net = ResNet50(num_classes=1000).to(device);
        if model == 'resnet50-lrelu':
            net = ResNet50(num_classes=1000,non_linearity_ty='leaky_relu').to(device);
        elif model == 'vgg16':
            net = vgg16(num_classes=1000).to(device);
        elif model == 'vgg16-lrelu':
            net = vgg16(num_classes=1000,non_linearity_ty='leaky_relu').to(device);
        elif model == 'vgg16_bn':
            net = vgg16_bn(num_classes=1000).to(device);
        elif model == 'vgg16_bn-lrelu':
            net = vgg16_bn(num_classes=1000,non_linearity_ty='leaky_relu').to(device);
        elif model[:18] == 'resnet50_dwt-lrelu':     # resnet50_dwt/bior4.4/Avg
            (wavename,_,downsample_ty) =model[19:].partition('/');
            net = ResNet50_dwt(num_classes=1000,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device);
        elif model[:13] == 'resnet50_dwt':     # resnet50_dwt/bior4.4/Avg
            (wavename,_,downsample_ty) =model[14:].partition('/');
            net = ResNet50_dwt(num_classes=1000,wavename=wavename,dwt_type=downsample_ty).to(device);
        elif model[:15] == 'vgg16_dwt-lrelu':
            (wavename,_,downsample_ty) =model[16:].partition('/');
            net = vgg16_dwt(num_classes=1000,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device);
        elif model[:9] == 'vgg16_dwt':
            (wavename,_,downsample_ty) =model[10:].partition('/');
            net = vgg16_dwt(num_classes=1000,wavename=wavename,dwt_type=downsample_ty).to(device);
        elif model[:18] == 'vgg16_bn_dwt-lrelu':
            (wavename,_,downsample_ty) =model[19:].partition('/');
            net = vgg16_bn_dwt(num_classes=1000,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device);    
        elif model[:11] == 'vgg16_bn_dwt':
            (wavename,_,downsample_ty) =model[12:].partition('/');
            net = vgg16_bn_dwt(num_classes=1000,wavename=wavename,dwt_type=downsample_ty).to(device);    
    elif dataset in ('fvc2000'):
        if model == 'resnet50':
            net = ResNet50(num_classes=10).to(device);
        if model == 'resnet50-lrelu':
            net = ResNet50(num_classes=10,non_linearity_ty='leaky_relu').to(device);
        elif model == 'vgg16':
            net = vgg16(num_classes=10).to(device);
        elif model == 'vgg16-lrelu':
            net = vgg16(num_classes=10,non_linearity_ty='leaky_relu').to(device);
        elif model == 'vgg16_bn':
            net = vgg16_bn(num_classes=10).to(device);
        elif model == 'vgg16_bn-lrelu':
            net = vgg16_bn(num_classes=10,non_linearity_ty='leaky_relu').to(device);
        elif model[:18] == 'resnet50_dwt-lrelu':     # resnet50_dwt/bior4.4/Avg
            (wavename,_,downsample_ty) =model[19:].partition('/');
            net = ResNet50_dwt(num_classes=10,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device);
        elif model[:13] == 'resnet50_dwt':     # resnet50_dwt/bior4.4/Avg
            (wavename,_,downsample_ty) =model[14:].partition('/');
            net = ResNet50_dwt(num_classes=10,wavename=wavename,dwt_type=downsample_ty).to(device);
        elif model[:15] == 'vgg16_dwt-lrelu':
            (wavename,_,downsample_ty) =model[16:].partition('/');
            net = vgg16_dwt(num_classes=10,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device);
        elif model[:9] == 'vgg16_dwt':
            (wavename,_,downsample_ty) =model[10:].partition('/');
            net = vgg16_dwt(num_classes=10,wavename=wavename,dwt_type=downsample_ty).to(device);
        elif model[:18] == 'vgg16_bn_dwt-lrelu':
            (wavename,_,downsample_ty) =model[19:].partition('/');
            net = vgg16_bn_dwt(num_classes=10,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device);
        elif model[:11] == 'vgg16_bn_dwt':
            (wavename,_,downsample_ty) =model[12:].partition('/');
            net = vgg16_bn_dwt(num_classes=10,wavename=wavename,dwt_type=downsample_ty).to(device);
    elif dataset in ('tinyimagenet'):
        if model == 'resnet50':
            net = ResNet50(num_classes=200).to(device);
        if model == 'resnet50-lrelu':
            net = ResNet50(num_classes=200,non_linearity_ty='leaky_relu').to(device);
        elif model == 'vgg16':
            net = vgg16(num_classes=200).to(device);
        elif model == 'vgg16-lrelu':
            net = vgg16(num_classes=200,non_linearity_ty='leaky_relu').to(device);
        elif model == 'vgg16_bn':
            net = vgg16_bn(num_classes=200).to(device);
        elif model == 'vgg16_bn-lrelu':
            net = vgg16_bn(num_classes=200,non_linearity_ty='leaky_relu').to(device);
        elif model[:18] == 'resnet50_dwt-lrelu':     # resnet50_dwt/bior4.4/Avg
            (wavename,_,downsample_ty) =model[19:].partition('/');
            net = ResNet50_dwt(num_classes=200,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device);
        elif model[:13] == 'resnet50_dwt':     # resnet50_dwt/bior4.4/Avg
            (wavename,_,downsample_ty) =model[14:].partition('/');
            net = ResNet50_dwt(num_classes=200,wavename=wavename,dwt_type=downsample_ty).to(device);
        elif model[:15] == 'vgg16_dwt-lrelu':
            (wavename,_,downsample_ty) =model[16:].partition('/');
            net = vgg16_dwt(num_classes=200,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device);
        elif model[:9] == 'vgg16_dwt':
            (wavename,_,downsample_ty) =model[10:].partition('/');
            net = vgg16_dwt(num_classes=200,wavename=wavename,dwt_type=downsample_ty).to(device);
        elif model[:18] == 'vgg16_bn_dwt-lrelu':
            (wavename,_,downsample_ty) =model[19:].partition('/');
            net = vgg16_bn_dwt(num_classes=200,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device);
        elif model[:11] == 'vgg16_bn_dwt':
            (wavename,_,downsample_ty) =model[12:].partition('/');
            net = vgg16_bn_dwt(num_classes=200,wavename=wavename,dwt_type=downsample_ty).to(device);
    elif dataset in ('cifar10','cifar10u'):
        if model == 'resnet50':
            net = ResNet50(num_classes=10).to(device);
        if model == 'resnet50-lrelu':
            net = ResNet50(num_classes=10,non_linearity_ty='leaky_relu').to(device);
        elif model == 'vgg16':
            net = vgg16(num_classes=10).to(device);
        elif model == 'vgg16-lrelu':
            net = vgg16(num_classes=10,non_linearity_ty='leaky_relu').to(device);
        elif model == 'vgg16_bn':
            net = vgg16_bn(num_classes=10).to(device);
        elif model == 'vgg16_bn-lrelu':
            net = vgg16_bn(num_classes=10,non_linearity_ty='leaky_relu').to(device);
        elif model[:18] == 'resnet50_dwt-lrelu':     # resnet50_dwt/bior4.4/Avg
            (wavename,_,downsample_ty) =model[19:].partition('/');
            net = ResNet50_dwt(num_classes=10,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device);
        elif model[:13] == 'resnet50_dwt':     # resnet50_dwt/bior4.4/Avg
            (wavename,_,downsample_ty) =model[14:].partition('/');
            net = ResNet50_dwt(num_classes=10,wavename=wavename,dwt_type=downsample_ty).to(device);
        elif model[:15] == 'vgg16_dwt-lrelu':
            (wavename,_,downsample_ty) =model[16:].partition('/');
            net = vgg16_dwt(num_classes=10,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device);
        elif model[:9] == 'vgg16_dwt':
            (wavename,_,downsample_ty) =model[10:].partition('/');
            net = vgg16_dwt(num_classes=10,wavename=wavename,dwt_type=downsample_ty).to(device);
        elif model[:18] == 'vgg16_bn_dwt-lrelu':
            (wavename,_,downsample_ty) =model[19:].partition('/');
            net = vgg16_bn_dwt(num_classes=10,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device);
        elif model[:11] == 'vgg16_bn_dwt':
            (wavename,_,downsample_ty) =model[12:].partition('/');
            net = vgg16_bn_dwt(num_classes=10,wavename=wavename,dwt_type=downsample_ty).to(device);
        elif model == "lenet5-lrelu":
            net = lenet5(num_classes=10,non_linearity_ty='leaky_relu').to(device)
        elif model[:16] == "lenet5_dwt-lrelu":
            (wavename,_,downsample_ty) =model[17:].partition('/');
            net = lenet5_dwt(num_classes=10,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device)
        elif model[:10] == "lenet5_dwt":
            (wavename,_,downsample_ty) =model[11:].partition('/');
            net = lenet5_dwt(num_classes=10,wavename=wavename,dwt_type=downsample_ty).to(device)
    elif dataset in ('fp302','fp302u'):
        if model == 'resnet50':
            net = ResNet50(num_classes=180).to(device);
        if model == 'resnet50-lrelu':
            net = ResNet50(num_classes=180,non_linearity_ty='leaky_relu').to(device);
        elif model == 'vgg16':
            net = vgg16(num_classes=180).to(device);
        elif model == 'vgg16-lrelu':
            net = vgg16(num_classes=180,non_linearity_ty='leaky_relu').to(device);
        elif model == 'vgg16_bn':
            net = vgg16_bn(num_classes=180).to(device);
        elif model == 'vgg16_bn-lrelu':
            net = vgg16_bn(num_classes=180,non_linearity_ty='leaky_relu').to(device);
        elif model[:18] == 'resnet50_dwt-lrelu':     # resnet50_dwt/bior4.4/Avg
            (wavename,_,downsample_ty) =model[19:].partition('/');
            net = ResNet50_dwt(num_classes=180,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device);
        elif model[:13] == 'resnet50_dwt':     # resnet50_dwt/bior4.4/Avg
            (wavename,_,downsample_ty) =model[14:].partition('/');
            net = ResNet50_dwt(num_classes=180,wavename=wavename,dwt_type=downsample_ty).to(device);
        elif model[:15] == 'vgg16_dwt-lrelu':
            (wavename,_,downsample_ty) =model[16:].partition('/');
            net = vgg16_dwt(num_classes=180,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device);
        elif model[:9] == 'vgg16_dwt':
            (wavename,_,downsample_ty) =model[10:].partition('/');
            net = vgg16_dwt(num_classes=180,wavename=wavename,dwt_type=downsample_ty).to(device);
        elif model[:18] == 'vgg16_bn_dwt-lrelu':
            (wavename,_,downsample_ty) =model[19:].partition('/');
            net = vgg16_bn_dwt(num_classes=180,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device);
        elif model[:11] == 'vgg16_bn_dwt':
            (wavename,_,downsample_ty) =model[12:].partition('/');
            net = vgg16_bn_dwt(num_classes=180,wavename=wavename,dwt_type=downsample_ty).to(device);
        elif model == "lenet5-lrelu":
            net = lenet5(num_classes=180,non_linearity_ty='leaky_relu').to(device)
        elif model[:16] == "lenet5_dwt-lrelu":
            (wavename,_,downsample_ty) =model[17:].partition('/');
            net = lenet5_dwt(num_classes=180,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device)
        elif model[:10] == "lenet5_dwt":
            (wavename,_,downsample_ty) =model[11:].partition('/');
            net = lenet5_dwt(num_classes=180,wavename=wavename,dwt_type=downsample_ty).to(device)
    elif dataset in ('cifar100','cifar100u'):
        if model == 'resnet50':
            net = ResNet50(num_classes=100).to(device);
        if model == 'resnet50-lrelu':
            net = ResNet50(num_classes=100,non_linearity_ty='leaky_relu').to(device);
        elif model == 'vgg16':
            net = vgg16(num_classes=100).to(device);
        elif model == 'vgg16-lrelu':
            net = vgg16(num_classes=100,non_linearity_ty='leaky_relu').to(device);
        elif model == 'vgg16_bn':
            net = vgg16_bn(num_classes=100).to(device);
        elif model == 'vgg16_bn-lrelu':
            net = vgg16_bn(num_classes=100,non_linearity_ty='leaky_relu').to(device);
        elif model[:18] == 'resnet50_dwt-lrelu':     # resnet50_dwt/bior4.4/Avg
            (wavename,_,downsample_ty) =model[19:].partition('/');
            net = ResNet50_dwt(num_classes=100,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device);
        elif model[:13] == 'resnet50_dwt':     # resnet50_dwt/bior4.4/Avg
            (wavename,_,downsample_ty) =model[14:].partition('/');
            net = ResNet50_dwt(num_classes=100,wavename=wavename,dwt_type=downsample_ty).to(device);
        elif model[:15] == 'vgg16_dwt-lrelu':
            (wavename,_,downsample_ty) =model[16:].partition('/');
            net = vgg16_dwt(num_classes=100,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device);
        elif model[:9] == 'vgg16_dwt':
            (wavename,_,downsample_ty) =model[10:].partition('/');
            net = vgg16_dwt(num_classes=100,wavename=wavename,dwt_type=downsample_ty).to(device);
        elif model[:18] == 'vgg16_bn_dwt-lrelu':
            (wavename,_,downsample_ty) =model[19:].partition('/');
            net = vgg16_bn_dwt(num_classes=100,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device);
        elif model[:11] == 'vgg16_bn_dwt':
            (wavename,_,downsample_ty) =model[12:].partition('/');
            net = vgg16_bn_dwt(num_classes=100,wavename=wavename,dwt_type=downsample_ty).to(device);
        elif model == "lenet5-lrelu":
            net = lenet5(num_classes=100,non_linearity_ty='leaky_relu').to(device)
        elif model[:16] == "lenet5_dwt-lrelu":
            (wavename,_,downsample_ty) =model[17:].partition('/');
            net = lenet5_dwt(num_classes=100,wavename=wavename,dwt_type=downsample_ty,non_linearity_ty='leaky_relu').to(device)
        elif model[:10] == "lenet5_dwt":
            (wavename,_,downsample_ty) =model[11:].partition('/');
            net = lenet5_dwt(num_classes=100,wavename=wavename,dwt_type=downsample_ty).to(device)
            
    if net == None:
        print(f"Model: {model} and dataset {dataset} combination is not supported yet")
        exit(1)
    
    return net;