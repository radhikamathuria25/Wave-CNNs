import torch
import re

def load_pretrain_weights(ptdir, model,dataset,device=torch.device('cpu')):

    pt_path=f'{ptdir}/{dataset}/{model}';
    expert_weight=None;
    if dataset in ('imagenet','tinyimagenet') or dataset[:11] == 'imagefolder':
        if model == 'resnet-50':
            expert_weight = torch.load(f'{pt_path}/nvidia_resnet50_200821.pth.tar');
            # layer name correction
            expert_weight_corrected={}
            for name,val in expert_weight.items():
                name = re.sub(r'module\.','',name);
                z = re.match(r'layer(\d)',name);
                if z is not None:
                    id = z.groups()[0];
                    name=re.sub(r'layer(\d)',f'layers.{int(id)-1}',name);
                
                name = re.sub(r'downsample\.1','downsample.2',name);

                expert_weight_corrected[name]=val;
            expert_weight = expert_weight_corrected;
        elif model == 'vgg16':
            expert_weight = torch.load(f'{pt_path}/nvidia_resnet50_200821.pth.tar');
        elif model[:13] == 'resnet-50_dwt':     # resnet-50_dwt/bior4.4/Avg
            (wavename,_,downsample_ty) =model[14:].partition('/');
            expert_weight = torch.load(f'{pt_path}/expert.pt');
            # layer name correction
            expert_weight_corrected={}
            for name,val in expert_weight.items():
                name = re.sub(r'module\.','',name);
                z = re.match(r'layer(\d)',name);
                if z is not None:
                    id = z.groups()[0];
                    name=re.sub(r'layer(\d)',f'layers.{int(id)-1}',name);
                
                name = re.sub(r'downsample\.1','downsample.2',name);

                expert_weight_corrected[name]=val;
            expert_weight = expert_weight_corrected;
            # expert_weight = torch.load(f'{pt_path}/{wavename}/resnet50_dwt_{wavename}_256_best.pth.tar');
        elif model[:11] == 'vgg16_bn_dwt':
            (wavename,_,downsample_ty) =model[12:].partition('/');
            expert_weight = torch.load(f'{pt_path}/{wavename}/vgg16_bn_dwt_{wavename}_256_best.pth.tar');
    elif dataset in ('fvc2000', 'cifar10', 'cifar100', 'fp302'):
        if model == 'resnet-50':
            expert_weight = torch.load(f'{pt_path}/expert.pt');
        elif model == 'vgg16':
            expert_weight = torch.load(f'{pt_path}/expert.pt');
        elif model[:13] == 'resnet-50_dwt':     # resnet-50_dwt/bior4.4/Avg
            expert_weight = torch.load(f'{pt_path}/expert.pt');
            (wavename,_,downsample_ty) =model[14:].partition('/');
            # expert_weight = torch.load(f'{pt_path}/{wavename}/resnet50_dwt_{wavename}_256_best.pth.tar');
        elif model[:12] == 'vgg16_bn_dwt':
            expert_weight = torch.load(f'{pt_path}/expert.pt');
            (wavename,_,downsample_ty) =model[13:].partition('/');
            # expert_weight = torch.load(f'{pt_path}/{wavename}/vgg16_bn_dwt_{wavename}_256_best.pth.tar');
            
    if expert_weight == None:
        print(f"Pretrain: {model} and dataset {dataset} combination is not supported yet")
        exit(1)
    
    return expert_weight;