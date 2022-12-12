import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from torchvision.datasets import ImageNet
from src.transform import *
from src.datasets import *
# from torchvision.datasets.folder import IMG_EXTENSIONS

def init_dataloader(dataset,datadir,train_bs,test_bs,num_workers=1):    
    
    if dataset == 'imagenet':
        ds_obj = ImageNet;
        datadir = f'{datadir}/imagenet';
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),    
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        target_transform=None;
        
    elif dataset == 'fvc2000':
        folder='FVC2000';
        # IMG_EXTENSIONS.append('tif');
        ds_obj = ImageFolder_custom;
        datadir = f'{datadir}/{folder}';
        transform_train = transforms.Compose([
                transforms.RandomRotation(45),    
                transforms.RandomResizedCrop(224),    
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        target_transform=None;
        
    elif dataset == 'fp302':
        folder='fp302';
        # IMG_EXTENSIONS.append('tif');
        ds_obj = ImageFolder_custom;
        datadir = f'{datadir}/{folder}';
        transform_train = transforms.Compose([
                transforms.RandomRotation(60),    
                transforms.RandomResizedCrop(32),    
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        target_transform=None;
        
    elif dataset == 'fp302u':
        folder='fp302';
        # IMG_EXTENSIONS.append('tif');
        ds_obj = ImageFolder_custom;
        datadir = f'{datadir}/{folder}';
        transform_train = transforms.Compose([
                transforms.RandomRotation(60),    
                transforms.RandomResizedCrop(224),    
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        target_transform=None;
        
    elif dataset == 'tinyimagenet':
        ds_obj = TinyImageNet
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),    
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        target_transform=None;
        
    elif dataset in ('cifar10','cifar100'):
        transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.RandomResizedCrop(224),    
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                # transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        transform_test = transforms.Compose([
                # transforms.ToPILImage(),
                # transforms.Resize(256),
                # transforms.CenterCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        target_transform=None;
        if dataset == 'cifar10':
            ds_obj = CIFAR10;
        elif dataset == 'cifar100':
            ds_obj = CIFAR100;
        
    elif dataset in ('cifar10u','cifar100u'):
        transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224),    
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        transform_test = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        target_transform=None;
        if dataset == 'cifar10u':
            ds_obj = CIFAR10;
        elif dataset == 'cifar100u':
            ds_obj = CIFAR100;
        
    elif dataset[:11] == 'imagefolder':
        folder=dataset[12:];
        ds_obj = ImageFolder_custom;
        datadir = f'{datadir}/{folder}';
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),    
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        target_transform=None;
        
    else:
        print(f"Dataset {dataset} is not supported yet");
        exit(1);
        
    train_ds = ds_obj(datadir, 'train', transform=transform_train, target_transform=target_transform);
    test_ds = ds_obj(datadir, 'val', transform=transform_test, target_transform=target_transform);
    
    train_dl = DataLoader(dataset=train_ds, batch_size=train_bs, pin_memory=False, shuffle=True, drop_last=False,num_workers=num_workers,persistent_workers=True);
    test_dl = DataLoader(dataset=test_ds, batch_size=test_bs, pin_memory=False, shuffle=False, drop_last=False,num_workers=num_workers,persistent_workers=True);
    
    return train_dl,test_dl,train_ds,test_ds;