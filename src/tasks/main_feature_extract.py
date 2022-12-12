import torch,torchvision
from src.models import *

from src.datasets import init_dataloader
from src.models import init_net,load_pretrain_weights
from src.utils import define_test_hook, register_activation_hook


activations={};

def run_feature_extract(args,logger=None):
    
    ################################### build model
    print('-'*40)
    print('Building model')
    print(f'MODEL: {args.model}, Dataset: {args.dataset}')
    net = init_net(args.model,args.dataset,device=args.device);
    w_net = net.state_dict();
    print('-'*40)
    print(net)
    print('')
    
    ##################################### Load Dataset
    print('-'*40)
    print('Loading Dataset')
    train_dl,test_dl,train_ds,test_ds = init_dataloader(args.dataset,args.datadir,args.batch_size,args.batch_size);
    
    ##################################### Load Pretrain Weights
    if args.init_weights==1:
        print('-'*40)
        print('Loading Pretrained Weights')
        expert_weight = load_pretrain_weights(args.ptdir,args.model,args.dataset);
        net.load_state_dict(expert_weight);
        net.to(args.device);
    
    if logger:
        x,_ = next(iter(train_dl));
        logger.add_graph(net,x.to(args.device));
        logger.add_weight_dist(w_net,"Weights");
    

    # Eval        
    register_activation_hook(activations,net,args.model,args.dataset);
    test_hook = define_test_hook(activations,logger,args.wavelet,args.device)

    x,_ = next(iter(test_dl));
    x = x.to(args.device)

    net.eval();
    output = net(x);

    test_hook(x,step=0);
    
    print(f'Done extracting activations');
