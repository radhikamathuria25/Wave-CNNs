import torch,torchvision
from src.models import *

from src.datasets import init_dataloader
from src.models import init_net,load_pretrain_weights
from src.utils import define_test_hook, register_activation_hook, get_logpath, glob_re, train_model


activations={};

def run_feature_evolve(args,logger=None):
    
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
    train_dl,test_dl,train_ds,test_ds = init_dataloader(args.dataset,args.datadir,args.batch_size,4*args.batch_size,num_workers=args.num_workers);
    
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
                
    # parallel_net = torch.nn.DataParallel(net, device_ids = [0,1,2,3], output_device=args.device)        
    
    # Eval        
    register_activation_hook(activations,net,args.model,args.dataset);
    test_hook_once = define_test_hook(activations,logger,args.wavelet,args.device)

    # Train Network
    print("Start Training ! ")
    w_expert,_,_,_=train_model(net,train_dl,test_dl,'',
                        lr0=args.lr0,a=args.lr_a,b=args.lr_b,momentum=args.momentum,weight_decay=args.w_decay,num_epoch=args.num_epoch,batch_size=args.batch_size,
                        loss_func = nn.CrossEntropyLoss(reduction='mean',label_smoothing=args.lbl_sm),
                        logger=logger,device=args.device,test_hook_once=test_hook_once,step0=0,
                        resume=args.resume_train);
    print("Training Done! ")

    print("Saving Model ... ")
    ptpath=args.ptdir + '/' + args.dataset + '/' + args.model + '/';
    torch.save(w_expert,ptpath + 'expert.pt');
    print("Done Saving Model! ")