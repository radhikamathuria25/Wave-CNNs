import torch
import numpy as np

import sys
import datetime
import random


from src.models import *
from src.tasks import *
from src.utils import * 
from src.tasks import *


if __name__ == '__main__':
    print('-'*40)
    
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.gpu) ## Setting cuda on GPU 
    logpath = get_logpath(args);
    pt_path = args.ptdir + '/' + args.dataset + '/' + args.model + '/';
    mkdir(logpath)
    mkdir(pt_path)        
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    
    sys.stdout = StdOutLogger(logpath,args.log_filename)
    
    log = TBLogger(logpath,args.log_filename);
    
    if args.task == 'feature_extract':
        run_feature_extract()(args,logger=log);
    elif args.task == 'feature_evolve':
        run_feature_evolve(args,logger=log);
    