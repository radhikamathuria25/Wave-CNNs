import torch,torchvision
from src.models.model.resnet import *
from src.utils.logger import TBLogger

from torchvision.datasets import ImageNet
from src.datasets import init_dataloader
from src.models import init_net,load_pretrain_weights
from src.utils import log_activation_maps, log_activation_dwt_maps, log_energy_hist, register_activation_hook, train_model, test_model, mkdir

from src.DWT_IDWT.DWT_IDWT_layer import DWT_2D

activations={};

model_name='resnet-50';
dataset='imagenet';
device=torch.device('cuda:0');

logdir='log'
exp_label='EvalualteModel'

# Using Tensorboard Logger
logfile='tb_train';
logpath = logdir + '/' + exp_label + '/' + dataset + '/' + model_name + '/'
mkdir(logpath);

logger=TBLogger(f'{logpath}/{logfile}');

wavelet='bior4.4';

def define_test_hook(activations,logger,wavelet,device):
    def test_hook(x,step,ids=None):
        for i in range(len(x)):
            if ids:
                if i not in ids:
                    continue;
            inp_act_grid = torchvision.utils.make_grid(x,nrow=1);
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

# Load dataset
train_dl,test_dl,train_ds,test_ds = init_dataloader(dataset,'data/',64,512,num_workers=32);
# Load Model
# model = init_net('resnet-50','imagenet',device=device);
model = init_net(model_name,dataset,device=device);
parallel_model = torch.nn.DataParallel(model, device_ids = [0,1,2,3], output_device=0)
# state = model.state_dict();


register_activation_hook(activations,model,model_name,dataset);

test_hook_once = define_test_hook(activations,logger,wavelet,device);

# # Train Network
# train_model(parallel_model,train_dl,test_dl,'',
#                        lr0=0.001,a=0.001,b=0.75,momentum=0.9,weight_decay=0.0005,num_epoch=2,batch_size=64,
#                        loss_func = nn.CrossEntropyLoss(reduction='mean',label_smoothing=0.1),
#                        logger=logger,device=device,test_hook_once=test_hook_once);

scenario=''
step=0;
test_loss,test_acc = test_model(parallel_model,test_dl,
                        loss_func = nn.CrossEntropyLoss(reduction='mean',label_smoothing=0.1),
                        device=device,
                        test_hook_once=test_hook_once,iter=step);
print(f'\tTestLoss: {test_loss} \tTestAcc: {test_acc}')

if logger:
    logger.add_weight_dist(parallel_model.state_dict(),f"{scenario}",step=step);
    logger.add_scalars({"loss":test_loss.item()},f"test_loss/{scenario}",step=step);
    logger.add_scalars({"accuracy":test_acc.item()},f"test_accuracy/{scenario}",step=step);