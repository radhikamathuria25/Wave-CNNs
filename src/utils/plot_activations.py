import torch
import torchvision


def log_activation_maps(activations,logger,ids=None,step=0):
    for name,val in activations.items():
        for i in range(len(val)):
            if ids:
                if i not in ids:
                    continue;
            val_chan = val[i].unsqueeze(1);
            act_grid = torchvision.utils.make_grid(val_chan,nrow=16);
            logger.add_imagegrid(f'{i}/{name}',act_grid,step=step);

def log_activation_dwt_maps(ll,lh,hl,hh,logger,wavelet,ids=None,step=0):        

    for name in ll.keys():
        val_ll = ll[name];
        val_lh = lh[name];
        val_hl = hl[name];
        val_hh = hh[name];
        for i in range(len(val_ll)):
            if ids:
                if i not in ids:
                    continue;
            val_chan_ll = val_ll[i].unsqueeze(1);
            val_chan_lh = val_lh[i].unsqueeze(1);
            val_chan_hl = val_hl[i].unsqueeze(1);
            val_chan_hh = val_hh[i].unsqueeze(1);
            act_grid_ll = torchvision.utils.make_grid(val_chan_ll,nrow=16);
            act_grid_lh = torchvision.utils.make_grid(val_chan_lh,nrow=16);
            act_grid_hl = torchvision.utils.make_grid(val_chan_hl,nrow=16);
            act_grid_hh = torchvision.utils.make_grid(val_chan_hh,nrow=16);
            logger.add_imagegrid(f'{i}/{name}/dwt({wavelet})/LL',act_grid_ll,step=step);
            logger.add_imagegrid(f'{i}/{name}/dwt({wavelet})/LH',act_grid_lh,step=step);
            logger.add_imagegrid(f'{i}/{name}/dwt({wavelet})/HL',act_grid_hl,step=step);
            logger.add_imagegrid(f'{i}/{name}/dwt({wavelet})/HH',act_grid_hh,step=step);
        

def log_energy_hist(ll,lh,hl,hh,logger,ids=None,step=0):
    # Calculate Energy
    energy = {};
    energy['ll']= {};
    energy['lh']= {};
    energy['hl']= {};
    energy['hh']= {};
    for name in ll.keys():
        val_ll = ll[name];
        val_lh = lh[name];
        val_hl = hl[name];
        val_hh = hh[name];

        energy_ll = val_ll.pow(2).sum(dim=(2,3));
        energy_lh = val_lh.pow(2).sum(dim=(2,3));
        energy_hl = val_hl.pow(2).sum(dim=(2,3));
        energy_hh = val_hh.pow(2).sum(dim=(2,3));
        
        energy_tot = energy_ll + energy_lh + energy_hl + energy_hh;
        
        energy['ll'][name] = energy_ll/energy_tot;
        energy['lh'][name] = energy_lh/energy_tot;
        energy['hl'][name] = energy_hl/energy_tot;
        energy['hh'][name] = energy_hh/energy_tot;
        
    for name in ll.keys():
        for i in range(len(energy['ll'][name])):
            if ids:
                if i not in ids:
                    continue;
            logger.add_histogram(energy['ll'][name][i],f"{i}/{name}/Energy[ll]",step=step);
            logger.add_histogram(energy['lh'][name][i],f"{i}/{name}/Energy[lh]",step=step);
            logger.add_histogram(energy['hl'][name][i],f"{i}/{name}/Energy[hl]",step=step);
            logger.add_histogram(energy['hh'][name][i],f"{i}/{name}/Energy[hh]",step=step);