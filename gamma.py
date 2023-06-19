# base
import argparse
import os
import time
import numpy as np 
import sys

# torch backends
import torch 
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader

# local 
from src.models import Model2
import src.dataset as dataset 
from src.backend_config import set_torch_config
from src.log import get_logger, logger

def load_data(name, root, cache_size=15):
    files = sorted([os.path.join(root,i) for i in os.listdir(root) if '.pt' in i])
    M = len(files)
    tfiles, vfiles = files[:int(M*0.75)], files[int(M*0.75):]
    traindata, valdata = dataset.DATASETS[name](tfiles, cache_size), dataset.DATASETS[name](vfiles, cache_size=3)
    trainsampler = DistributedSampler(traindata,\
                num_replicas=args.world_size,\
                rank=args.rank)
    valsampler = DistributedSampler(valdata,\
                num_replicas=args.world_size,\
                rank = args.rank)
    trainloader, valloader = DataLoader(traindata, 512000, sampler=trainsampler, pin_memory=True), \
    DataLoader(valdata, 512000, pin_memory=True, sampler=valsampler)
    return (trainloader,traindata), (valdata,valloader)

def optim_config(model):
    optim = torch.optim.Adam(model.parameters(), lr=args.lr0)
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optim,step_size=args.decay_freq,gamma=args.lr_decay)
        return optim, scheduler
    else:
        return optim
    
def config_loss():
    criterion = nn.MSELoss().to(args.gpu)
    return criterion

def config_distributed_model(base_model):
    # Send model to GPU 
    model = base_model.to(args.gpu)
    # Wrap model in DDP
    ddp = DDP(model, device_ids=[args.gpu])
    return ddp

def validate(epoch, model, valloader, file=False):
    criterion = nn.MSELoss().to(args.gpu) 
    model.module.eval()
    preds, true = [], []
    with torch.no_grad():
        for ix, (x,y) in enumerate(valloader):
            x, g = x.to(args.gpu), y.to(args.gpu).unsqueeze(1)
            ghat = model(x)
            preds.append(ghat)
            true.append(g) 
    # This is often a source of a CUDA OOM error - for some reason mis-matched shapes (e.g., [n,x] vs [n,1,x]) causes huge 
    # attempted memory allocation (I have seen up to 1TB!!!)
    preds, true = torch.concat(preds,dim=0), torch.concat(true,dim=0)
    preds, true = preds.reshape(preds.shape[0],2), true.reshape(preds.shape[0],2)
    rmse = torch.sqrt(criterion(preds,true))
    dist.all_reduce(rmse)
    if args.rank==0:
        logger(args, f"Validation RMSE: {rmse.item()/args.world_size:.8f}")
        if file:
            torch.save((preds.cpu(),true.cpu()),os.path.join('./results',f"{args._name}{os.environ['SLURM_JOB_ID']}_Epoch{epoch}.pt"))
        dist.barrier()
    else:
        dist.barrier() 

def train(model, traindata, trainloader, optim, valdata=False, valloader=False, scheduler=False):

    # Time model training 
    start=time.time()

    # Config loss and model 
    criterion = config_loss()
    ddp_model = config_distributed_model(model)
    logger(args, "Beginning training...")
    batch_count = 0
    for epoch in range(args.epochs):
        if args.scheduler:
            logger(args,f"Epoch {epoch+1} LR: {scheduler.get_last_lr()[0]}")
        # In-training validation
        validate(epoch ,ddp_model, valloader, True)
        logger(args,f"Epoch {epoch}")
        ddp_model.module.train()
        torch.cuda.empty_cache()
        for (large_x, large_y) in trainloader:
            
            large_x, large_y = large_x.to(args.gpu), large_y.to(args.gpu)

            for index in range(0, large_x.shape[0]-args.batch_size, args.batch_size):
                x, g = large_x[index:index+args.batch_size], large_y[index:index+args.batch_size].unsqueeze(1)
                
                optim.zero_grad()
                ghat = ddp_model(x)
                loss = criterion(ghat,g)
                loss.backward()
                optim.step()
                
                if batch_count % args.print_freq == 0:
                    logger(args,f"\tEpoch {epoch} Batch {batch_count+1} Loss: {loss.item():.8f}")
                
                batch_count += 1
        traindata.load_new_epoch()
        valdata.load_new_epoch()
        if args.scheduler:
            scheduler.step()
    logger(args,f"Training finished in...{(time.time()-start)/60:.4f} minutes")
    return ddp_model.module


def main():
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=5,\
                        help="Number of epochs to train the network for")
    parser.add_argument('--batch_size',type=int,default=64,\
                        help="Size of minibatch")
    parser.add_argument('--lr0',type=float,default=1e-3,\
                        help="""Initial learning rate. In the case of no scheduler
                                this is the constant learning rate.""")
    parser.add_argument('--scheduler',type=bool,default=False,
                        help="""Whether a learning rate scheduler is used. If true, 
                        the default is exponential LR decay.""")
    parser.add_argument('--lr_decay',type=float,default=0.75,\
                        help="Decay rate for learning rate scheduler.")
    parser.add_argument('--decay_freq',type=int,default=5,\
                        help="""Frequency of learning rate decays in number of epochs.
                                I.e., if set to `1` then decay occurs at the end of each epoch.""")
    parser.add_argument('--print_freq',type=int,default=100,\
                        help="How frequently to log training performance in number of batches.")
    parser.add_argument('--save_freq',type=int,default=5,\
                        help="How frequently should the model be saved during training (in epochs).")
    # parse args
    global args
    args = parser.parse_args()
    # Parse script name from `sys` to use for naming outputs and logs
    args._name = sys.argv[0].replace('.py','')
    # configure logger 
    args.log = get_logger('./logs',f"{args._name}{os.environ['SLURM_JOB_ID']}.log")
    # Set world_size to total number of processes
    args.world_size = int(os.environ['SLURM_NPROCS'])
    # Set rank to process_id
    args.rank = int(os.environ['SLURM_PROCID'])
    # GPU is equivalent to local rank
    # n_processes_per_node = gpus_per_node
    args.gpu = int(os.environ['SLURM_LOCALID'])

    # Set backend config - disables reduced precision
    set_torch_config()

    # Init parallel backend
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size = args.world_size,
        rank = args.rank
    )
    logger(args,"Process group initialised...")

    # Load data
    (trainloader,traindata), (valdata,valloader) = load_data(args._name, './data')
    logger(args,"Data loaded...")
    
    # Config model
    model = Model2(1)

    # Load optimiser
    if args.scheduler:
        optim, scheduler = optim_config(model)
    else:
        optim, scheduler = optim_config(model), False

    model = train(model, traindata, trainloader, optim, valdata, valloader, scheduler)

    if args.rank==0:
        torch.save(model.cpu().state_dict(),os.path.join("./models",f"{args._name}{os.environ['SLURM_JOB_ID']}.pth"))

if __name__=="__main__":
    main()
