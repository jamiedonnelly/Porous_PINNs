import torch 
from torch import nn
import numpy as np
import pandas as pd
import os
import argparse
from src.models import JointModel, Model1, Model2
from torch.utils.data import DataLoader, Dataset

class Base(Dataset):
    def __init__(self, inputs: torch.tensor):
        self.X = inputs
        super().__init__()
        pass
    
    def __getitem__(self, index):
        return self.X[index]
    
    def __len__(self):
        return self.X.shape[0]
    
def inference(model: nn.Module, input: torch.tensor, time: float) -> None:
    dataset = Base(input)
    loader = DataLoader(dataset, args.batch_size)
    preds = []
    with torch.no_grad():
        for ix, input in enumerate(loader):
            if ix == 0:
                mu, sd = torch.mean(input,dim=0), torch.std(input,dim=0)
                print(f"Xu stats:\n{(mu[3],sd[3])}\nXv stats:\n{(mu[4],sd[4])}\nXp stats:\n{(mu[5],sd[5])}\n")
            t, x, y = input[:,0].unsqueeze(1), input[:,1].unsqueeze(1), \
            input[:,2].unsqueeze(1) 
            u, v, p, g = model(input)
            y = torch.concat([t+0.002, x, y, u, v, p, g],dim=1).double().reshape(input.shape[0],7)
            ydf = pd.DataFrame(y.numpy(),columns=['t','x','y','u','v','p','g'],dtype=np.float64)
            preds.append(ydf)
    df = pd.concat(preds,axis=0)
    fname = os.path.join(args.root, f"{time+0.002}.csv")
    df.to_csv(fname,index=False)
    print(f"{fname} saved...")

def scaling(data):
    data[['u','v']] = data[['u','v']]/0.004
    data['p'] = data['p']/1e4
    data.loc[data['g']<1e-2, 'g'] = 0.0
    return data

def inverse_scaling(data):
    data[['u','v']] = data[['u','v']]*0.004
    data['p'] = data['p']*1e4
    data.loc[data['g']<1e-2, 'g'] = 0.0
    return data

def process(file,transforms=False) -> torch.tensor:
    if transforms:
        data = scaling(pd.read_csv(file))
    else:
        data = pd.read_csv(file)
    time = data['t'].unique().item()
    X = torch.tensor(data[['t','x','y','u','v','p','g']].values).double()
    return time, X

def configure_model() -> nn.Module:
    model = JointModel()
    model.load_state_dict(args.model_path)
    return model

def latest_file(root):
    files = sorted([os.path.join(root,f) for f in os.listdir(root) if '.csv' in f])
    return files[-1]

def simulate(model):
    model.eval()
    breakthrough = False
    #while not breakthrough:
    for i in range(5):
        input_file = latest_file(args.root)
        if i == 0:
            transforms=True
        else:
            transforms=False
        time, X = process(input_file,transforms=transforms)
        print(time)
        inference(model, X, time)
        pass

def main():
    # parse arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str,help="Model path used for inference")
    parser.add_argument('--root',type=str,default='./',help="Directory to save predictions into")
    parser.add_argument('--batch_size',type=int,default=128000,help="")
    global args
    args = parser.parse_args()

    # initiate model
    m1, m2 = Model1(), Model2()
    model = JointModel(m1,  m2)
    model.load_state_dict(torch.load(args.model))

    # run inference
    simulate(model)
        
if __name__=="__main__":
    main()





