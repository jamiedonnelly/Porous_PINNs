import random
import torch 
import os
from torch.utils.data import Dataset, DataLoader 
import torch.nn.functional as F
import numpy as np 

class BaseDataset(Dataset):
    def __init__(self, files, cache_size=15):
        super().__init__
        self.files, self.cache_size = sorted(files), cache_size
        self.indices = torch.load('/share/home2/donnel39/porous/misc/training_indexes.pt')
        self.load_new_epoch()
        print(f"Dataset length: {self.__len__()}")
        pass

    def load_new_epoch(self):
        if self.cache_size > len(self.files):
            epoch_files = self.files
        else:
            epoch_files = np.random.choice(self.files, size=self.cache_size, replace=False)
        x, y = [], []
        for file in epoch_files:
            i, o = torch.load(file)
            x.append(i[self.indices])
            y.append(o[self.indices])
        self.x, self.y = torch.concat(x, dim=0), torch.concat(y, dim=0)
        self.apply_transforms()

    def apply_transforms(self):
        raise NotImplementedError
    
    def __getitem__(self,index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        if not hasattr(self, 'x') or not hasattr(self, 'y'):
            return 0
        assert isinstance(self.x, torch.Tensor) and isinstance(self.y, torch.Tensor), "Data needs to be of type `torch.Tensor`"
        assert self.x.shape[0]==self.y.shape[0], "Input and output tensors need to have the same shape at the 0 axis"
        return len(self.x)

class VelocityData(BaseDataset):
    def __init__(self, files, cache_size=15):
        super().__init__(files,cache_size)

    def apply_transforms(self):
        self.y = self.y[:,:-2]
        # Velocity scaling 
        self.x[:, 3:5] /= 0.004
        self.y /= 0.004
        # pressure scaling
        self.x[:, 5] /= 1e4
        # gamma truncating
        self.x[:,-1][self.x[:,-1]<1e-2] = 0.0

class PressureData(BaseDataset):
    def __init__(self, files, cache_size=15):
        super().__init__(files,cache_size)

    def apply_transforms(self):
        self.y = self.y[:,-2]
        # Velocity scaling
        self.x[:, 3:5] /= 0.004
        # Pressure scaling 
        self.x[:, 5] /= 1e4
        # Gamma truncating 
        self.x[:,-1][self.x[:,-1]<1e-2] = 0.0   

class PhaseData(BaseDataset):
    def __init__(self, files, cache_size=15):
        super().__init__(files, cache_size)
    
    def apply_transforms(self):
        # Velocity scaling 
        self.x[:, 3:5] /= 0.004
        self.y[:, :2] /= 0.004
        # pressure scaling
        self.x[:, 5] /= 1e4
        self.y[:, 2] /= 1e4
        # gamma truncating
        self.x[:,-1][self.x[:,-1]<1e-2] = 0.0
        self.y[:,-1][self.y[:,-1]<1e-2] = 0.0
        # rearranging input 
        self.x = torch.cat([self.x[:,:3], self.y[:,:-1], self.x[:,-1].unsqueeze(1)], dim=1)
        self.y = self.y[:,-1]
        
class JointData(BaseDataset):
    def __init__(self, files, cache_size=15):
        super().__init__(files, cache_size)

    def apply_transforms(self):
        # Velocity scaling
        self.x[:, 3:5] /= 0.004
        self.y[:, :2] /= 0.004
        # pressure scaling
        self.x[:, 5] /= 1e4
        self.y[:, 2] /= 1e4
        # gamma truncating
        self.x[:,-1][self.x[:,-1]<1e-2] = 0.0
        self.y[:,-1][self.y[:,-1]<1e-2] = 0.0


DATASETS = {'velocity':VelocityData,'gamma':PhaseData,'pressure':PressureData,'pinns':JointData}
