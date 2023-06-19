import torch 
from torch import nn

class Base(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def _params(self):
        for layer in self.model:
            if (hasattr(layer,'weight') & hasattr(layer, 'bias')) & ~(isinstance(layer, nn.BatchNorm1d)):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.uniform_(layer.bias)
                
    def forward(self, x):
        raise NotImplementedError


class Model1(Base):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(7,16),
                      nn.BatchNorm1d(16),
                      nn.ReLU(True),
                      nn.Linear(16,32),
                      nn.BatchNorm1d(32),
                      nn.ReLU(True),
                      nn.Linear(32,64),
                      nn.BatchNorm1d(64),
                      nn.ReLU(True),
                      nn.Linear(64,128),
                      nn.BatchNorm1d(128),
                      nn.ReLU(True),
                      nn.Linear(128,256),
                      nn.BatchNorm1d(256),
                      nn.ReLU(True),
                      nn.Linear(256,512),
                      nn.BatchNorm1d(512),
                      nn.ReLU(True),
                      nn.Linear(512,1024),
                      nn.BatchNorm1d(1024),
                      nn.ReLU(True),
                      nn.Linear(1024,2048),
                      nn.BatchNorm1d(2048),
                      nn.ReLU(True),
                      nn.Linear(2048,2048),
                      nn.BatchNorm1d(2048),
                      nn.ReLU(True),
                      nn.Linear(2048,1024),
                      nn.BatchNorm1d(1024),
                      nn.ReLU(True),  
                      nn.Linear(1024,512),
                      nn.BatchNorm1d(512),
                      nn.ReLU(True),                                    
                      nn.Linear(512,256),
                      nn.BatchNorm1d(256),
                      nn.ReLU(True),
                      nn.Linear(256,128),
                      nn.BatchNorm1d(128),
                      nn.ReLU(True),
                      nn.Linear(128,64),
                      nn.BatchNorm1d(64),
                      nn.ReLU(True),
                      nn.Linear(64,32),
                      nn.BatchNorm1d(32),
                      nn.ReLU(True),
                      nn.Linear(32,16),
                      nn.BatchNorm1d(16),
                      nn.ReLU(True),
                      nn.Linear(16,6),
                      nn.BatchNorm1d(6),
                      nn.ReLU(True),
                      nn.Linear(6,3))
        super()._params()
        self.model = self.model.double()

    def forward(self, input):
        output = self.model(input)
        return output[:,0].unsqueeze(1), \
        output[:,1].unsqueeze(1), output[:,2].unsqueeze(1)

class Model2(Base):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(7,16),
                      nn.BatchNorm1d(16),
                      nn.ReLU(True),
                      nn.Linear(16,32),
                      nn.BatchNorm1d(32),
                      nn.ReLU(True),
                      nn.Linear(32,64),
                      nn.BatchNorm1d(64),
                      nn.ReLU(True),
                      nn.Linear(64,128),
                      nn.BatchNorm1d(128),
                      nn.ReLU(True),
                      nn.Linear(128,256),
                      nn.BatchNorm1d(256),
                      nn.ReLU(True),
                      nn.Linear(256,512),
                      nn.BatchNorm1d(512),
                      nn.ReLU(True),
                      nn.Linear(512,1024),
                      nn.BatchNorm1d(1024),
                      nn.ReLU(True),
                      nn.Linear(1024,2048),
                      nn.BatchNorm1d(2048),
                      nn.ReLU(True),
                      nn.Linear(2048,2048),
                      nn.BatchNorm1d(2048),
                      nn.ReLU(True),
                      nn.Linear(2048,1024),
                      nn.BatchNorm1d(1024),
                      nn.ReLU(True),  
                      nn.Linear(1024,512),
                      nn.BatchNorm1d(512),
                      nn.ReLU(True),    
                      nn.Linear(512,256),
                      nn.BatchNorm1d(256),
                      nn.ReLU(True),
                      nn.Linear(256,128),
                      nn.BatchNorm1d(128),
                      nn.ReLU(True),
                      nn.Linear(128,64),
                      nn.BatchNorm1d(64),
                      nn.ReLU(True),
                      nn.Linear(64,32),
                      nn.BatchNorm1d(32),
                      nn.ReLU(True),
                      nn.Linear(32,16),
                      nn.BatchNorm1d(16),
                      nn.ReLU(True),
                      nn.Linear(16,4),
                      nn.BatchNorm1d(4),
                      nn.ReLU(True),
                      nn.Linear(4,1),
                      nn.Sigmoid())
        super()._params()
        self.model = self.model.double()

    def forward(self, input):   
        return self.model(input)

class JointModel(nn.Module):
    def __init__(self, model1, model2):
        super().__init__()
        self.model1, self.model2 = model1, model2
        # Init params of both models
        JointModel._params(self.model1.model)
        JointModel._params(self.model2.model)
        pass

    def _params(model):
        for layer in model:
            if (hasattr(layer,'weight') & hasattr(layer, 'bias')) & ~(isinstance(layer, nn.BatchNorm1d)):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.uniform_(layer.bias)

    def forward(self, input):
        t, x, y, u0, v0, p0, g0 = input[:,0].unsqueeze(1), input[:,1].unsqueeze(1),\
        input[:,2].unsqueeze(1), input[:,3].unsqueeze(1), input[:,4].unsqueeze(1),\
        input[:,5].unsqueeze(1), input[:,6].unsqueeze(1)
        in1 = torch.concat([t,x,y,u0,v0,p0,g0],dim=1)
        u1, v1, p1 = self.model1(in1)
        g1 = self.model2(torch.concat([t,x,y,u1,v1,p1,g0],dim=1))
        return  u1, v1, p1, g1
        
