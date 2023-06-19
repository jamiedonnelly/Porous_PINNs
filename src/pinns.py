import torch 
from torch import nn 
from torch.autograd import grad
from typing import List


def fc(layer_shapes: List[List[int]]) -> nn.Sequential:
    base = nn.Sequential()
    for ix, layer in enumerate(layer_shapes):
        if ix == len(layer_shapes)-1:
            # No activation for the output layer
            new_layer = nn.Linear(layer[0],layer[1])
            nn.init.xavier_normal_(new_layer.weight)
            nn.init.constant_(new_layer.bias,0)
            base.append(new_layer)
        else:
            new_layer = nn.Linear(layer[0],layer[1])
            nn.init.xavier_normal_(new_layer.weight)
            nn.init.constant_(new_layer.bias,0)
            base.append(new_layer)
            base.append(nn.ReLU(True))
    return base

