import torch
from torch import nn

class ContinuityLoss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, X, uhat, vhat):
        du_dX = torch.autograd.grad(
        inputs=X, 
        outputs=uhat, 
        grad_outputs=torch.ones_like(uhat), 
        retain_graph=True, 
        create_graph=True
        )[0].flatten(0)
        dv_dX = torch.autograd.grad(
        inputs=X, 
        outputs=vhat, 
        grad_outputs=torch.ones_like(vhat), 
        retain_graph=True, 
        create_graph=True
        )[0].flatten(0)
        resid = du_dX[0] + dv_dX[1]
        return torch.linalg.norm(resid, dim=0, ord=2)


class MomentumLoss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        pass
    
    def _u_momentum(self, X, uhat, phat):
        du_dX = torch.autograd.grad(
        inputs=X, 
        outputs=uhat, 
        grad_outputs=torch.ones_like(uhat), 
        retain_graph=True, 
        create_graph=True
        )[0].flatten(0)
        dp_dX = torch.autograd.grad(
        inputs=X, 
        outputs=phat, 
        grad_outputs=torch.ones_like(phat), 
        retain_graph=True, 
        create_graph=True
        )[0].flatten(0)
        du_dt, du_dx, dp_dx = du_dX[0], du_dX[1], dp_dX[1]
        
        resid = du_dt + (uhat*du_dx) - 9.81 + dp_dx        

        return torch.linalg.norm(resid,dim=0,ord=2)
    
    def _v_momentum(self, X, vhat, phat):
        dv_dX = torch.autograd.grad(
        inputs=X, 
        outputs=vhat, 
        grad_outputs=torch.ones_like(vhat), 
        retain_graph=True, 
        create_graph=True
        )[0].flatten(0)
        dp_dX = torch.autograd.grad(
        inputs=X, 
        outputs=phat, 
        grad_outputs=torch.ones_like(phat), 
        retain_graph=True, 
        create_graph=True
        )[0].flatten(0)
        dv_dt, dv_dy, dp_dy = dv_dX[0], dv_dX[2], dp_dX[2]
        
        resid = dv_dt + (vhat*dv_dy) - 9.81 + dp_dy       

        return torch.linalg.norm(resid,dim=0,ord=2)
    
    def forward(self, X, uhat, vhat, phat):
        return self._u_momentum(X, uhat, phat) + self._v_momentum(X, vhat, phat)
    
    
class AdvectionLoss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, X, ghat, uhat, vhat):
        dg_dX = torch.autograd.grad(
        inputs=X, 
        outputs=ghat, 
        grad_outputs=torch.ones_like(ghat), 
        retain_graph=True, 
        create_graph=True
        )[0].flatten(0)
        dgu_dX = torch.autograd.grad(
        inputs=X, 
        outputs=ghat*uhat, 
        grad_outputs=torch.ones_like(ghat), 
        retain_graph=True, 
        create_graph=True
        )[0].flatten(0)
        dgv_dX = torch.autograd.grad(
        inputs=X, 
        outputs=ghat*vhat,
        grad_outputs=torch.ones_like(ghat), 
        retain_graph=True, 
        create_graph=True
        )[0].flatten(0)
        resid = dg_dX[0] + dgu_dX[1] + dgv_dX[2]
        return torch.linalg.norm(resid, dim=0, ord=2)