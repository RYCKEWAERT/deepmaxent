import torch.nn as nn
import torch.nn.functional as F

class poisson_loss(nn.Module):
    def __init__(self):
        super(poisson_loss, self).__init__()

    def forward(self, input, target):
        loss_lambda = F.poisson_nll_loss(input, target, log_input=True)
        return loss_lambda
    
class deepmaxent_loss(nn.Module):
    def __init__(self):
        super(deepmaxent_loss, self).__init__()
    def forward(self, input, target):
        loss = -((target)*(input.log_softmax(0))).mean(0).mean()
        return loss

class bce_loss(nn.Module):
    def __init__(self):
        super(bce_loss, self).__init__()
    def forward(self, lbd, target):
        t_n = target
        t_n[t_n > 0] = 1
        loss_BCE = F.binary_cross_entropy_with_logits(lbd, t_n)
        return loss_BCE

class ce_loss(nn.Module):
    def __init__(self):
        super(ce_loss, self).__init__()
        
    def forward(self, lbd, t_n):
        loss_CE = F.cross_entropy(lbd, t_n)
        return loss_CE