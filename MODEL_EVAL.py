import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
#
def norm(input, p=2, dim=1, eps=1e-08):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)
#
class MODEL_EVAL(nn.Module):
    def __init__(self, conf):
        super(MODEL_EVAL, self).__init__()
        self.conf = conf
        #------------m1----------------------------
        self.m1L1 = nn.Sequential(
                nn.Linear(self.conf.m1Size, self.conf.hdn1),
                nn.Tanh(),
            )

        self.m1R2 = nn.Sequential(
                nn.Linear(self.conf.hdn1, self.conf.rep),
                nn.Tanh(),
            )
        #=========================================================
        #--------------m2------------------------------
        #=========================================================
        
        self.m2L1 = nn.Sequential(
                nn.Linear(self.conf.m2Size, self.conf.hdn2),
                nn.Tanh(),
            )

        self.m2R2 = nn.Sequential(
                nn.Linear(self.conf.hdn2, self.conf.rep),
                nn.Tanh(),
            )


    def forward(self, x1, x2):
        if self.conf.input_norm:
            x1 = norm(x1)
            x2 = norm(x2)
        #s
        m1EMB = self.m1L1(x1)
        m1EMB = self.m1R2(m1EMB)

        #=========================
        m2EMB = self.m2L1(x2)
        m2EMB = self.m2R2(m2EMB)

        if self.conf.emb_norm:
            m1EMB = norm(m1EMB)
            m2EMB = norm(m2EMB)

        return m1EMB, m2EMB
