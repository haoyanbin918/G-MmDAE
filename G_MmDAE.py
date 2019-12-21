import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
#
def norm(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)
#
class Net0(nn.Module):
    def __init__(self, conf):
        super(Net0, self).__init__()
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

        self.m1drop = nn.Dropout(conf.drp)

        #-----------------to m1 self-----------------
        self.m1S3_bias = nn.Parameter(torch.zeros(self.conf.hdn1))

        self.m1S4 = nn.Sequential(
                nn.Linear(self.conf.hdn1, self.conf.m1Size),
                nn.Tanh(),
            )
        #--------------------end---------------------
        
        #-----------------to m2---------------------
        self.m1C3_bias = nn.Parameter(torch.zeros(self.conf.hdn2))

        self.m1C4 = nn.Sequential(
                nn.Linear(self.conf.hdn2, self.conf.m2Size),
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

        self.m2drop = nn.Dropout(conf.drp)

        #------------------to m2 self-----------------------
        
        self.m2S3_bias = nn.Parameter(torch.zeros(self.conf.hdn2))

        self.m2S4 = nn.Sequential(
                nn.Linear(self.conf.hdn2, self.conf.m2Size),
                nn.Tanh(),
            )

        #-------------------to m1--------------------

        self.m2C3_bias = nn.Parameter(torch.zeros(self.conf.hdn1))

        self.m2C4 = nn.Sequential(
                nn.Linear(self.conf.hdn1, self.conf.m1Size),
                nn.Tanh(),
            )
        #--------------------end---------------------


    def forward(self, x1, x2):
    	if self.conf.input_norm:
            x1 = norm(x1)
            x2 = norm(x2)
        #
        m1EMB = self.m1L1(x1)
        m1EMB = self.m1R2(m1EMB)

        m1OUTd = self.m1drop(m1EMB)

        m1OUTs = F.linear(m1OUTd, self.m1R2[0].weight.t(), self.m1S3_bias)
        m1OUTs = F.tanh(m1OUTs)
        m1OUTs = self.m1S4(m1OUTs)
        #

        m1OUTc = F.linear(m1OUTd, self.m2R2[0].weight.t(), self.m1C3_bias)
        m1OUTc = F.tanh(m1OUTc)
        m1OUTc = self.m1C4(m1OUTc)

        #=========================

        m2EMB = self.m2L1(x2)
        m2EMB = self.m2R2(m2EMB)

        m2OUTd = self.m2drop(m2EMB)

        m2OUTs = F.linear(m2OUTd, self.m2R2[0].weight.t(), self.m2S3_bias)
        m2OUTs = F.tanh(m2OUTs)
        m2OUTs = self.m2S4(m2OUTs)

        m2OUTc = F.linear(m2OUTd, self.m1R2[0].weight.t(), self.m2C3_bias)
        m2OUTc = F.tanh(m2OUTc)
        m2OUTc = self.m2C4(m2OUTc)    

        return m1OUTs, m1OUTc, m2OUTs, m2OUTc