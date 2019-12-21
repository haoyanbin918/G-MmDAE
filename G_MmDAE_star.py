import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
#
def norm(input, p=2, dim=1, eps=1e-08):
    return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)

def compcosdis(EMBs):
	W12 = torch.mm(EMBs, EMBs.transpose(0, 1))
	#
	W1 = torch.norm(EMBs, 2, 1)
	W1 = torch.stack([W1], 1)
	#
	W2 = torch.norm(EMBs, 2, 1)
	W2 = torch.stack([W2], 1)
	#
	W = torch.mm(W1, W2.transpose(0, 1))
	#
	dis = 1 - W12 / W.clamp(min=1e-08)
	#
	return dis

def computeprob(dises):
	probs = F.softmax(-dises*dises, dim=1)
	diag = torch.diag(torch.diag(probs))
	probs = probs - diag
	sum_p = torch.sum(probs, 1, keepdim=True).expand_as(probs)
	probs = probs / sum_p

	return probs.clamp(min=1e-08)
#
class Net(nn.Module):
    def __init__(self, conf):
        super(Net, self).__init__()
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
        m1OUTs = torch.tanh(m1OUTs)
        m1OUTs = self.m1S4(m1OUTs)
        #

        m1OUTc = F.linear(m1OUTd, self.m2R2[0].weight.t(), self.m1C3_bias)
        m1OUTc = torch.tanh(m1OUTc)
        m1OUTc = self.m1C4(m1OUTc)

        #=========================

        m2EMB = self.m2L1(x2)
        m2EMB = self.m2R2(m2EMB)

        m2OUTd = self.m2drop(m2EMB)

        m2OUTs = F.linear(m2OUTd, self.m2R2[0].weight.t(), self.m2S3_bias)
        m2OUTs = torch.tanh(m2OUTs)
        m2OUTs = self.m2S4(m2OUTs)

        m2OUTc = F.linear(m2OUTd, self.m1R2[0].weight.t(), self.m2C3_bias)
        m2OUTc = torch.tanh(m2OUTc)
        m2OUTc = self.m2C4(m2OUTc) 

        #---- for KLDiv-----
        #m12EMBs  = torch.cat([m1EMB, m2EMB], 1)
        if self.conf.emb_norm:
            m1EMB = norm(m1EMB)
            m2EMB = norm(m2EMB)
        #
        dis_cos_1 = compcosdis(m1EMB)
        Q1 = computeprob(dis_cos_1)
        Q1 = torch.log(Q1)

        dis_cos_2 = compcosdis(m2EMB)
        Q2 = computeprob(dis_cos_2)
        Q2 = torch.log(Q2)

        # 
        return Q1, Q2, m1OUTs, m1OUTc, m2OUTs, m2OUTc