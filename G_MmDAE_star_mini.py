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

        #self.m1drop = nn.Dropout(conf.drp)
        
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

        #self.m2drop = nn.Dropout(conf.drp)
        #--------------------end---------------------
        

    def forward(self, x1, x2):
        m1EMB = self.m1L1(x1)
        m1EMB = self.m1R2(m1EMB)


        #m1OUTd = self.m1drop(m1EMB)

        #=========================
        m2EMB = self.m2L1(x2)
        m2EMB = self.m2R2(m2EMB)

        if self.conf.emb_norm:
            m1EMB = norm(m1EMB)
            m2EMB = norm(m2EMB)
        #m2OUTd = self.m2drop(m2EMB)

        #---- for KLDiv-----
        #m12EMBs  = torch.cat([m1EMB, m2EMB], 1)
        dis_cos_1 = compcosdis(m1EMB)
        Q1 = computeprob(dis_cos_1)
        Q1 = torch.log(Q1)

        dis_cos_2 = compcosdis(m2EMB)
        Q2 = computeprob(dis_cos_2)
        Q2 = torch.log(Q2)

        # 
        return Q1, Q2, m1EMB, m2EMB