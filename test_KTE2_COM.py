import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from numpy import linalg as LA
#from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
#from AEorigub import AEorigub
from MODEL_EVAL import MODEL_EVAL
#from Config import Config

#torch.manual_seed(1)
#
class Config:
    def __init__(self):
        self.m1Size = 2048
        self.m2Size = 300
        self.hdn1 = 4096
        self.hdn2 = 2048
        self.rep = 1024

        self.drp = 0.2        
        #self.balance = 0.5
        self.lr = 0.05
        self.momentum = 0.9
        self.epochs = 500
        self.batch_size = 256

#
def confCheck(name, conf):
    ind_hdn = name.index('hdn')
    ind_rep = name.index('rep')
    ind__   = name[ind_hdn:].index('_')+ind_hdn

    hdn = int(name[ind_hdn+3:ind_rep])
    rep = int(name[ind_rep+3:ind__])

    if conf.hdn!=hdn:
        ValueError("hdn in conf and the model doesnot match")
    if conf.rep!=rep:
        ValueError("rep in conf and the model doesnot match")

#
def nptovariable(npdata):
    out = npdata.tolist()
    out = torch.FloatTensor(out)
    out = torch.autograd.Variable(out)

    return out

def simNorm(sim):
    simN = LA.norm(sim, axis=1, keepdims=True)
    simnew = sim/np.tile(simN, (1, sim.shape[1]))

    return simnew

def computeEmb(model, x1, x2):

    assert len(x1) == len(x2)

    Interval = 20000

    N = x1.shape[0]/Interval
    if x1.shape[0]%Interval > 0:
        N = N+1

    for i in range(N):
        print "%d/%d"%(i+1, N)
        start = Interval*i
        if i<N-1:
            end = Interval*(i+1)
        else:
            end = x1.shape[0]
        #
        input1 = nptovariable(x1[start:end, :])
        input2 = nptovariable(x2[start:end, :])
        
        embed1, embed2 = model(input1, input2)

        embed1 = embed1.cpu().data.numpy()
        embed2 = embed2.cpu().data.numpy()
        #
        if i==0:
            embeds = np.hstack([embed1, embed2])
        else:
            temp = np.hstack([embed1, embed2])
            embeds = np.vstack([embeds, temp])

    return embeds
#
def linearComb(e1, e2, d, savnam, comname):

    Sa = cosine_similarity(e1[:, :d], e2[:, :d])
    Sb = cosine_similarity(e1[:, d:], e2[:, d:])

    SS = Sa
    finalName = savnam + '1000_%s.npy'%comname
    np.save(finalName, SS)

    SS = 0.75*Sa + 0.25*Sb
    finalName = savnam + '7525_%s.npy'%comname
    np.save(finalName, SS)

    SS = 0.5*Sa + 0.5*Sb
    finalName = savnam + '5050_%s.npy'%comname
    np.save(finalName, SS)

    SS = 0.25*Sa + 0.75*Sb
    finalName = savnam + '2575_%s.npy'%comname
    np.save(finalName, SS)

    SS = Sb
    finalName = savnam + '0100_%s.npy'%comname
    np.save(finalName, SS)
    



#
def main():
    #os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    conf = Config()

    #model = AEorig(conf)

    print "Loading trained model from"
    modelpre_name = './models_KTE2/FTNETTt_2_KTE2_R50CnWg2W_h40962048r1024_hy10bl00eh100.pth'
    print modelpre_name

    conf = Config()
    conf.hdn1 = 4096
    conf.hdn2 = 2048
    conf.rep = 1024

    dim = conf.rep
    fea1 = 'CNNn'
    fea2 = 'gle'

    
    comname = modelpre_name.split('/')[-1][:-4]
    #confCheck(model_name, conf)

    print "Initializing model"
    model = MODEL_EVAL(conf)
    model_dict = model.state_dict()
    
    modelpre = torch.load(modelpre_name, map_location=lambda storage, loc: storage)
    modelpre_dict = modelpre.state_dict()

    print "Updating model"
    pretrained_params = {k:v for k, v in modelpre_dict.items() if k in model_dict}
    model_dict.update(pretrained_params)
    
    model.eval()

    #cpu().data.numpy()

    #cudnn.benchmark = True

    FeaRootPath = '/home/robin/LNK/target_files/ALL_Fea_Matrix/'
    #===================target===========================
    #=================need changed=======================
    print "Loading test data: target data from" 
    mod1FeaPath = FeaRootPath + 'CNN_Res50/target_KTE2_COM_Res50_%s.npy'%fea1
    print mod1FeaPath
    mod1 = np.load(mod1FeaPath)
    #
    mod2FeaPath = FeaRootPath + 'WV_%s/target_KTE2_COM_WV%s.npy'%(fea2, fea2)
    print mod2FeaPath
    mod2 = np.load(mod2FeaPath)

    # compute embedings
    print "Computing embeddings"
    tg_embed = computeEmb(model, mod1, mod2)
    
    #
    #===================target===========================
    #=================need changed=======================
    print "Loading test data: target data from" 
    mod1FeaPath = FeaRootPath + 'CNN_Res50/target1to147_COM_Res50_%s.npy'%fea1
    print mod1FeaPath
    mod1 = np.load(mod1FeaPath)

    mod2FeaPath = FeaRootPath + 'WV_%s/target1to147_COM_WV%s.npy'%(fea2, fea2)
    print mod2FeaPath
    mod2 = np.load(mod2FeaPath)

    # compute embedings
    print "Computing embeddings"
    gt_embed = computeEmb(model, mod1, mod2)

    #=======================================================
    #====================anchor1to28====================
    print "Loading test data: anchor data from"
    anchor_m1FeaPath = FeaRootPath+'CNN_Res50/anchor1to28_Res50_%s.npy'%fea1
    print anchor_m1FeaPath
    mod1 = np.load(anchor_m1FeaPath)

    anchor_m2FeaPath = FeaRootPath+'WV_%s/anchor1to28_WV%s.npy'%(fea2, fea2)
    print anchor_m2FeaPath
    mod2 = np.load(anchor_m2FeaPath)

    # compute embedings
    print "Computing embeddings"
    ac1to28_embed = computeEmb(model, mod1, mod2)

    #===================anchor29to122===================
    print "Loading test data: anchor data from"
    anchor_m1FeaPath = FeaRootPath+'CNN_Res50/anchor29to122_Res50_%s.npy'%fea1
    print anchor_m1FeaPath
    mod1 = np.load(anchor_m1FeaPath)

    anchor_m2FeaPath = FeaRootPath+'WV_%s/anchor29to122_WV%s.npy'%(fea2, fea2)
    print anchor_m2FeaPath
    mod2 = np.load(anchor_m2FeaPath)

    # compute embedings
    print "Computing embeddings"
    ac29to122_embed = computeEmb(model, mod1, mod2)

    #===================anchor123to147===================
    print "Loading test data: anchor data from"
    anchor_m1FeaPath = FeaRootPath+'CNN_Res50/anchor123to147_Res50_%s.npy'%fea1
    print anchor_m1FeaPath
    mod1 = np.load(anchor_m1FeaPath)

    anchor_m2FeaPath = FeaRootPath+'WV_%s/anchor123to147_WV%s.npy'%(fea2, fea2)
    print anchor_m2FeaPath
    mod2 = np.load(anchor_m2FeaPath)

    # compute embedings
    print "Computing embeddings"
    ac123to147_embed = computeEmb(model, mod1, mod2)

    #=======================================================
    #=================compute cosine similarity=============
    #=======================================================
    print "Computing cosine similarity for KTE2 COM"
    S = cosine_similarity(ac1to28_embed, tg_embed)
    savename = './eval_KTE2_COM/retrieval_1to28/sim_%s.npy'%comname
    np.save(savename, S)

    savename = './eval_KTE2_COM/retrieval_1to28_LC/sim_'
    linearComb(ac1to28_embed, tg_embed, dim, savename, comname)

    print "Computing cosine similarity for KTE2 COM"
    S = cosine_similarity(ac29to122_embed, tg_embed)
    savename = './eval_KTE2_COM/retrieval_29to122/sim_%s.npy'%comname
    np.save(savename, S)

    savename = './eval_KTE2_COM/retrieval_29to122_LC/sim_'
    linearComb(ac29to122_embed, tg_embed, dim, savename, comname)

    print "Computing cosine similarity for KTE2 COM"
    S = cosine_similarity(ac123to147_embed, tg_embed)
    savename = './eval_KTE2_COM/retrieval_123to147/sim_%s.npy'%comname
    np.save(savename, S)

    savename = './eval_KTE2_COM/retrieval_123to147_LC/sim_'
    linearComb(ac123to147_embed, tg_embed, dim, savename, comname)

    #--------------------------------------------------------
    print "Computing cosine similarity for GT"
    S = cosine_similarity(ac1to28_embed, gt_embed)
    savename = './eval_KTE2_COM/retrieval_GT_1to28/sim_%s.npy'%comname
    np.save(savename, S)

    savename = './eval_KTE2_COM/retrieval_GT_1to28_LC/sim_'
    linearComb(ac1to28_embed, gt_embed, dim, savename, comname)

    print "Computing cosine similarity for GT"
    S = cosine_similarity(ac29to122_embed, gt_embed)
    savename = './eval_KTE2_COM/retrieval_GT_29to122/sim_KTE2_%s.npy'%comname
    np.save(savename, S)

    savename = './eval_KTE2_COM/retrieval_GT_29to122_LC/sim_'
    linearComb(ac29to122_embed, gt_embed, dim, savename, comname)

    print "Computing cosine similarity for GT"
    S = cosine_similarity(ac123to147_embed, gt_embed)
    savename = './eval_KTE2_COM/retrieval_GT_123to147/sim_KTE2_%s.npy'%comname
    np.save(savename, S)

    savename = './eval_KTE2_COM/retrieval_GT_123to147_LC/sim_'
    linearComb(ac123to147_embed, gt_embed, dim, savename, comname)


    #=================================================================
    #----------------norm----------------------
    #Sa1 = simNorm(Sa1)
    #Sa2 = simNorm(Sa2)

    #----------------------------------------------------

if __name__ == '__main__':
	main()
