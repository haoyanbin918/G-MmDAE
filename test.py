import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
# from numpy import linalg as LA
#from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
#from AEorigub import AEorigub
from MODEL_EVAL import MODEL_EVAL
from Config import Config

#torch.manual_seed(1)
#


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
def main():
    #os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    pool = 'PoolA'  # target pool, Pool-A or Pool-B

    conf = Config()

    print "Loading trained model from"
    modelpre_name = './models/file.pth'
    print modelpre_name
    
    print "Initializing model"
    model = MODEL_EVAL(conf)
    model_dict = model.state_dict()
    
    modelpre = torch.load(modelpre_name, map_location=lambda storage, loc: storage)
    modelpre_dict = modelpre.state_dict()

    print "Updating model"
    pretrained_params = {k:v for k, v in modelpre_dict.items() if k in model_dict}
    model_dict.update(pretrained_params)
    
    model.eval()

    FeaRootPath = './features'
    #===================target===========================
    #=================need changed=======================
    print "Loading test data: target data from" 
    mod1FeaPath = FeaRootPath + 'm1_file.npy'
    print mod1FeaPath
    mod1 = np.load(mod1FeaPath)
    #
    mod2FeaPath = FeaRootPath + 'm2_file.npy'
    print mod2FeaPath
    mod2 = np.load(mod2FeaPath)

    # compute embedings
    print "Computing embeddings"
    tg_embed = computeEmb(model, mod1, mod2)
    
    #
    #=======================================================
    #====================anchor1to28====================
    print "Loading test data: anchor data from"
    anchor_m1FeaPath = FeaRootPath+'m1_file.npy'
    print anchor_m1FeaPath
    mod1 = np.load(anchor_m1FeaPath)

    anchor_m2FeaPath = FeaRootPath+'m2_file.npy'
    print anchor_m2FeaPath
    mod2 = np.load(anchor_m2FeaPath)

    # compute embedings
    print "Computing embeddings"
    ac1to28_embed = computeEmb(model, mod1, mod2)

    #===================anchor29to122===================
    print "Loading test data: anchor data from"
    anchor_m1FeaPath = FeaRootPath+'m1_file.npy'
    print anchor_m1FeaPath
    mod1 = np.load(anchor_m1FeaPath)

    anchor_m2FeaPath = FeaRootPath+'m2_file.npy'
    print anchor_m2FeaPath
    mod2 = np.load(anchor_m2FeaPath)

    # compute embedings
    print "Computing embeddings"
    ac29to122_embed = computeEmb(model, mod1, mod2)

    #===================anchor123to147===================
    print "Loading test data: anchor data from"
    anchor_m1FeaPath = FeaRootPath+'m1_file.npy'
    print anchor_m1FeaPath
    mod1 = np.load(anchor_m1FeaPath)

    anchor_m2FeaPath = FeaRootPath+'m2_file.npy'
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
    savename = './evaluate/retrieval_1to28/sim.npy'
    np.save(savename, S)

    print "Computing cosine similarity for KTE2 COM"
    S = cosine_similarity(ac29to122_embed, tg_embed)
    savename = './evaluate/retrieval_29to122/sim.npy'
    np.save(savename, S)

    print "Computing cosine similarity for KTE2 COM"
    S = cosine_similarity(ac123to147_embed, tg_embed)
    savename = './evaluate/retrieval_123to147/sim.npy'
    np.save(savename, S)


    # linearComb(ac123to147_embed, gt_embed, dim, savename, comname)


    #=================================================================
    #----------------norm----------------------
    #Sa1 = simNorm(Sa1)
    #Sa2 = simNorm(Sa2)

    #----------------------------------------------------

if __name__ == '__main__':
	main()
