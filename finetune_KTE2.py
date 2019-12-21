import time
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
#from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from NETTt import NETTt

#
def get_parser():
    parser = argparse.ArgumentParser(description='model parameters')
    parser.add_argument('--th', default=0, type=int)
    parser.add_argument('--hy', default=1, type=int)
    parser.add_argument('--bl', default=0.0, type=float)
    #parser.add_argument('--path', default='AE4lossTw3_1_VTTKTE2_R50CnWg_h40962048r1024_bl5eh150', type=str)

    return parser
#==============================
parser = get_parser()
opts = parser.parse_args()
#=============================
#
def conf2name(conf):
    name1 = '_h%d%dr%d'%(conf.hdn1, conf.hdn2, conf.rep)
    #name2 = '_lr%s'%(str(conf.lr)[2:])
    name3 = '_hy%02dbl%02deh%d'%(conf.hy*10, conf.bl*10, conf.epochs)
    name = name1+name3

    return name

def configPrint(conf):
    print "Mod1 dim: ", conf.m1Size
    print "Mod2 dim: ", conf.m2Size
    print "hdn1, hdn2: ", conf.hdn1, conf.hdn2
    print "rep: ", conf.rep
    print "lr: ", conf.lr
    print "epochs: ", conf.epochs
#
def compProb(vecs):

    dis = 1 - cosine_similarity(vecs)
    exps = np.exp(-dis**2)
    diag = np.diag(np.diag(exps))
    exps = exps - diag
    sum_e = np.sum(exps, axis=1)
    probs = exps/np.tile(sum_e, (exps.shape[0], 1)).T

    return np.maximum(probs, 1e-08)

#
def np2FVaribleGpu(nda):
    out = nda.tolist()
    out = torch.FloatTensor(out)
    out = torch.autograd.Variable(out).cuda()

    return out

#
def __iterate_minibatches__(mod1, mod2, batchsize, shuffle=False):
    assert len(mod1) == len(mod2)
    if shuffle:
        indices = np.arange(len(mod1))
        np.random.shuffle(indices)

    for start_idx in range(0, len(mod1) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        yield mod1[excerpt], mod2[excerpt], mod1[excerpt], mod2[excerpt], mod2[excerpt], mod1[excerpt]
#
#
class Config:
    def __init__(self):
        self.m1Size = 2048
        self.m2Size = 300
        self.hdn1 = 4096
        self.hdn2 = 2048
        self.rep = 1024
        self.drp = 0.2
        
        self.bl = 0.0
        self.hy = 1.0
        self.lr = 0.0001
        #self.momentum = 0.8
        self.epochs = 100
        self.batch_size = 1000
#
def train(mod1, mod2, model, criterion, optimizer, epoch, config):
	# switch to train mode
    #model.train()

    train_err = 0.0

    train_err1s = 0.0
    train_err2s = 0.0
    train_err1c = 0.0
    train_err2c = 0.0

    train_errkld1 = 0
    train_errkld2 = 0

    train_batches = 0
    start_time = time.time()

    for batch in __iterate_minibatches__(mod1, mod2, config.batch_size, shuffle=True):
    	
        input1, input2, target1s, target1c, target2s, target2c = batch

        P1 = compProb(input1)
        P2 = compProb(input2)

        PP1 = config.bl*P1 + (1-config.bl)*P2
        PP2 = config.bl*P2 + (1-config.bl)*P1
        #P = config.lam*P1 + (1-config.lam)*P2

        PP1 = np2FVaribleGpu(PP1)
        PP2 = np2FVaribleGpu(PP2)

        input1 = np2FVaribleGpu(input1)
        input2 = np2FVaribleGpu(input2)

        target1s = np2FVaribleGpu(target1s)
        target1c = np2FVaribleGpu(target1c)
        target2s = np2FVaribleGpu(target2s)
        target2c = np2FVaribleGpu(target2c)

        # compute out
    	lgQ1, lgQ2, m1OUTs, m1OUTc, m2OUTs, m2OUTc = model(input1, input2)

    	# compute loss

        loss1s = criterion[0](m1OUTs, target1s)
        loss1c = criterion[0](m1OUTc, target1c)
        loss2s = criterion[0](m2OUTs, target2s)
        loss2c = criterion[0](m2OUTc, target2c)

        losskld1 = criterion[1](lgQ1, PP1)
        losskld2 = criterion[1](lgQ2, PP2)

        loss = config.hy*(config.bl*(loss1s + loss2s) + (1-config.bl)*(loss1c + loss2c)) + losskld1 + losskld2

        #loss = criterion(lgQ1, P2)
    	
        # combined loss
        #loss = loss_c1 + loss_c2 + config.bl*loss_kld

    	# compute gradient and do Adam step
    	optimizer.zero_grad()
    	loss.backward()
    	optimizer.step()

    	# measure
    	train_err += loss.item()

    	train_err1s += loss1s.item()
    	train_err2s += loss2s.item()
        train_err1c += loss1c.item()
        train_err2c += loss2c.item()

        train_errkld1 += losskld1.item()
        train_errkld2 += losskld2.item()

    	train_batches += 1

    print("Epoch {} of {} took {:.3f}s".format(epoch + 1, config.epochs, time.time() - start_time)),
    print("training \tloss: {:.6f} loss1s: {:.6f} loss2s: {:.6f} loss1c: {:.6f} loss2c: {:.6f} losskld1: {:.6f} losskld2: {:.6f}".format(\
        train_err / train_batches, train_err1s / train_batches, train_err2s / train_batches, train_err1c / train_batches,\
         train_err2c / train_batches, train_errkld1 / train_batches, train_errkld2 / train_batches))

#
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    conf = Config()
    conf.hy = opts.hy
    conf.bl = opts.bl
    configPrint(conf)

    print "Initializing model"

    model = NETTt(conf)
    model_dict = model.state_dict()

    print "Loading trained model from"
    modelpre_name = './models_VTT/AE4lossTw3_%d_VTTKTE2_R50CnWg_h40962048r1024_bl5eh150.pth'%opts.th
    print modelpre_name

    comname = modelpre_name.split('/')[-1][:-4]
    #confCheck(model_name, conf)

    modelpre = torch.load(modelpre_name, map_location=lambda storage, loc: storage)
    modelpre_dict = modelpre.state_dict()

    print "Updating model"
    pretrained_params = {k:v for k, v in modelpre_dict.items() if k in model_dict}
    model_dict.update(pretrained_params)

    model.cuda()

    #-==-=-=-=-=============s

    feaName = 'R50CnWg2W'
    #======================

    partname = conf2name(conf)
    modelName = './models_KTE2/FTNETTt_%d_KTE2_%s%s.pth'%(opts.th, feaName, partname)
    print "Saving model to ", modelName

    
    # loss
    #cosine_crit = nn.CosineEmbeddingLoss(0.1).cuda()
    mse_crit1 = nn.MSELoss().cuda()
    kld_crit2 = nn.KLDivLoss(reduction='sum').cuda()

    criterion = [mse_crit1, kld_crit2]
    #criterion = nn.KLDivLoss(reduction='sum').cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=conf.lr, momentum=conf.momentum, nesterov=False)
    #print 'There are %d parameter groups' % len(optimizer.param_groups)

    cudnn.benchmark = True

    #---------train data loading-----------
    print "Loading training data from"
    trainRootPath = '/home/robin/LNK/target_files/All_Train_Files/Unsupervised_train_data/'

    m1DataPath = trainRootPath+'Train_KTE2_ImgRes50_CNNn_20000.npy'
    print m1DataPath
    mod1 = np.load(m1DataPath)

    m2DataPath = trainRootPath+'Train_KTE2_WVgle_20000.npy'
    print m2DataPath
    mod2 = np.load(m2DataPath)

    #m1SigmaPath = trainRootPath+'R50CNN4W_SIGMA.txt'
    #print m1SigmaPath
    #m1Sigma = np.loadtxt(m1SigmaPath, dtype=np.float32, delimiter=',')

    #m2SigmaPath = trainRootPath+'WVgle4W_SIGMA.txt'
    #print m2SigmaPath
    #m2Sigma = np.loadtxt(m2SigmaPath, dtype=np.float32, delimiter=',')
    
    assert len(mod1) == len(mod2)
    assert mod1.shape[1]==conf.m1Size
    assert mod2.shape[1]==conf.m2Size
    #assert m1Sigma.shape[0] == len(mod1)
    #assert m2Sigma.shape[0] == len(mod2)
    # train

    print "Training"
    model.train()
    for epoch in range(conf.epochs):
        train(mod1, mod2, model, criterion, optimizer, epoch, conf)

    print('Finished Training')

    # save model
    print "Saving model"
    torch.save(model, modelName)
    #torch.save(model.state_dict(), 'AEorig_pretra_params.pth')

    print time.asctime(time.localtime(time.time()))


if __name__ == '__main__':
    main()