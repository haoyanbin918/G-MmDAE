import time
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
#from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from G_MmDAE_star_mini import Net
from Config import Config
from utils import AverageMeter, compProb, np2FVaribleGpu

#
# def get_parser():
#     parser = argparse.ArgumentParser(description='model parameters')
#     parser.add_argument('--th', default=0, type=int)
#     parser.add_argument('--cw', default=1.0, type=float)

#     return parser
# #==============================
# parser = get_parser()
# opts = parser.parse_args()
#=============================
#
#
def input2selec(inds, all_idx):
    mismatch = 0.8
    indicate = []
    inds2 = np.zeros(len(inds), dtype=np.int)
    for one in range(len(inds)):
        match = np.random.uniform() > mismatch
        targ = match and 1 or -1
        indicate.append(targ)
        if targ == 1:
            inds2[one] = inds[one]
        else:
            rndindex = np.random.choice(all_idx)
            while rndindex == inds[one]:
                rndindex = np.random.choice(all_idx)
            inds2[one] = rndindex

    return inds2, indicate



#
def __iterate_minibatches__(x1, x2, batchsize, shuffle=False):
    assert len(x1) == len(x2)
    if shuffle:
        indices = np.arange(len(x1))
        np.random.shuffle(indices)

    for start_idx in range(0, len(x1) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        yield x1[excerpt], x2[excerpt], excerpt

#
def train(mod1, mod2, sigmaV1, sigmaV2, model, criterion, optimizer, epoch, config):
    # switch to train mode
    #model.train()
    all_ids = range(len(mod1))

    train_err = AverageMeter()

    train_err1 = AverageMeter()
    train_err2 = AverageMeter()
    train_err3 = AverageMeter()

    # train_batches = 0
    start_time = time.time()

    for batch in __iterate_minibatches__(mod1, mod2, config.batch_size, shuffle=True):
        
        input1_1, input1_2, ind1 = batch
        #print ind1
        #print '----------------------------'

        #time1 = time.time()
        ind2, target = input2selec(ind1, all_ids)

        #time1 = time.time()
        input2_1 = mod1[ind2]
        input2_2 = mod2[ind2]

        P1_1 = compProb(input1_1, sigmaV1[ind1])
        P1_2 = compProb(input1_2, sigmaV2[ind1])

        P1 = config.alpha*P1_1 + (1-config.alpha)*P1_2
        
        P2_1 = compProb(input2_1, sigmaV1[ind2])
        P2_2 = compProb(input2_2, sigmaV2[ind2])

        P2 = config.alpha*P2_2 + (1-config.alpha)*P2_1

        P1 = np2FVaribleGpu(P1)
        P2 = np2FVaribleGpu(P2)

        input1_1 = np2FVaribleGpu(input1_1)
        input2_2 = np2FVaribleGpu(input2_2)

        target = torch.FloatTensor(target)
        target = torch.autograd.Variable(target).cuda()

        # compute out
        lgQ1, lgQ2, output1, output2 = model(input1_1, input2_2)

        # compute loss

        loss1 = criterion[0](lgQ1, P1)
        loss2 = criterion[0](lgQ2, P2)

        loss3 = criterion[1](output1, output2, target)

        loss = loss1 + loss2 + config.beta*loss3

        #loss = criterion(lgQ1, P2)
        
        # combined loss
        #loss = loss_c1 + loss_c2 + config.bl*loss_kld

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure
        train_err.update(loss.item())

        train_err1.update(loss1.item())
        train_err2.update(loss2.item())
        train_err3.update(loss3.item())

        # train_batches += 1

    print("Epoch {} of {} took {:.3f}s".format(epoch + 1, config.epochs, time.time() - start_time)),
    print("training \tloss: {:.6f} loss1: {:.6f} loss2: {:.6f} loss3: {:.6f}".format(train_err.avg,\
     train_err1.avg, train_err2.avg, train_err3.avg))

#
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    conf = Config()

    print "Initializing model"
    model = Net(conf)

    pool = 'PoolA'  # target pool, Pool-A or Pool-B

    finetune = False

    if finetune:
        conf.lr = 0.0001
        model_dict = model.state_dict()
        print "Loading trained model from"
        modelpre_name = './models/...' # path of the pretrained model
        print modelpre_name

        modelpre = torch.load(modelpre_name, map_location=lambda storage, loc: storage)
        modelpre_dict = modelpre.state_dict()

        print "Updating model"
        pretrained_params = {k:v for k, v in modelpre_dict.items() if k in model_dict}
        model_dict.update(pretrained_params)

    model.cuda()

    th = 1  # number
    if finetune:
        modelName = 'G_MmDAE_star_mini_%s_beta%g_%d_%s.pth'%(pool, conf.beta, th, 'fine')
    else:
        modelName = 'G_MmDAE_star_mini_%s_beta%g_%d.pth'%(pool, conf.beta, th)
    #
    saveFoler = './models'
    if not os.path.exists(saveFoler):
        os.mkdir(saveFoler)
    savePath = os.path.join(saveFoler, modelName)

    print "Saving model to ", savePath
    #  HHHHHHHHHere  --------------____---
    # loss
    cosine_crit = nn.CosineEmbeddingLoss(conf.gamma).cuda()
    #mse_crit1 = nn.MSELoss().cuda()
    kld_crit = nn.KLDivLoss(reduction='sum').cuda()

    criterion = [kld_crit, cosine_crit]
    #criterion = nn.KLDivLoss(size_average=False).cuda()

    if conf.optmer == 'adm':
        optimizer = torch.optim.Adam(model.parameters(), lr=conf.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=conf.lr, momentum=conf.momentum, nesterov=True)
    #optimizer = torch.optim.SGD(model.parameters(), lr=conf.lr, momentum=conf.momentum, nesterov=False)
    #print 'There are %d parameter groups' % len(optimizer.param_groups)

    # print config settings
    txt = conf.print_item()

    cudnn.benchmark = True

    #---------train data loading-----------
    print "Loading training data from"
    trainRootPath = '/home/robin/LNK/target_files/All_Train_Files/Unsupervised_train_data/'

    m1DataPath = trainRootPath+'Train_KTE2_ImgRes50_CNN_20000.npy'
    print m1DataPath
    mod1 = np.load(m1DataPath)

    m2DataPath = trainRootPath+'Train_KTE2_WVgle_20000.npy'
    print m2DataPath
    mod2 = np.load(m2DataPath)
    
    assert len(mod1) == len(mod2)
    assert mod1.shape[1]==conf.m1Size
    assert mod2.shape[1]==conf.m2Size

    print "Loading sigma vector"
    # sigmaV1Path = './train_data/file.npy'
    # sigmaV1 = np.load(sigmaV1Path)
    # sigmaV2Path = './train_data/file.npy'
    # sigmaV2 = np.load(sigmaV2Path)
    # # or
    sigmaV1 = np.sqrt(np.ones(len(mod1))*0.5)
    sigmaV2 = np.sqrt(np.ones(len(mod1))*0.5)
    
    assert sigmaV1.shape[0] == sigmaV2.shape[0] == len(mod1)
    # train

    print "Training"
    model.train()
    for epoch in range(conf.epochs):
        train(mod1, mod2, sigmaV1, sigmaV2, model, criterion, optimizer, epoch, conf)

    print('Finished Training')

    # save model
    print "Saving model"
    torch.save(model, modelName)
    #torch.save(model.state_dict(), 'AEorig_pretra_params.pth')

    print time.asctime(time.localtime(time.time()))

if __name__ == '__main__':
    main()