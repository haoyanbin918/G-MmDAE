import time
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
#from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from G_MmDAE import Net0
from Config import Config
from utils import AverageMeter, np2FVaribleGpu

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
def train(mod1, mod2, model, criterion, optimizer, epoch, config):
    # switch to train mode
    #model.train()

    train_err = AverageMeter()

    lossS1_err = AverageMeter()
    lossC1_err = AverageMeter()
    lossS2_err = AverageMeter()
    lossC2_err = AverageMeter()

    # train_batches = 0
    start_time = time.time()

    for batch in __iterate_minibatches__(mod1, mod2, config.batch_size, shuffle=True):
        
        input1, input2, targetS1, targetC1, targetS2, targetC2 = batch

        input1 = np2FVaribleGpu(input1)
        input2 = np2FVaribleGpu(input2)

        targetS1 = np2FVaribleGpu(targetS1)
        targetC1 = np2FVaribleGpu(targetC1)
        targetS2 = np2FVaribleGpu(targetS2)
        targetC2 = np2FVaribleGpu(targetC2)
        
        # compute out
        outputS1, outputC1, outputS2, outputC2 = model(input1, input2)

        # compute loss
        lossS1 = criterion(outputS1, targetS1)
        lossC1 = criterion(outputC1, targetC1)

        lossS2 = criterion(outputS2, targetS2)
        lossC2 = criterion(outputC2, targetC2)
        # combined loss
        loss = config.alpha*lossS1 + (1-config.alpha)*lossC1 + config.alpha*lossS2 + (1-config.alpha)*lossC2

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure
        train_err.update(loss.item())

        lossS1_err.update(lossS1.item())
        lossC1_err.update(lossC1.item())
        lossS2_err.update(lossS2.item())
        lossC2_err.update(lossC2.item())

        # train_batches += 1

    print("Epoch {} of {} took {:.3f}s".format(epoch + 1, config.epochs, time.time() - start_time)),
    print("training loss:{:.6f} lossS1:{:.6f} lossC1:{:.6f} lossS2:{:.6f} lossC2:{:.6f}".format(train_err.avg,\
        lossS1_err.avg, lossC1_err.avg, lossS2_err.avg, lossC2_err.avg))

#
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    conf = Config()
    conf.optmer = 'sgd'
    conf.lr = 0.01

    print "Initializing model"
    model = Net0(conf)

    pool = 'PoolA'  # target pool, Pool-A or Pool-B

    finetune = False

    if finetune:
        conf.lr = 0.001
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
        modelName = 'G_MmDAE_%s_beta%g_%d_%s.pth'%(pool, conf.beta, th, 'fine')
    else:
        modelName = 'G_MmDAE_%s_beta%g_%d.pth'%(pool, conf.beta, th)
    #
    saveFoler = './models'
    if not os.path.exists(saveFoler):
        os.mkdir(saveFoler)
    savePath = os.path.join(saveFoler, modelName)

    print "Saving model to ", savePath
    #  HHHHHHHHHere  --------------____---
    # loss
    criterion = nn.MSELoss().cuda()

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