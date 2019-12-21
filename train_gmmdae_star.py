import time
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
#from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from G_MmDAE_star import Net
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

        yield mod1[excerpt], mod2[excerpt], mod1[excerpt], mod2[excerpt], mod2[excerpt], mod1[excerpt], excerpt

#
def train(mod1, mod2, sigmaV1, sigmaV2, model, criterion, optimizer, epoch, config):
    # switch to train mode
    #model.train()

    train_err = AverageMeter()

    train_err1s = AverageMeter()
    train_err2s = AverageMeter()
    train_err1c = AverageMeter()
    train_err2c = AverageMeter()

    train_errkld1 = AverageMeter()
    train_errkld2 = AverageMeter()

    # train_batches = 0
    start_time = time.time()

    for batch in __iterate_minibatches__(mod1, mod2, config.batch_size, shuffle=True):
        
        input1, input2, target1s, target1c, target2s, target2c, inds = batch
        
        P1 = compProb(input1, sigmaV1[inds])
        P2 = compProb(input2, sigmaV2[inds])

        PP1 = config.alpha*P1 + (1-config.alpha)*P2
        PP2 = config.alpha*P2 + (1-config.alpha)*P1

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

        loss = config.lam*(config.alpha*(loss1s + loss2s) + (1-config.alpha)*(loss1c + loss2c)) + losskld1 + losskld2

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure
        train_err.update(loss.item())

        train_err1s.update(loss1s.item())
        train_err2s.update(loss2s.item())
        train_err1c.update(loss1c.item())
        train_err2c.update(loss2c.item())

        train_errkld1.update(losskld1.item())
        train_errkld2.update(losskld2.item())

    print("Epoch {} of {} took {:.3f}s".format(epoch + 1, config.epochs, time.time() - start_time)),
    print("training \tloss: {:.6f} loss1s: {:.6f} loss2s: {:.6f} loss1c: {:.6f} loss2c: {:.6f} losskld1: {:.6f} losskld2: {:.6f}".format(\
        train_err.avg, train_err1s.avg, train_err2s.avg, train_err1c.avg,\
         train_err2c.avg, train_errkld1.avg, train_errkld2.avg))

#
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    conf = Config()
    conf.optmer = 'adm'

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
        modelName = 'G_MmDAE_star_%s_lam%g_%d_%s.pth'%(pool, conf.lam, th, 'fine')
    else:
        modelName = 'G_MmDAE_star_%s_lam%g_%d.pth'%(pool, conf.lam, th)
    #
    saveFoler = './models'
    if not os.path.exists(saveFoler):
        os.mkdir(saveFoler)
    savePath = os.path.join(saveFoler, modelName)

    print "Saving model to ", savePath

    # loss
    mse_crit1 = nn.MSELoss().cuda()
    kld_crit2 = nn.KLDivLoss(reduction='sum').cuda()

    criterion = [mse_crit1, kld_crit2]

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