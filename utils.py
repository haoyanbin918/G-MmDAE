import numpy as np
import torch
import shutil
# from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#
def save_checkpoint(state, is_best):
    # filename = '%s/ckpt_%d.pth.tar' % (state['save_folder'], state['epoch'])
    filename = '%s/ckpt.pth.tar' % (state['save_folder'])
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))
#
def compProb(vecs, sigma=1.0):
    if sigma is not 1.0:
        assert len(sigma.shape) == 1
        sigma = sigma**2 * 2
    dis = 1 - cosine_similarity(vecs, vecs, 'cosine')
    exps = np.exp(-dis**2/sigma)
    diag = np.diag(np.diag(exps))
    exps = exps - diag
    sum_e = np.sum(exps, axis=1)
    probs = exps/np.tile(sum_e, (exps.shape[0], 1)).T

    return np.maximum(probs, 1e-08)

#
#
def np2FVaribleGpu(nda):
    out = nda.tolist()
    out = torch.FloatTensor(out)
    out = torch.autograd.Variable(out).cuda()

    return out
