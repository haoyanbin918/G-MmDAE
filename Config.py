#
class Config:
    def __init__(self):
        self.m1Size = 2048  # m1, CNN size
        self.m2Size = 300   # m2, word2vec size
        self.hdn1 = 4096    # for m1 branch
        self.hdn2 = 2048    # for m2 branch
        self.rep = 1024     # representation size
        self.drp = 0.2      # droupout, must be 0.0<= <1.0
        
        self.alpha = 0.5    # parameter alpha used in G_MmDAE (Eq. 2) and G_MmDAE_star (Eq. 10)
        self.lam = 1.0      # parameter lambda used in G_MmDAE_star (Eq. 11)
        self.beta = 1.0     # parameter beta used in G_MmDAE_star_mini (Eq. 14)
        self.gamma = 0.1   # parameter margin used in G_MmDAE_star_mini (Eq. 13)
        
        self.optmer = 'adm'  # optimizer, could be 'sgd' or 'adm'
        self.lr = 0.0001    # learning rate
        self.momentum = 0.9 # momentum 
        self.epochs = 100   # epochs
        self.batch_size = 1000 # batch size
        self.input_norm = False   # input norm
        self.emb_norm = True  # embedding norm

    #
    def print_item(self):
        txt = 'Model settings \n'
        txt = txt + 'm1Size: {}\nm2Size: {}\nhdn1: {}\nhdn2: {}\nrep: {}\ndropout: {}\n'\
            .format(self.m1Size, self.m2Size, self.hdn1, self.hdn2, self.rep, self.drp)
        txt = txt + 'alpha: {}\nlambda: {}\nbeta: {}\ngamma: {}\n'.format(self.alpha, self.lam, self.beta, self.gamma)
        txt = txt + 'optimizer: {} \nlr: {}\nmomentum: {}\nepochs: {}\nbatch_size: {}\ninput_norm: {}\nemb_norm: {}\n'\
            .format(self.optmer, self.lr, self.momentum, self.epochs, self.batch_size, self.input_norm, self.emb_norm)
        print txt
        return txt
#-----------------------s
