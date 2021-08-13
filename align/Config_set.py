from __future__ import division
from __future__ import print_function
class config():
    def __init__(self, args):
        self.datasetPath = args.datasetPath
        self.dataset_division = args.dataset_division
        #self.entity_embed = 'wiki'
        self.output = args.output

        self.optim_type = args.optim_type  # 'Adagrad'
        self.train_epochs = args.train_epochs
        self.neg_k = args.neg_k  # number of negative samples for each positive one
        self.metric = args.metric  # L1/L2

        self.early_stop = args.early_stop
        self.start_valid = args.start_valid
        self.eval_freq = args.eval_freq
        self.eval_save_freq = args.eval_save_freq
        self.sample_neg_freq = args.sample_neg_freq
        self.patience = args.patience  # 20
        self.patience_val = args.patience_val

        # Super Parameter
        self.cuda = args.cuda
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.LeakyReLU_alpha = args.LeakyReLU_alpha
        self.dropout = args.dropout
        self.alpha1 = args.alpha1
        self.alpha2 = args.alpha2
        self.alpha3 = args.alpha3
        self.gamma = args.gamma  # 1.0,2.0,5.0
        self.l_bata = args.l_bata

        self.top_k = args.top_k  # [1, 3, 5, 10, 50, 100]
        self.csls_k = args.csls_k  # 5
        self.seed = args.seed

        # Result save file name
        self.model_param = '_'.join(
            ['epochs', str(self.train_epochs),
             'negk', str(self.neg_k),
             's_neg', str(self.sample_neg_freq),
             'lr', str(self.learning_rate),
             'wd', str(self.weight_decay),
             'la', str(self.LeakyReLU_alpha),
             'do', str(self.dropout),
             'al1', str(self.alpha1),
             'al2', str(self.alpha2),
             'al3', str(self.alpha3),
             'ga', str(self.gamma),
             'ba', str(self.l_bata) + '_HET_align'])
