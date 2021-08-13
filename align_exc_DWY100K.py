from __future__ import division
from __future__ import print_function
import os
import random
import time
import numpy as np
import torch
import autil.argclass as argclass
from align import Config_set, align_setmodel_noValid, attr_setmodel_noValid
from autil import printclass


def run(datasets, link_version, model_type='E', fileName='E(7.23)'):
    print('run begin------------------')
    # parameter configuration
    new_config = Config_set.config(argclass.load_args('args_15K.json'))
    new_config.datasetPath += datasets
    new_config.dataset_division += link_version
    new_config.output += datasets + link_version #
    new_config.output += fileName + '/'

    if '100K' in new_config.datasetPath:
        new_config.start_valid = 10
        new_config.eval_freq = 10
        new_config.eval_save_freq = 10
        new_config.sample_neg_freq = 20

        new_config.patience = 10
        new_config.learning_rate = 0.005
        if model_type == 'M':
            #new_config.patience = 5
            new_config.learning_rate = 0.001

    # E: align_setmodel_noValid，  M: attr_setmodel_noValid
    if model_type == 'E':
        thisModel = align_setmodel_noValid
    else:
        thisModel = attr_setmodel_noValid

    if not os.path.exists(new_config.output):
        print('output not exists' + new_config.output)
        os.makedirs(new_config.output)

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    new_config.cuda = False
    new_config.cuda = new_config.cuda and torch.cuda.is_available()  # cuda
    random.seed(new_config.seed)
    np.random.seed(new_config.seed)
    torch.manual_seed(new_config.seed)
    # Initialization, printing and logging
    print_Class = printclass.Myprint(new_config.output, 'train_log' +
                                     time.strftime('%Y-%m-%d %H_%M_%S', time.localtime(time.time())) + '.txt')
    myprint_fun = print_Class.print
    myprint_fun('!!cuda.is_available:' + str(new_config.cuda))
    myprint_fun("server== 182/align_nma2 ")
    myprint_fun("start==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    myprint_fun('model arguments:' + new_config.model_param)
    myprint_fun('output Path:' + new_config.output)

    myprint_fun("===train align_model: ")
    model = thisModel.modelClass2(new_config, myprint_fun)
    best_epochs, last_epochs = model.model_train()
    model.compute_test(best_epochs, 'best')
    myprint_fun("===test last_epochs: ")
    model.compute_test(last_epochs, 'last')
    myprint_fun("end==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))

if __name__ == '__main__':

    # DWY100K/dbp_wd/ , DWY100K/dbp_yg/
    datasets = 'DWY100K/dbp_wd/'
    run(datasets, link_version='1/', model_type='E', fileName='E(332_7.30)t')

    datasets = 'DWY100K/dbp_wd/'
    run(datasets, link_version='1/', model_type='M', fileName='M(332_7.30)t')


