from __future__ import division
from __future__ import print_function
import os
import random
import time
import numpy as np
import torch
import autil.argclass as argclass
from align import Config_set, align_setmodel2, attr_setmodel2
from autil import printclass


def run(datasets, link_version, model_type='E', fileName='E(7.23)'):
    print('run begin------------------')
    # parameter configuration
    new_config = Config_set.config(argclass.load_args('args_15K.json'))
    new_config.datasetPath += datasets
    new_config.dataset_division += link_version  # 1-5, tt5, tt10, tt15, tt25, tt30
    new_config.output += datasets + link_version
    new_config.output += fileName + '/'

    # E: align_setmodel2，  M: attr_setmodel2
    if model_type == 'E':
        thisModel = align_setmodel2
    else:
        thisModel = attr_setmodel2

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
    model.compute_test(best_epochs, 'best')  #  Testing
    myprint_fun("===test last_epochs: ")
    model.compute_test(last_epochs, 'last')  #

    myprint_fun("end==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))


if __name__ == '__main__':
    # EN_DE_15K_V1, EN_FR_15K_V1, fr_en(dbp), ja_en(dbp)#
    datasets = 'EN_DE_15K_V1/'
    run(datasets, link_version='1/', model_type='E', fileName='E(7.23)tt')
    # run(datasets, link_version='2/', model_type='E', fileName='E(7.23)')
    # run(datasets, link_version='3/', model_type='E', fileName='E(7.23)')
    # run(datasets, link_version='4/', model_type='E', fileName='E(7.23)')
    # run(datasets, link_version='5/', model_type='E', fileName='E(7.23)')
