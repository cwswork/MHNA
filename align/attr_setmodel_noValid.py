import time
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from autil import alignment2
from align import attr_align_model2, align_setmodel2


class modelClass2():
    def __init__(self, config, myprint_fun):
        super(modelClass2, self).__init__()

        self.myprint = myprint_fun
        self.config = config
        self.best_mode_pkl_title = config.output + time.strftime('%Y-%m-%d_%H_%M_%S-', time.localtime(time.time()))

        #  Load data
        input_data = align_setmodel2.load_data(config, model_type='M')

        # Test set, validation set, training set
        self.train_links = input_data.train_links
        # self.test_links = input_data.test_links
        self.train_links_tensor = input_data.train_links_tensor
        self.test_links_tensor = input_data.test_links_tensor

        # Model and optimizer
        self.mymodel = attr_align_model2.HET_attr_align2(input_data, config)  # E+R+M+V
        if config.cuda:
            self.mymodel.cuda()
        # [TensorboardX]Summary_Writer
        self.board_writer = SummaryWriter(log_dir=self.best_mode_pkl_title + '-M/', comment='HET_attr_align')

        # optimizer
        self.parameters = filter(lambda p: p.requires_grad, self.mymodel.parameters())
        self.myprint('All parameter names in the model:' + str(len(self.mymodel.state_dict())))
        for i in self.mymodel.state_dict():
            self.myprint(i)
        if config.optim_type == 'Adagrad':
            self.optimizer = optim.Adam(self.parameters, lr=config.learning_rate,
                            weight_decay=config.weight_decay)  # weight_decay =5e-4
        else:
            self.optimizer = optim.SGD(self.parameters, lr=config.learning_rate,
                            weight_decay=config.weight_decay)

        # Weight initialization
        self.mymodel.init_weights()


    #### model train
    def model_train(self, epochs_beg=0):
        self.myprint("model training start==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
        t_begin = time.time()

        bad_counter = 0
        best_hits1 = 0  # best
        best_epochs = 0
        for epochs_i in range(epochs_beg, self.config.train_epochs):  # epochs=1001

            epochs_i_t = time.time()
            #  Forward pass
            self.mymodel.train()
            self.optimizer.zero_grad()
            es_embed, ec_embed = self.mymodel()

            #   generate negative samples
            self.regen_neg(epochs_i, es_embed, ec_embed)

            # loss、acc
            loss_train, e_out_embed = self.mymodel.get_loss(es_embed, ec_embed, self.train_neg_pairs_es,
                                                            self.train_neg_pairs_ec)  # loss:
            # Backward and optimize
            loss_train.backward()
            self.optimizer.step()

            if epochs_i % 5 != 0:
                self.myprint('Epoch-{:04d}: train_loss-{:.4f}, cost time-{:.4f}s'.format(
                    epochs_i, loss_train.data.item(), time.time() - epochs_i_t))
            else:
                # accuracy: hits, mr, mrr
                result_train = alignment2.my_accuracy(e_out_embed, self.train_links_tensor, top_k=self.config.top_k,
                                                    metric=self.config.metric)

                # [TensorboardX]
                self.board_writer.add_scalar('train_loss', loss_train.data.item(), epochs_i)
                self.board_writer.add_scalar('train_hits1', result_train[0][0], epochs_i)

                self.myprint('Epoch-{:04d}: train_loss-{:.4f}, cost time-{:.4f}s'.format(
                    epochs_i, loss_train.data.item(), time.time() - epochs_i_t))
                self.print_result('Train', result_train)


            # ********************no early stop********************************************
            if epochs_i >= self.config.start_valid and epochs_i % self.config.eval_freq == 0:
                # From left
                result_test = alignment2.my_accuracy(e_out_embed, self.test_links_tensor, top_k=self.config.top_k,
                                                     metric=self.config.metric)
                self.print_result('Temp Test From Left', result_test)

                # save best model in valid
                if result_test[0][0] >= best_hits1:
                    best_hits1 = result_test[0][0]
                    best_epochs = epochs_i
                    bad_counter = 0
                    self.myprint('Epoch-{:04d}, better result, best_hits1:{:.4f}..'.format(epochs_i, best_hits1))
                    self.save_model(epochs_i, 'best-epochs')
                else:
                    # no best, but save model every 20 epochs
                    if epochs_i % self.config.eval_save_freq == 0:
                        self.save_model(epochs_i, 'eval-epochs')
                    # bad model, stop train
                    bad_counter += 1
                    self.myprint('bad_counter++:' + str(bad_counter))
                    if bad_counter == self.config.patience:  # patience=15
                        self.myprint('Epoch-{:04d}, bad_counter.'.format(epochs_i))
                        break

        self.save_model(epochs_i, 'last-epochs')  # save last epochs
        self.myprint("Optimization Finished!")
        self.myprint('Best epoch-{:04d}:'.format(best_epochs))
        self.myprint('Last epoch-{:04d}:'.format(epochs_i))
        self.myprint("Total time elapsed: {:.4f}s".format(time.time() - t_begin))

        return best_epochs, epochs_i

    # get negative samples
    def regen_neg(self, epochs_i, es_embed, ec_embed):
        if epochs_i % self.config.sample_neg_freq == 0:  # sample negative pairs every 20 epochs
            with torch.no_grad():
                # Negative sample sampling-training pair (positive sample and negative sample)
                self.train_neg_pairs_es = alignment2.gen_neg(es_embed, self.train_links, self.config.metric, self.config.neg_k)
                self.train_neg_pairs_ec = alignment2.gen_neg(ec_embed, self.train_links, self.config.metric, self.config.neg_k)


    def compute_test(self, epochs_i, name_epochs):
        ''' run best model '''
        model_savefile = '{}-epochs{}-{}.pkl'.format(self.best_mode_pkl_title + name_epochs, epochs_i, self.config.model_param)
        self.myprint('\nLoading {} - {}th epoch'.format(name_epochs, epochs_i))
        self.re_test(model_savefile)


    def re_test(self, model_savefile):
        ''' restart run best model '''
        # Restore best model
        self.myprint('Loading file: ' + model_savefile)
        self.mymodel.load_state_dict(torch.load(model_savefile))  # load_state_dict()
        self.mymodel.eval()  # self.train(False)
        es_embed, ec_embed = self.mymodel()
        e_out_embed_test = torch.cat((es_embed, ec_embed), dim=1)

        # From left
        result_test = alignment2.my_accuracy(e_out_embed_test, self.test_links_tensor, top_k=self.config.top_k,
                                                    metric=self.config.metric)
        result_str1 = self.print_result('Test From Left', result_test)
        # From right
        result_test = alignment2.my_accuracy(e_out_embed_test, self.test_links_tensor, top_k=self.config.top_k,
                                                    metric=self.config.metric, fromLeft=False)
        result_str2 = self.print_result('Test From right', result_test)

        model_result_file = '{}_Result-{}.txt'.format(self.best_mode_pkl_title, self.config.model_param)
        with open(model_result_file, "a") as ff:
            ff.write(result_str1)
            ff.write('\n')
            ff.write(result_str2)
            ff.write('\n')

        return es_embed, ec_embed


    def re_train(self, epochs_beg, model_savefile):
        ''' restart run best model '''
        # Restore best model
        es_embed, ec_embed = self.re_test(model_savefile)

        self.myprint('==== begin retrain ====')
        #   generate negative samples
        self.regen_neg(0, es_embed, ec_embed)
        self.model_train(epochs_beg)


    def save_model(self, better_epochs_i, epochs_name):
        # save model to file
        model_savefile = self.best_mode_pkl_title + epochs_name + \
                         str(better_epochs_i) + '-' + self.config.model_param + '.pkl'
        torch.save(self.mymodel.state_dict(), model_savefile)


    def print_result(self, pt_type, run_re):
        ''' Output result '''
        hits, mr, mrr = run_re[0], run_re[1], run_re[2]
        result = pt_type
        result += "==results: hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}".format(self.config.top_k, hits, mr, mrr)
        self.myprint(result)
        return result
