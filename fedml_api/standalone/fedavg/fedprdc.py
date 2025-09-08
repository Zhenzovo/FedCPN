import copy
import logging

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune
import wandb
from fedml_api.standalone.fedavg.client import Client
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from thop import profile

import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.dates as mdates

from scipy.optimize import minimize, Bounds, NonlinearConstraint

import random
from collections import OrderedDict

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 或其他已知的无衬线字体
plt.rcParams['pdf.fonttype'] = 42  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
torch.set_printoptions(precision=6, sci_mode=False)


class FedAPTA(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

        self.prune_strategy = args.pr_strategy
        self._ln_prune = self._ln_prune_r18
        self.prune_prob = {
            '0': [0, 0, 0, 0],
            'AD': [0, 0, 0, 0],
            '0.1': [0.1, 0.1, 0.1, 0.1],
            '0.2': [0.2, 0.2, 0.2, 0.2],
            '0.3': [0.3, 0.3, 0.3, 0.3],
            '0.4': [0.4, 0.4, 0.4, 0.4],
            '0.5': [0.5, 0.5, 0.5, 0.5],
            '0.6': [0.6, 0.6, 0.6, 0.6],
            '0.7': [0.7, 0.7, 0.7, 0.7],
            '0.8': [0.8, 0.8, 0.8, 0.8],
            '0.9': [0.9, 0.9, 0.9, 0.9],
        }
        # self.ad_prob = ['0', '0.1', '0.3', '0.5', '0.7']
        self.ad_prob = ['0', '0.2', '0.4', '0.6', '0.8']
        self.ad_prob_same = ['0', '0.2', '0.4', '0.6', '0.8']

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
            self.D.append(c.get_sample_number())
        logging.info("############setup_clients (END)#############")

    def recover(self, model_state, init_state):
        # xx
        return

    def get_params(self, model_list, n_par=None):
        if n_par == None:
            exp_mdl = model_list[0]
            n_par = 0
            for name, param in exp_mdl.named_parameters():
                n_par += len(param.data.reshape(-1))

        param_mat = np.zeros((len(model_list), n_par)).astype('float32')
        for i, mdl in enumerate(model_list):
            idx = 0
            for name, param in mdl.named_parameters():
                temp = param.data.cpu().numpy().reshape(-1)
                param_mat[i, idx:idx + len(temp)] = temp
                idx += len(temp)
        return np.copy(param_mat)

    def set_model_from_params(self, mdl, params):
        dict_param = dict(mdl.named_parameters())
        idx = 0
        for name, param in mdl.named_parameters():
            weights = param.data
            length = len(weights.reshape(-1))
            dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).cuda())
            idx += length

        # mdl.load_state_dict(dict_param, strict=False)
        mdl.load_state_dict(dict_param)
        return mdl

    def train(self):
        print("=======================Clients=======================")
        pruning_rates = [0] * 10
        if self.args.pr_strategy == 'loop':
            pruning_rates = [0, 0.2, 0.4, 0.6, 0.8, 0, 0.2, 0.4, 0.6, 0.8]
        elif self.args.pr_strategy in ['0.2', '0.4', '0.6', '0.8']:
            pruning_rates = [float(self.args.pr_strategy)] * 10
        if self.args.test:
            pruning_rates = [0, 0.2, 0.4, 0.6, 0.8, 0, 0.2, 0.4, 0.6, 0.8]
        print('客户端剪枝率（PRUNING RATE OF CLIENTS）：', pruning_rates)
        logging.info("=================%s training on %s================="
                     % (self.args.pr_type, self.args.dataset))
        w_global = copy.deepcopy(self.model_trainer.get_model_params())
        init_model = copy.deepcopy(self.model_trainer.get_model())
        # new
        n_list = []
        for idx, client in enumerate(self.client_list):
            n_list.append(client.get_sample_number())

        n_data_per_client = sum(n_list) / len(n_list)
        n_iter_per_epoch = np.ceil(n_data_per_client / self.args.batch_size)
        n_minibatch = (self.args.epochs * n_iter_per_epoch).astype(np.int64)

        weight_list = np.asarray(n_list)
        weight_list = weight_list / np.sum(weight_list) * self.args.client_num_in_total

        n_par = len(self.get_params([init_model])[0])
        parameter_drifts = np.zeros((self.args.client_num_in_total, n_par)).astype('float32')
        init_par_list = self.get_params([init_model], n_par)[0]
        clnt_params_list = np.ones(self.args.client_num_in_total).astype('float32').reshape(-1,
                                                                                            1) * init_par_list.reshape(
            1, -1)  # n_clnt X n_par
        state_gadient_diffs = np.zeros((self.args.client_num_in_total + 1, n_par)).astype(
            'float32')  # including cloud state

        cur_glb_model = copy.deepcopy(init_model)
        cur_glb_model.load_state_dict(copy.deepcopy(w_global))
        glb_mdl_param = self.get_params([cur_glb_model], n_par)[0]

        # freeze layers.
        # --------ResNet--------
        if self.args.model in ["pre_r18", "pre_resnet18", "r18", "resnet18"]:
            # freeze layers
            init_model = self._freeze_layers_r18(self.args.freeze_layer, init_model)
        # --------ShuffleNet--------
        elif self.args.model in ["pre_ShuffleNet", "pre_shufflenet", "pre_sfnet", "ShuffleNet", "shufflenet", "sfnet"]:
            # freeze layers
            init_model = self._freeze_layers_sfnet(self.args.freeze_layer, init_model)

        best_acc = 0.0
        prefix = '_'.join(['pr' + str(self.args.pr) + 'dc' + str(self.args.dc), self.args.model,
                           self.args.partition_method, self.args.dataset, '_'])
        suffix = self.args.suffix
        # while os.path.exists('./checkpoints/%s%d.pth.tar' % (prefix, suffix)) or os.path.exists(
        #         f'.//RESULTS//{prefix}{suffix}.txt'):
        #     suffix += 1
        filename = prefix + str(suffix)
        # filename = './checkpoints/%s%d.pth.tar' % (prefix, suffix)

        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))
            # w_locals = []

            if self.args.update_client or round_idx == 0:
                client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                       self.args.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))

            # Adjusting lr.
            if self.args.lr_schedule and round_idx in self.args.milestones:
                self.args.lr = self.args.lr * self.args.lr_gama

            global_mdl = torch.tensor(glb_mdl_param, dtype=torch.float32).cuda()  # Theta
            delta_g_sum = np.zeros(n_par)

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                # ---------------------------------pruning and training------------------------------------------
                local_model = copy.deepcopy(init_model)
                local_model.load_state_dict(copy.deepcopy(w_global))

                if self.args.pr:
                    # if self.prune_strategy == "AD":
                    #     pr_strategy = self.ad_prob[idx % len(self.ad_prob)]  # get pruning ratio of specific client
                    #     pr_prob = self.prune_prob[pr_strategy]  # get pruning ratios for layers
                    #     self.prune_prob['AD'] = self.prune_prob[pr_strategy]  # copy pruning ratios for layers
                    # else:
                    #     pr_prob = self.prune_prob[self.prune_strategy]  # get pruning ratios for layers
                    # pr_model = self._ln_prune(local_model, pr_prob, round_idx, remove=0)

                    if self.args.prm == 'apta':
                        # todo
                        layers_pr_dict = self.new_calculate_layerwise_pruning_rates(local_model, pruning_rates[idx])
                        if self.args.oldpr:
                            for k in layers_pr_dict.keys():
                                layers_pr_dict[k] = pruning_rates[idx]

                        self.validate_pruning_amount(local_model, layers_pr_dict, pruning_rates[idx])
                        final_pr = list(layers_pr_dict.values())
                        pr_model = self.prune_model_layerwise(local_model, final_pr)

                        for k, v in layers_pr_dict.items():
                            print(f"Layer:{k} ---- pruning rate: {v}")
                    elif self.args.prm == 'fedlps':
                        pr_prob = self.prune_prob[str(pruning_rates[idx])]
                        pr_model = self._ln_prune(local_model, pr_prob)
                    elif self.args.prm == 'feddrop' and self.args.model in ["pre_r18", "pre_resnet18"]:
                        pr_model = self.feddrop_pruning(local_model, pruning_rates[idx])
                    # elif self.args.prm == 'fedmp':
                    #     pr_model = self._ln_prune(local_model, pruning_rates[idx])
                    elif self.args.prm == 'fedrolex' and self.args.model in ["pre_r18", "pre_resnet18"]:
                        pr_model = self.fedrolex_pruning(local_model, pruning_rates[idx], round_idx)
                    else:
                        raise ValueError("Invalid prm")

                    print("L1 pruning on client %s finished" % str(client_idx))
                else:
                    pr_model = local_model

                # Local training.
                if self.args.dataset in ["cifar10", "cf10", "cifar100", "cf100"]:
                    alpha = 0.01
                else:  # "mnist", "mn", "fashionmnist", "FashionMNIST", "fmn", "fmnist", "svhn", "SVHN", "sn"
                    alpha = 0.1
                local_update_last = state_gadient_diffs[idx]  # delta theta_i
                global_update_last = state_gadient_diffs[-1] / weight_list[idx]  # delta theta
                alpha = alpha / weight_list[idx]
                hist_i = torch.tensor(parameter_drifts[idx], dtype=torch.float32).cuda()  # h_i

                self.model_trainer.set_model(pr_model)
                w = client.train(copy.deepcopy(pr_model.state_dict()), round_idx, alpha, local_update_last,
                                 global_update_last, global_mdl, hist_i)

                # local recovery
                recovered_w = self.recover(w, w_global)
                recovered_mdl = copy.deepcopy(init_model)
                recovered_mdl.load_state_dict(recovered_w)


                # update
                curr_model_par = self.get_params([recovered_mdl], n_par)[0]
                delta_param_curr = curr_model_par - glb_mdl_param
                parameter_drifts[idx] += delta_param_curr
                beta = 1 / n_minibatch / self.args.lr

                state_g = local_update_last - global_update_last + beta * (-delta_param_curr)
                delta_g_cur = (state_g - state_gadient_diffs[idx]) * weight_list[idx]  # g_i更新量
                delta_g_sum += delta_g_cur
                state_gadient_diffs[idx] = state_g
                clnt_params_list[idx] = curr_model_par

            # print(f'验证{client_indexes}')
            # avg_mdl_param_sel = np.mean(clnt_params_list[client_indexes], axis=0)
            avg_mdl_param_sel = np.average(clnt_params_list[client_indexes], axis=0,
                                           weights=weight_list[client_indexes])

            delta_g_cur = 1 / self.args.client_num_in_total * delta_g_sum
            state_gadient_diffs[-1] += delta_g_cur

            # ag
            glb_mdl_param = avg_mdl_param_sel + np.mean(parameter_drifts, axis=0) if self.args.dc else avg_mdl_param_sel
            cur_glb_model = self.set_model_from_params(copy.deepcopy(init_model).cuda(), glb_mdl_param)
            w_global = copy.deepcopy(cur_glb_model.state_dict())

            # ------------------------------------ Test results ----------------------------------------
            if round_idx == self.args.comm_round - 1:
                best_acc = self._local_test_on_all_clients(round_idx, w_global, best_acc, init_model, filename)
                # best_acc = self._local_test_on_all_clients(round_idx, init_model, is_pruned, best_acc)

            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                best_acc = self._local_test_on_all_clients(round_idx, w_global, best_acc, init_model, filename)
                # best_acc = self._local_test_on_all_clients(round_idx, init_model, is_pruned, best_acc)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _freeze_layers_r18(self, layer_num, model):
        if layer_num == 0:
            return model
        if layer_num == 1:
            for name, child in model.named_children():
                if "layer2" not in name:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    break
        if layer_num == 2:
            for name, child in model.named_children():
                if "layer3" not in name:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    break
        if layer_num == 3:
            for name, child in model.named_children():
                if "layer4" not in name:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    break
        if layer_num == 4:
            for name, child in model.named_children():
                if "avgpool" not in name:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    break
        return model

    def _freeze_layers_sfnet(self, layer_num, model):
        if layer_num == 0:
            return model
        # frozen layer_num: {0, 1, 2, 3, 4}
        # corresponding stages: {0, 2, 3, 4, conv5.0} (not break InvertedResidual block)
        frozen_modules = [0, 55, 149, 199, 203]  # index: layer_num; value: frozen modules until this layer_num.
        for idx, m in enumerate(model.named_modules()):
            if idx <= 1:  # 0 -- squeezenet, 1 -- conv1(this is a nn.Sequential, not a conv layer)
                continue
            if idx <= frozen_modules[layer_num]:
                if isinstance(m[1], (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                    m[1].requires_grad_(False)
        return model

    def solve_pi(self, N_list, goal_list, p):
        pass
        # return pi_original_order.tolist()

    def new_calculate_layerwise_pruning_rates(self, model, PRi, lambda_param=1e-4):
        pass
        # return layers_pr_dict  # 每个元素是对应非冻结卷积层的剪枝率

    def prune_model_layerwise(self, model, final_pr):
        pass
        # return model

    def _local_test_on_all_clients(self, round_idx, w_global, best_acc, init_model, filename):

        print("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        client = self.client_list[0]

        self.model_trainer.set_model(copy.deepcopy(init_model))
        self.model_trainer.set_model_params(copy.deepcopy(w_global))
        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])


            # train data
            train_local_metrics = client.local_test(False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['num_samples']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['num_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['losses']))
            # train_metrics = copy.deepcopy(train_local_metrics)

            if client_idx == 0:
                # test data
                test_local_metrics = client.local_test(True)

                test_metrics = copy.deepcopy(test_local_metrics)

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = test_metrics['num_correct'] / test_metrics['num_samples']
        test_loss = test_metrics['losses'] / test_metrics['num_samples']
        if test_acc > best_acc:
            best_acc = test_acc

            # save the ckpt.
            if self.args.store_ckpt:
                # prefix = 'prdc_' + self.args.dataset + '_'
                # suffix = 0
                # while os.path.exists('./checkpoints/%s%d.pth.tar' % (prefix, suffix)):
                #     suffix += 1
                # filename = './checkpoints/%s%d.pth.tar' % (prefix, suffix)
                # filename = './checkpoints/%s.pth.tar' % self.args.prefix
                path = './checkpoints/%s.pth.tar' % filename
                # if round_idx == 0:
                #     path = './checkpoints/0_%s.pth.tar' % filename
                model = self.model_trainer.get_model()
                model.load_state_dict(w_global)
                state = {
                    'round': round_idx + 1,
                    'best_acc': best_acc,
                    'model': model,
                }
                torch.save(state, path)
                logging.info("Checkpoint is saved at %s" % path)

            print(f'!!!!!!!!!!!!!!!!!!!!THE BEST RESULT UPDATE(STRAT)!!!!!!!!!!!!!!!!!!!!')
            stats = {'round': round_idx,
                     'dataset': self.args.dataset,
                     'LOSS': test_loss,
                     'ACC': test_acc,
                     'precision_micro': test_metrics['precision_micro'],
                     'precision_macro': test_metrics['precision_macro'],
                     'precision_weighted': test_metrics['precision_weighted'],
                     'recall_micro': test_metrics['recall_micro'],
                     'recall_macro': test_metrics['recall_macro'],
                     'recall_weighted': test_metrics['recall_weighted'],
                     'f1_micro': test_metrics['f1_micro'],
                     'f1_macro': test_metrics['f1_macro'],
                     'f1_weighted': test_metrics['f1_weighted'],
                     'auc_micro': test_metrics['auc_micro'],
                     'auc_macro': test_metrics['auc_macro'],
                     'auc_weighted': test_metrics['auc_weighted'],
                     'fnr_micro': test_metrics['fnr_micro'],
                     'fnr_macro': test_metrics['fnr_macro'],
                     'fnr_weighted': test_metrics['fnr_weighted'],
                     'fpr_micro': test_metrics['fpr_micro'],
                     'fpr_macro': test_metrics['fpr_macro'],
                     'fpr_weighted': test_metrics['fpr_weighted'],
                     'fnp_micro': test_metrics['fnp_micro'],
                     'fnp_macro': test_metrics['fnp_macro'],
                     'fnp_weighted': test_metrics['fnp_weighted'],
                     'ftp_micro': test_metrics['ftp_micro'],
                     'ftp_macro': test_metrics['ftp_macro'],
                     'ftp_weighted': test_metrics['ftp_weighted'],
                     'num_classes': test_metrics['num_classes'],
                     'fpr': test_metrics['fpr'],
                     'tpr': test_metrics['tpr'],
                     'roc_auc': test_metrics['roc_auc'],
                     'precision': test_metrics['precision'],
                     'recall': test_metrics['recall'],
                     'average_precision': test_metrics['average_precision'],
                     'confusion_matrix': test_metrics['confusion_matrix'],
                     # '='*10: '='*10,
                     # 'average_precision_micro': test_metrics['average_precision_micro'],
                     # 'average_precision_macro': test_metrics['average_precision_macro'],
                     # 'average_precision_weighted': test_metrics['average_precision_weighted']
                     }

            path = f'.//RESULTS//{filename}'
            txt_filename = f'{path}//metrics.txt'
            if not os.path.exists(path):
                os.makedirs(path)
            with open(txt_filename, 'w') as f:
                for key, value in stats.items():
                    f.write(f'{key}: {value}\n')
                f.write('\n')


            fpr, tpr, roc_auc = test_metrics['fpr'], test_metrics['tpr'], test_metrics['roc_auc']
            # 绘制ROC曲线1
            plt.figure(figsize=(12, 8))
            plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Micro-average ROC Curve on {self.args.dataset}')
            plt.legend(loc="lower right")
            plt.savefig(f'{path}//ROC-Micro-average.png')
            plt.close()

            # 绘制ROC曲线2
            plt.figure(figsize=(12, 8))
            plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Macro-average ROC Curve on {self.args.dataset}')
            plt.legend(loc="lower right")
            plt.savefig(f'{path}//ROC-Macro-average.png')
            plt.close()

            # 绘制ROC曲线3
            plt.figure(figsize=(12, 8))
            plt.plot(fpr["weighted"], tpr["weighted"],
                     label=f'Weighted-average ROC curve (AUC = {roc_auc["weighted"]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Weighted-average ROC Curve on {self.args.dataset}')
            plt.legend(loc="lower right")
            plt.savefig(f'{path}//ROC-Weighted-average.png')
            plt.close()

            # 绘制ROC曲线4
            plt.figure(figsize=(12, 8))
            plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})')
            plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})')
            plt.plot(fpr["weighted"], tpr["weighted"],
                     label=f'Weighted-average ROC curve (AUC = {roc_auc["weighted"]:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve on {self.args.dataset}')
            plt.legend(loc="lower right")
            plt.savefig(f'{path}//ROC.png')
            plt.close()

            precision, recall, average_precision = test_metrics['precision'], test_metrics['recall'], test_metrics[
                'average_precision']
            # 绘制PR曲线1
            plt.figure(figsize=(12, 8))
            plt.plot(recall["micro"], precision["micro"],
                     label=f'Micro-average PR curve (AP = {average_precision["micro"]:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Micro-average Precision-Recall Curve on {self.args.dataset}')
            plt.legend(loc="lower left")
            plt.savefig(f'{path}//PR-Micro-average.png')
            plt.close()

            # 绘制PR曲线2
            plt.figure(figsize=(12, 8))
            plt.plot(recall["macro"], precision["macro"],
                     label=f'Macro-average PR curve (AP = {average_precision["macro"]:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Macro-average Precision-Recall Curve on {self.args.dataset}')
            plt.legend(loc="lower left")
            plt.savefig(f'{path}//PR-Macro-average.png')
            plt.close()

            # 绘制PR曲线3
            plt.figure(figsize=(12, 8))
            plt.plot(recall["weighted"], precision["weighted"],
                     label=f'Weighted-average PR curve (AP = {average_precision["weighted"]:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Weighted-average Precision-Recall Curve on {self.args.dataset}')
            plt.legend(loc="lower left")
            plt.savefig(f'{path}//PR-Weighted-average.png')
            plt.close()

            # 绘制PR曲线4
            plt.figure(figsize=(12, 8))
            plt.plot(recall["micro"], precision["micro"],
                     label=f'Micro-average PR curve (AP = {average_precision["micro"]:.2f})')
            plt.plot(recall["macro"], precision["macro"],
                     label=f'Macro-average PR curve (AP = {average_precision["macro"]:.2f})')
            plt.plot(recall["weighted"], precision["weighted"],
                     label=f'Weighted-average PR curve (AP = {average_precision["weighted"]:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve on {self.args.dataset}')
            plt.legend(loc="lower left")
            plt.savefig(f'{path}//PR.png')
            plt.close()

            # 绘制混淆矩阵
            if self.args.dataset in ['cf100', 'emn']:
                plt.figure(figsize=(100, 80))
                locator = mdates.AutoDateLocator(minticks=8, maxticks=18)  # 自动选择刻度位置!!!!!!!!!!!!!!!!!!!!!!!!!!
                ax = plt.gca()  # 获取当前坐标轴!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                ax.xaxis.set_major_locator(locator)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            else:
                plt.figure(figsize=(10, 8))
            # print('???????????????????????????????????')
            # print(test['confusion_matrix'].shape)
            sns.heatmap(test_metrics['confusion_matrix'], annot=True, fmt='g', cmap='Blues',
                        xticklabels=np.arange(test_metrics['num_classes']),
                        yticklabels=np.arange(test_metrics['num_classes']))
            plt.title(f'Confusion Matrix on {self.args.dataset}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            # plt.show()
            plt.savefig(f'{path}//confusion_matrix.png')
            # plt.savefig(f'{fold}//confusion_matrix_on_{self.args.dataset}.pdf', format="pdf",bbox_inches='tight', transparent=True)
            # with PdfPages(f'{fold}//confusion_matrix_on_{self.args.dataset}.pdf') as pdf:
            #     pdf.savefig()  # saves the current figure into a pdf page
            plt.close()  # close the figure to free memory

            # 绘制漏检率fnr和误警率fpr
            plt.figure(figsize=(12, 6))
            plt.plot(['micro', 'macro', 'weighted'],
                     [test_metrics['fnr_micro'], test_metrics['fnr_macro'], test_metrics['fnr_weighted']],
                     label='False Negative Rate (FNR)', marker='o')
            plt.plot(['micro', 'macro', 'weighted'],
                     [test_metrics['fpr_micro'], test_metrics['fpr_macro'], test_metrics['fpr_weighted']],
                     label='False Positive Rate (FPR)', marker='o')
            plt.title(f'False Negative Rate and False Positive Rate on {self.args.dataset}')
            plt.xlabel('Average Type')
            plt.ylabel('Rate')
            plt.legend(framealpha=1)
            plt.savefig(f'{path}//fnr_fpr.png')
            plt.close()

            # 绘制假阴性概率 (FNP) 和假阳性概率 (FTP)
            plt.figure(figsize=(12, 6))
            plt.plot(['micro', 'macro', 'weighted'],
                     [test_metrics['fnp_micro'], test_metrics['fnp_macro'], test_metrics['fnp_weighted']],
                     label='False Negative Probability (FNP)', marker='o')
            plt.plot(['micro', 'macro', 'weighted'],
                     [test_metrics['ftp_micro'], test_metrics['ftp_macro'], test_metrics['ftp_weighted']],
                     label='False Positive Probability (FTP)', marker='o')
            plt.title(f'False Negative Probability and False Positive Probability on {self.args.dataset}')
            plt.xlabel('Average Type')
            plt.ylabel('Probability')
            plt.legend(framealpha=1)
            plt.savefig(f'{path}//fnp_ftp.png')
            # plt.close()

            plt.close('all')  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            print(f'!!!!!!!!!!!!!!!!!!!!THE BEST RESULT UPDATE(END)!!!!!!!!!!!!!!!!!!!!')

        # metrics of each client.
        for i in range(len(train_metrics['num_correct'])):
            print("client: {}, training_acc: {:.6f}, training_loss: {:.6f}".format
                  (i, train_metrics['num_correct'][i] / train_metrics['num_samples'][i],
                   train_metrics['losses'][i] / train_metrics['num_samples'][i]))

        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        print('Train/Acc: {:.6f}, Train/Loss: {:.6f}'.format(train_acc, train_loss))

        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "Best/Acc": best_acc, "round": round_idx})
        print('Test/Acc: {:.6f}, Test/Loss: {:.6f}, Best/Acc: {:.6f}'.format(test_acc, test_loss, best_acc))

        return best_acc

    def validate_pruning_amount(self, model, layers_pr_dict, PRi):
        pass

    # FedLPS
    def _ln_prune_r18(self, glb_model, pr_prob, remove=0):
        # index: frozen stage; value: conv number until this stage.
        # Note: values in frozen_conv is 1 bigger than it in sqnet, mbnet, and sfnet.
        frozen_conv = [0, 5, 10, 15, 20]

        conv_count = 0
        down_count = 1  # r18's stage1 has no 'downsample' layer
        for name, module in glb_model.named_modules():
            if isinstance(module, nn.Conv2d):
                if conv_count == 0:  # The first conv layer in resnet.
                    conv_count += 1
                    continue

                if 'downsample' in name:
                    # The first downsample conv layer, only prune 'out_planes'.
                    if down_count == 0:
                        # Use the pruning probability in stage1(pr_prob[0]) to prune 'out_planes'(dim=0).
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[down_count], n=1, dim=0)
                        down_count += 1
                    else:  # The other downsample conv layer.
                        if down_count == 1:  # r18's stage1 has no 'downsample' layer
                            conv_count += 1
                            down_count += 1
                            continue
                        if not self.args.freeze_pruning and self.args.freeze_layer >= down_count + 1:  # Not prune frozen layers.
                            down_count += 1
                            conv_count += 1
                            continue
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[down_count - 1], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[down_count], n=1, dim=0)
                        down_count += 1
                    conv_count += 1
                    continue

                else:  # Normal conv layers in blocks.
                    if not self.args.freeze_pruning and frozen_conv[
                        self.args.freeze_layer] > conv_count:  # Not prune frozen Stage1.
                        conv_count += 1
                        continue
                    if conv_count == 1:  # Stage1's 1st conv layer.
                        # Pruning 'out_planes'.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=0)
                    elif conv_count <= 5:  # Stage1's other conv layers.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=0)

                    elif conv_count == 6:  # Stage2's 1st conv layer.
                        if self.args.freeze_layer != 1:
                            torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[1], n=1, dim=0)
                    elif conv_count <= 10:  # Stage2's other conv layers.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[1], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[1], n=1, dim=0)

                    elif conv_count == 11:  # Stage3's 1st conv layer.
                        if self.args.freeze_layer != 2:
                            torch_prune.ln_structured(module, name="weight", amount=pr_prob[1], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[2], n=1, dim=0)
                    elif conv_count <= 15:  # Stage3's other conv layers.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[2], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[2], n=1, dim=0)

                    elif conv_count == 16:  # Stage4's 1st conv layer.
                        if self.args.freeze_layer != 3:
                            torch_prune.ln_structured(module, name="weight", amount=pr_prob[2], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[3], n=1, dim=0)
                    elif conv_count <= 20:  # Stage4's other conv layers.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[3], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[3], n=1, dim=0)

                    conv_count += 1
                    continue

            elif isinstance(module, nn.BatchNorm2d):
                # 'conv_count' in nn.BatchNorm2d is 1 bigger than nn.Conv2d.
                if conv_count == 1:  # The 1st bn in resnet.
                    continue

                if not self.args.freeze_pruning and frozen_conv[
                    self.args.freeze_layer] + 1 >= conv_count:  # Not prune frozen Stage1.
                    continue
                if conv_count == 2:  # Stage1's 1st bn layer.
                    # Pruning 'out_planes'.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[0])
                elif conv_count <= 6:  # Stage1's other bn layers.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[0])

                elif conv_count == 7:  # Stage2's 1st conv layer.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[1])
                elif conv_count <= 11:  # Stage2's other conv layers.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[1])

                elif conv_count == 12:  # Stage3's 1st conv layer.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[2])
                elif conv_count <= 16:  # Stage3's other conv layers.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[2])

                elif conv_count == 17:  # Stage4's 1st conv layer.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[2])
                elif conv_count <= 21:  # Stage4's other conv layers.
                    torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[2])

            elif isinstance(module, nn.Linear) and self.args.freeze_layer != 4:
                torch_prune.ln_structured(module, name="weight", amount=pr_prob[-1], n=2, dim=1)

        # stat(glb_model, (3, 28, 28))
        # print(list(glb_model.named_buffers()))
        return glb_model

    def _ln_prune_sfnet(self, glb_model, pr_prob, remove=0):
        # frozen layer_num: {0, 1, 2, 3, 4}
        # corresponding stages: {0, 2, 3, 4, conv5.0} (not break stage block)
        frozen_modules = [0, 55, 149, 199, 203]  # index: layer_num; value: frozen modules until this layer_num.
        prune_count = 0

        for idx, m in enumerate(glb_model.named_modules()):
            if not self.args.freeze_pruning and frozen_modules[self.args.freeze_layer] >= idx:
                # Not prune frozen layers.
                continue

            if isinstance(m[1], nn.Conv2d):
                if "conv1" in m[0]:
                    # The first conv layer in sfnet.
                    continue
                if m[0] == "stage2.0.branch1.0" or prune_count == 0:  # stage2's 1st conv layer.
                    # Pruning 'out_planes'.
                    torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[0], n=1, dim=0)
                else:  # other conv layers.
                    torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[0], n=1, dim=1)
                    torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[0], n=1, dim=0)
                prune_count += 1

            elif isinstance(m[1], nn.BatchNorm2d):
                if "conv1" in m[0]:
                    # The first BN layer in sfnet.
                    continue
                torch_prune.l1_unstructured(m[1], name="weight", amount=pr_prob[0])

            elif isinstance(m[1], nn.Linear):
                if prune_count == 0:  # 1st layer in un-pruned sub-model.
                    if m[0] == "fc":  # The 1st un-pruned layer is the last fc layer.
                        break
                    else:
                        torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[-1], n=1, dim=0)
                        prune_count += 1
                        continue
                elif m[0] == "fc":  # last fc layer.
                    torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[-1], n=1, dim=1)
                    prune_count += 1
                    break
                else:  # other fc layers.
                    torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[-1], n=1, dim=1)
                    torch_prune.ln_structured(m[1], name="weight", amount=pr_prob[-1], n=1, dim=0)
                    prune_count += 1

        # stat(glb_model, (3, 28, 28))
        # print(list(glb_model.named_buffers()))
        # dummy_input = torch.randn(1, 3, 32, 32)  # .to(device)
        # flops, params = profile(glb_model, (dummy_input,))
        # print('flops: ', flops, 'params: ', params)
        # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
        return glb_model

    # FedDROP
    def feddrop_pruning(self, model, ratio, seed=None):
        pass
        # return model


    def extract_submodel_resnet_structure(self, model: nn.Module, ratio: float, round_n: int = 0):
        ratio = 1 - ratio
        model = copy.deepcopy(model)
        state_dict = model.state_dict()
        pruned_dict = OrderedDict()
        indices_dict = {}

        for key, weight in state_dict.items():
            if ('conv' in key) and ('weight' in key or 'bias' in key):
                out_channels = weight.size(0)
                new_out = int(out_channels * ratio)
                indices_out = torch.roll(torch.arange(out_channels), -round_n % out_channels)[:new_out]

                if 'weight' in key:
                    # 卷积weight是4维
                    in_channels = weight.size(1)
                    indices_in = torch.arange(in_channels)
                    indices_dict[key] = (indices_out, indices_in)
                    pruned_dict[key] = weight[indices_out][:, indices_in, :, :]
                else:
                    # 卷积bias是一维，按输出通道剪
                    indices_dict[key] = (indices_out,)
                    pruned_dict[key] = weight[indices_out]

            else:
                pruned_dict[key] = weight

        return pruned_dict, indices_dict

    def build_pruned_model_with_masks(self, global_model: nn.Module,
                                      submodel_sd: dict,
                                      indices_dict: dict) -> nn.Module:
        # 深拷贝一个模型结构
        model = copy.deepcopy(global_model)

        # 获取原模型参数（完整）
        full_sd = global_model.state_dict()

        # 遍历子模型中所有存在剪枝的层（来自 submodel_sd 和 indices_dict）
        for name, sub_param in submodel_sd.items():
            if name not in indices_dict:
                continue  # 未剪枝的跳过

            # 从原始模型中获取该参数
            full_param = full_sd[name]
            mask = torch.zeros_like(full_param)

            if len(full_param.shape) == 1:
                # 对于 1D 参数（如 bias） 输出通道
                indices = indices_dict[name][0]
                mask[indices] = 1

            elif len(full_param.shape) == 4:
                indices_out, indices_in = indices_dict[name]
                for i in indices_out:
                    for j in indices_in:
                        mask[i, j, :, :] = 1
            else:
                raise NotImplementedError(f"Unsupported param shape: {name} {full_param.shape}")

            # 使用 torch.nn.utils.prune 自定义掩码方式
            module_name, param_name = name.rsplit('.', 1)
            module = dict(model.named_modules())[module_name]

            torch_prune.custom_from_mask(module, param_name, mask)

        return model

    def fedrolex_pruning(self, model, ratio, round_n=None):
        # if self.args.model in ["pre_r18", "pre_resnet18"]:
        sub_model, indices_dict = self.extract_submodel_resnet_structure(model, ratio, round_n=round_n)
        # elif self.args.model in ["pre_ShuffleNet", "pre_shufflenet", "pre_sfnet"]:
        #     sub_model, indices_dict = self.extract_submodel_shufflenet_structure(model, ratio, round_n=round_n)
        # else:
        #     raise ValueError(f"Unsupported model: {self.args.model}")
        pruned_model = self.build_pruned_model_with_masks(model, sub_model, indices_dict)
        return pruned_model
