import logging
import os
import torch
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, roc_curve, average_precision_score, auc
from scipy import interp
from sklearn.preprocessing import label_binarize


try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def get_model(self):
        return self.model.cpu()

    def set_model(self, model):
        self.model = model

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    # def train(self, train_data, device, args, alpha, local_update_last, global_update_last, global_model_param, hist_i):
    #
    #     state_update_diff = torch.tensor(-local_update_last + global_update_last, dtype=torch.float32).cuda()
    #
    #     model = self.model
    #
    #     if args.dataparallel == 1:
    #         model = nn.DataParallel(model)
    #     # else:
    #     #     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    #
    #     # model.to(device)
    #     model.cuda()
    #     model.train()
    #
    #     # train and update
    #     # criterion = nn.CrossEntropyLoss().to(device)
    #     criterion = nn.CrossEntropyLoss().cuda()
    #     if args.client_optimizer == "sgd":
    #         optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
    #     else:
    #         optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
    #                                      weight_decay=args.wd, amsgrad=False)
    #
    #     epoch_loss = []
    #     for epoch in range(args.epochs):
    #         batch_loss = []
    #         for batch_idx, (x, labels) in enumerate(train_data):
    #             # x, labels = x.to(device), labels.to(device)
    #             x, labels = x.cuda(), labels.cuda()
    #             model.zero_grad()
    #             log_probs = model(x)
    #             loss_l = criterion(log_probs, labels)
    #             # # todo
    #             #
    #             # theta_now = model.state_dict()
    #             # # 添加惩罚项
    #             # # tmp = [h[key].cuda() + theta_now[key].cuda() - theta[key].cuda() for key in h.keys()]
    #             # # penalty = torch.sqrt(sum(torch.sum(t ** 2) for t in tmp))
    #             # penalty_terms = []
    #             # for key in h.keys():
    #             #     penalty_terms.append((theta_now[key].cuda() + h[key].cuda() - theta[key].cuda()) ** 2)
    #             # penalty = torch.sqrt(torch.sum(torch.stack(penalty_terms)))
    #             #
    #             # # 添加梯度修正项
    #             # # 初始化内积结果
    #             # # gradient_correction = 0
    #             # # # 遍历权重字典的每个键
    #             # # for key in theta_now.keys():
    #             # #     # 提取对应的张量
    #             # #     tensor1 = theta_now[key].cuda()
    #             # #     tensor2 = g_i[key].cuda() - g[key].cuda()
    #             # #     # 计算对应张量的内积
    #             # #     key_inner_product = torch.sum(tensor1 * tensor2)
    #             # #     # 将内积结果累加
    #             # #     gradient_correction += (1 / (args.lr * args.epochs)) * key_inner_product
    #             # gradient_correction_terms = []
    #             # for key in theta_now.keys():
    #             #     gradient_correction_terms.append(theta_now[key].cuda() * (g_i[key].cuda() - g[key].cuda()))
    #             # gradient_correction = torch.sum(torch.stack(gradient_correction_terms)) / (args.lr * args.epochs)
    #
    #             # 综合目标函数
    #             # loss = loss + 0.5 * alpha * penalty + gradient_correction
    #
    #
    #             local_parameter = None
    #             for param in model.parameters():
    #                 if not isinstance(local_parameter, torch.Tensor):
    #                     # Initially nothing to concatenate
    #                     local_parameter = param.reshape(-1)
    #                 else:
    #                     local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)
    #
    #             loss_cp = alpha / 2 * torch.sum((local_parameter - (global_model_param - hist_i)) * (local_parameter - (global_model_param - hist_i)))
    #             loss_cg = torch.sum(local_parameter * state_update_diff)
    #
    #             loss = loss_l + loss_cp + loss_cg
    #
    #
    #             loss.backward()
    #
    #             # Uncommet this following line to avoid nan loss
    #             # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    #
    #             optimizer.step()
    #             batch_loss.append(loss.item())
    #         epoch_loss.append(sum(batch_loss) / len(batch_loss))
    #         logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
    #             self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

    def get_params_for_pr(self, pr, model_list, n_par=None):

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
        if pr:
            for w in param_mat:
                w[-1], w[-2] = w[-2], w[-1]
        return np.copy(param_mat)

    def train(self, train_data, device, args, round_idx, alpha, local_update_last, global_update_last, global_model_param, hist_i):  # ok

        print(f'--- Training Client {self.id}')

        lr_decay_per_round = 0.998
        learning_rate = args.lr * (lr_decay_per_round ** round_idx)
        n_trn = len(train_data.dataset)
        print_per = 1
        weight_decay = 1e-3
        sch_step = 1
        sch_gamma = 1
        state_update_diff = torch.tensor(-local_update_last + global_update_last, dtype=torch.float32).cuda()
        # state_update_diff = torch.tensor(local_update_last - global_update_last, dtype=torch.float32).cuda()
        trn_gen = train_data
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

        model = self.model

        if args.dataparallel == 1:
            model = nn.DataParallel(model)

        # model.train()
        # model = model.cuda()
        model.cuda()
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

        n_par = self.get_params_for_pr(args.pr, [model]).shape[1]

        for e in range(args.epochs):
            # Training
            epoch_loss = 0
            # trn_gen_iter = trn_gen.__iter__()
            # print('='*50)
            # print('n_trn: ', n_trn)
            # print('='*50)
            # for i in range(int(np.ceil(n_trn / args.batch_size))):  # 逐批
            for batch_idx, (batch_x, batch_y) in enumerate(train_data):
                # batch_x, batch_y = trn_gen_iter.__next__()
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

                y_pred = model(batch_x)

                ## Get f_i estimate
                loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())

                loss_f_i = loss_f_i / list(batch_y.size())[0]
                # print('loss_f_i: ', loss_f_i.item())

                # local_parameter = None
                # for param in model.parameters():
                #     if not isinstance(local_parameter, torch.Tensor):
                #         # Initially nothing to concatenate
                #         local_parameter = param.reshape(-1)
                #     else:
                #         local_parameter = torch.cat((local_parameter, param.reshape(-1)), 0)
                local_parameter = self.get_params_for_pr(args.pr, [model], n_par)[0]
                local_parameter = torch.tensor(local_parameter, dtype=torch.float32).cuda()

                loss_cp = alpha / 2 * torch.sum((local_parameter - (global_model_param - hist_i)) * (
                            local_parameter - (global_model_param - hist_i)))
                loss_cg = torch.sum(local_parameter * state_update_diff)
                # loss_cg = torch.sum(local_parameter * state_update_diff) / (args.lr * args.epochs)

                # print('=' * 50)
                # print(f'local_parameter: {local_parameter}, state_update_diff: {state_update_diff}')
                # print(f'loss_f_i: {loss_f_i}, loss_cp: {loss_cp}, loss_cg: {loss_cg}')
                # print('='*50)
                loss = loss_f_i + loss_cp + loss_cg if args.dc else loss_f_i
                # loss = loss_f_i + loss_cp
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)  # Clip gradients to prevent exploding
                optimizer.step()
                epoch_loss += loss.item() * list(batch_y.size())[0]

            if (e + 1) % print_per == 0:
                epoch_loss /= n_trn
                if weight_decay != None:
                    # Add L2 loss to complete f_i
                    params = self.get_params_for_pr(args.pr, [model], n_par)
                    epoch_loss += (weight_decay) / 2 * np.sum(params * params)

                print("Epoch: {}, Training Loss: {:.6f}, LR: {:.6f}".format(e, epoch_loss, scheduler.get_lr()[0]))

                model.train()
            scheduler.step()

        # # Freeze model
        # for params in model.parameters():
        #     params.requires_grad = False
        # model.eval()

        return model


    # def test(self, test_data, device, args):
    #     model = self.model
    #
    #     if args.dataparallel == 1:
    #         model = nn.DataParallel(model)
    #
    #     metrics = {
    #         'test_correct': 0,
    #         'test_loss': 0,
    #         'test_total': 0
    #     }
    #
    #     w_decay = 0
    #
    #     acc_overall = 0
    #     loss_overall = 0
    #     loss_fn = torch.nn.CrossEntropyLoss(reduction='sum').cuda()
    #
    #     n_tst = len(test_data.dataset)
    #     batch_size = min(6000, n_tst)
    #     batch_size = min(2000, n_tst)
    #     tst_gen = test_data
    #     model.eval()
    #     model = model.cuda()
    #     with torch.no_grad():
    #         # tst_gen_iter = tst_gen.__iter__()
    #         # for i in range(int(np.ceil(n_tst / batch_size))):
    #         for batch_idx, (batch_x, batch_y) in enumerate(test_data):
    #             # batch_x, batch_y = tst_gen_iter.__next__()
    #             batch_x = batch_x.cuda()
    #             batch_y = batch_y.cuda()
    #             y_pred = model(batch_x)
    #
    #             loss = loss_fn(y_pred, batch_y.reshape(-1).long())
    #
    #             loss_overall += loss.item()
    #
    #             # Accuracy calculation
    #             y_pred = y_pred.cpu().numpy()
    #             y_pred = np.argmax(y_pred, axis=1).reshape(-1)
    #             batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
    #             batch_correct = np.sum(y_pred == batch_y)
    #             acc_overall += batch_correct
    #
    #     # loss_overall /= n_tst
    #     if w_decay != None:
    #         # Add L2 loss
    #         params = self.get_params([model], n_par=None)
    #         loss_overall += w_decay / 2 * np.sum(params * params) * n_tst
    #
    #     model.train()
    #
    #     metrics['test_correct'] = acc_overall
    #     metrics['test_loss'] = loss_overall
    #     metrics['test_total'] = n_tst
    #     # print('='*50)
    #     # print(metrics)
    #     return metrics


    def test(self, test_data, device, args, b_use_test_dataset):  # ok
        model = self.model

        if args.dataparallel == 1:
            model = nn.DataParallel(model)

        model.cuda()
        model.eval()

        metrics = {
            'num_correct': 0,
            'losses': 0,
            'num_samples': 0
        }

        # criterion = nn.CrossEntropyLoss().to(device)
        criterion = nn.CrossEntropyLoss().cuda()

        # eva
        all_targets = []
        all_predictions = []
        all_pred_proba = []

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.cuda()
                target = target.cuda()
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['num_correct'] += correct.item()
                metrics['losses'] += loss.item() * target.size(0)
                metrics['num_samples'] += target.size(0)

                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_pred_proba.append(torch.nn.functional.softmax(pred, dim=1).cpu().numpy())  # 计算概率
        if b_use_test_dataset:
            # 计算评估指标
            all_targets = np.array(all_targets)
            all_predictions = np.array(all_predictions)
            all_pred_proba = np.concatenate(all_pred_proba)  # 合并概率数组

            confusion = confusion_matrix(all_targets, all_predictions)
            accuracy = accuracy_score(all_targets, all_predictions)
            precision_micro = precision_score(all_targets, all_predictions, average='micro')
            precision_macro = precision_score(all_targets, all_predictions, average='macro')
            precision_weighted = precision_score(all_targets, all_predictions, average='weighted')
            recall_micro = recall_score(all_targets, all_predictions, average='micro')
            recall_macro = recall_score(all_targets, all_predictions, average='macro')
            recall_weighted = recall_score(all_targets, all_predictions, average='weighted')
            f1_micro = f1_score(all_targets, all_predictions, average='micro')
            f1_macro = f1_score(all_targets, all_predictions, average='macro')
            f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
            average_auc_micro = roc_auc_score(np.eye(len(np.unique(all_targets)))[all_targets], all_pred_proba, multi_class='ovr', average='micro')
            average_auc_macro = roc_auc_score(np.eye(len(np.unique(all_targets)))[all_targets], all_pred_proba, multi_class='macro', average='macro')
            average_auc_weighted = roc_auc_score(np.eye(len(np.unique(all_targets)))[all_targets], all_pred_proba, multi_class='ovr', average='weighted')
            # average_precision_micro = average_precision_score(np.eye(len(np.unique(all_targets)))[all_targets], all_pred_proba, average='micro')
            # average_precision_macro = average_precision_score(np.eye(len(np.unique(all_targets)))[all_targets], all_pred_proba, average='macro')
            # average_precision_weighted = average_precision_score(np.eye(len(np.unique(all_targets)))[all_targets], all_pred_proba, average='weighted')

            # # 绘制PR曲线
            # import matplotlib.pyplot as plt
            # from sklearn.metrics import PrecisionRecallDisplay
            #
            # plt.figure(figsize=(12, 8))
            #
            # PrecisionRecallDisplay.from_predictions(all_targets, all_pred_proba[:, 1], average='micro', name='Micro-average')
            # PrecisionRecallDisplay.from_predictions(all_targets, all_pred_proba[:, 1], average='macro', name='Macro-average')
            # PrecisionRecallDisplay.from_predictions(all_targets, all_pred_proba[:, 1], average='weighted', name='Weighted-average')
            #
            # plt.title('Precision-Recall curve')
            # plt.legend()
            # plt.show()
            #
            # # 绘制ROC曲线
            # from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
            #
            # plt.figure(figsize=(12, 8))
            #
            # RocCurveDisplay.from_predictions(all_targets, all_pred_proba[:, 1], average='micro', name='Micro-average')
            # RocCurveDisplay.from_predictions(all_targets, all_pred_proba[:, 1], average='macro', name='Macro-average')
            # RocCurveDisplay.from_predictions(all_targets, all_pred_proba[:, 1], average='weighted', name='Weighted-average')
            #
            # plt.title('ROC curve')
            # plt.legend()
            # plt.show()

            # 计算漏检率和误警率
            fnr_micro = 1 - recall_micro
            fnr_macro = 1 - recall_macro
            fnr_weighted = 1 - recall_weighted
            fpr_micro = 1 - precision_micro  # 计算微平均 FPR
            fpr_macro = 1 - precision_macro
            fpr_weighted = 1 - precision_weighted

            # # 计算假阴性概率 (FNP) 和假阳性概率 (FTP)
            # num_classes = all_pred_proba.shape[1]
            # fnp_micro = np.zeros(num_classes)
            # ftp_micro = np.zeros(num_classes)
            # fnp_macro = np.zeros(num_classes)
            # ftp_macro = np.zeros(num_classes)
            # fnp_weighted = np.zeros(num_classes)
            # ftp_weighted = np.zeros(num_classes)
            #
            # for i in range(num_classes):
            #     true_positive = np.sum((all_targets == i) & (all_predictions == i))
            #     false_negative = np.sum((all_targets == i) & (all_predictions != i))
            #     false_positive = np.sum((all_targets != i) & (all_predictions == i))
            #     true_negative = np.sum((all_targets != i) & (all_predictions != i))
            #
            #     fnp_micro[i] = false_negative / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            #     ftp_micro[i] = false_positive / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
            #
            #     fnp_macro[i] = false_negative / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            #     ftp_macro[i] = false_positive / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
            #
            #     fnp_weighted[i] = false_negative / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            #     ftp_weighted[i] = false_positive / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
            #
            # fnp_micro_avg = np.mean(fnp_micro)
            # ftp_micro_avg = np.mean(ftp_micro)
            # fnp_macro_avg = np.mean(fnp_macro)
            # ftp_macro_avg = np.mean(ftp_macro)
            # fnp_weighted_avg = np.mean(fnp_weighted)
            # ftp_weighted_avg = np.mean(ftp_weighted)


            num_classes = len(np.unique(all_targets))

            # 初始化每个类别的 FNP 和 FTP
            fnp = np.zeros(num_classes)
            ftp = np.zeros(num_classes)
            # 初始化每个类别的样本数量
            class_counts = np.zeros(num_classes)

            # 计算每个类别的 TP、FP、TN、FN
            for i in range(num_classes):
                true_positive = np.sum((all_targets == i) & (all_predictions == i))
                false_negative = np.sum((all_targets == i) & (all_predictions != i))
                false_positive = np.sum((all_targets != i) & (all_predictions == i))
                true_negative = np.sum((all_targets != i) & (all_predictions != i))

                # 计算每个类别的 FNP 和 FTP
                fnp[i] = false_negative / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
                ftp[i] = false_positive / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
                # 记录每个类别的样本数量
                class_counts[i] = np.sum(all_targets == i)

            # 计算 Micro 平均 FNP 和 FTP
            total_true_positive = np.sum(
                [np.sum((all_targets == i) & (all_predictions == i)) for i in range(num_classes)])
            total_false_negative = np.sum(
                [np.sum((all_targets == i) & (all_predictions != i)) for i in range(num_classes)])
            total_false_positive = np.sum(
                [np.sum((all_targets != i) & (all_predictions == i)) for i in range(num_classes)])
            total_true_negative = np.sum(
                [np.sum((all_targets != i) & (all_predictions != i)) for i in range(num_classes)])

            fnp_micro_avg = total_false_negative / (total_true_positive + total_false_negative) if (total_true_positive + total_false_negative) > 0 else 0
            ftp_micro_avg = total_false_positive / (total_true_negative + total_false_positive) if (total_true_negative + total_false_positive) > 0 else 0

            # 计算 Macro 平均 FNP 和 FTP
            fnp_macro_avg = np.mean(fnp)
            ftp_macro_avg = np.mean(ftp)

            # 计算 Weighted 平均 FNP 和 FTP
            total_samples = np.sum(class_counts)
            fnp_weighted_avg = np.sum(fnp * class_counts) / total_samples
            ftp_weighted_avg = np.sum(ftp * class_counts) / total_samples


            # 二值化标签，准备计算FPR, TPR, Precision, Recall
            all_targets_bin = label_binarize(all_targets, classes=range(num_classes))

            # 计算每个类别的 FPR, TPR, Precision, Recall
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            precision = dict()
            recall = dict()
            average_precision = dict()

            for i in range(num_classes):
                fpr[i], tpr[i], _ = roc_curve(all_targets_bin[:, i], all_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                precision[i], recall[i], _ = precision_recall_curve(all_targets_bin[:, i], all_pred_proba[:, i])
                average_precision[i] = average_precision_score(all_targets_bin[:, i], all_pred_proba[:, i])

            # 计算 micro-average ROC 和 PR 曲线
            fpr["micro"], tpr["micro"], _ = roc_curve(all_targets_bin.ravel(), all_pred_proba.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            precision["micro"], recall["micro"], _ = precision_recall_curve(all_targets_bin.ravel(),
                                                                            all_pred_proba.ravel())
            average_precision["micro"] = average_precision_score(all_targets_bin, all_pred_proba, average="micro")

            # 计算 macro-average ROC 和 PR 曲线
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(num_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= num_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            all_recall = np.unique(np.concatenate([recall[i] for i in range(num_classes)]))
            mean_precision = np.zeros_like(all_recall)
            for i in range(num_classes):
                mean_precision += np.interp(all_recall, recall[i][::-1], precision[i][::-1])
            mean_precision /= num_classes
            precision["macro"] = mean_precision
            recall["macro"] = all_recall
            average_precision["macro"] = average_precision_score(all_targets_bin, all_pred_proba, average="macro")

            # 计算 weighted-average ROC 和 PR 曲线
            # 这里需要根据每个类别的样本数量进行加权
            weights = np.bincount(all_targets) / len(all_targets)
            fpr["weighted"] = all_fpr
            tpr["weighted"] = np.zeros_like(all_fpr)
            for i in range(num_classes):
                tpr["weighted"] += np.interp(all_fpr, fpr[i], tpr[i]) * weights[i]
            roc_auc["weighted"] = auc(fpr["weighted"], tpr["weighted"])

            precision["weighted"] = np.zeros_like(all_recall)
            for i in range(num_classes):
                precision["weighted"] += np.interp(all_recall, recall[i][::-1], precision[i][::-1]) * weights[i]
            recall["weighted"] = all_recall
            average_precision["weighted"] = average_precision_score(all_targets_bin, all_pred_proba, average="weighted")

            metrics.update({
                'confusion_matrix': confusion,
                'accuracy': accuracy,
                'precision_micro': precision_micro,
                'precision_macro': precision_macro,
                'precision_weighted': precision_weighted,
                'recall_micro': recall_micro,
                'recall_macro': recall_macro,
                'recall_weighted': recall_weighted,
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'auc_micro': average_auc_micro,
                'auc_macro': average_auc_macro,
                'auc_weighted': average_auc_weighted,
                # 'average_precision_micro': average_precision_micro,
                # 'average_precision_macro': average_precision_macro,
                # 'average_precision_weighted': average_precision_weighted,
                'fnr_micro': fnr_micro,
                'fnr_macro': fnr_macro,
                'fnr_weighted': fnr_weighted,
                'fpr_micro': fpr_micro,  # 微平均 FPR
                'fpr_macro': fpr_macro,
                'fpr_weighted': fpr_weighted,
                'fnp_micro': fnp_micro_avg,
                'fnp_macro': fnp_macro_avg,
                'fnp_weighted': fnp_weighted_avg,
                'ftp_micro': ftp_micro_avg,
                'ftp_macro': ftp_macro_avg,
                'ftp_weighted': ftp_weighted_avg,
                'num_classes': num_classes,
                # 'all_targets': all_targets,
                # 'all_pred_proba': all_pred_proba,
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc,
                'precision': precision,
                'recall': recall,
                'average_precision': average_precision
            })

        # print('=' * 50)
        # print(metrics)
        return metrics






    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
