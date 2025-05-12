import csv
import logging
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from ..utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMSA')


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse


class TMR:
    def __init__(self, args):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_re = nn.CrossEntropyLoss(reduction='none')
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()

    def do_train(self, model, feature_statistic, dataloader, return_epoch_results=False):
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        label_all = np.round(dataloader['train'].dataset.labels['M']) + 3
        class_counts = torch.tensor(np.bincount(label_all.astype(int)), dtype=torch.float)
        class_weights = class_counts.max() / class_counts
        # max_weight = 10.0
        # class_weights = torch.clamp(class_weights, max=max_weight)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.args.device))
        model.train()

        while True:
            epochs += 1
            if epochs > self.args.init_epoch:
                l_file = f'/home/data2t1/xxx/MSA/MTL/log/umap/features_l_{epochs}.csv'
                with open(l_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    #writer.writerow([f'feature_{i}' for i in range(1380)])
                    writer.writerow([f'feature_{i}' for i in range(2300)])
                v_file = f'/home/data2t1/xxx/MSA/MTL/log/umap/features_v_{epochs}.csv'
                with open(v_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    #writer.writerow([f'feature_{i}' for i in range(1440)])
                    writer.writerow([f'feature_{i}' for i in range(2300)])
                a_file = f'/home/data2t1/xxx/MSA/MTL/log/umap/features_a_{epochs}.csv'
                with open(a_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    #writer.writerow([f'feature_{i}' for i in range(1500)])
                    writer.writerow([f'feature_{i}' for i in range(2300)])

                label_file = f'/home/data2t1/xxx/MSA/MTL/log/umap/labels_{epochs}.csv'
                with open(label_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow('y')
            y_pred, y_true = [], []

            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = (torch.round(labels.view(-1, 1)) + 3).long().squeeze()

                    output = model(text, audio, vision)

                    # TODO: calculate typicality and consistency
                    w_l = 0.0
                    w_v = 0.0
                    w_a = 0.0
                    consistency = 0
                    features = [output['feature_l'], output['feature_v'], output['feature_a']]
                    confidence, pred = F.normalize(output['logits_output_basic'], dim=1).max(1)

                    # loss
                    loss_basic = self.criterion(output['logits_output_basic'], labels)
                    loss_all = None
                    if epochs > self.args.init_epoch:
                        with open(f'/home/data2t1/xxx/MSA/MTL/log/umap/labels_{epochs}.csv', mode='a',
                                  newline='') as file:
                            writer = csv.writer(file)
                            # 将隐藏层向量转为 numpy 并逐行写入
                            ls = labels.detach().cpu().numpy()  # [16, 1]
                            for label in ls:
                                writer.writerow([label])
                        with open(f'/home/data2t1/xxx/MSA/MTL/log/umap/features_l_{epochs}.csv', mode='a',
                                  newline='') as file:
                            writer = csv.writer(file)
                            features_l = features[0].detach().cpu().numpy()  # [16, 2300]
                            writer.writerows(features_l)
                        with open(f'/home/data2t1/xxx/MSA/MTL/log/umap/features_v_{epochs}.csv', mode='a',
                                  newline='') as file:
                            writer = csv.writer(file)
                            features_v = features[1].detach().cpu().numpy()  # [16, 2300]
                            writer.writerows(features_v)
                        with open(f'/home/data2t1/xxx/MSA/MTL/log/umap/features_a_{epochs}.csv', mode='a',
                                  newline='') as file:
                            writer = csv.writer(file)
                            features_a = features[2].detach().cpu().numpy()  # [16, 2300]
                            writer.writerows(features_a)
                        # loss_l_late = self.criterion(output['logits_l_late'], labels)
                        # loss_v_late = self.criterion(output['logits_v_late'], labels)
                        # loss_a_late = self.criterion(output['logits_a_late'], labels)
                        loss_l_late = self.criterion_re(output['logits_l_late'], labels)
                        loss_v_late = self.criterion_re(output['logits_v_late'], labels)
                        loss_a_late = self.criterion_re(output['logits_a_late'], labels)

                        # base loss
                        l_typicality, v_typicality, a_typicality, consistency, consistency_label, cpcc_loss, loss_center = feature_statistic.forward(
                            features, labels, pred=pred, confidence=confidence, modify_T=True)
                        loss_base = loss_basic + 1 * cpcc_loss + 0.1 * loss_center
                        # late loss
                        typicality_sum = l_typicality + v_typicality + a_typicality
                        w_l = l_typicality / typicality_sum + 1e-8
                        w_v = v_typicality / typicality_sum + 1e-8
                        w_a = a_typicality / typicality_sum + 1e-8
                        # 使用 torch.isfinite() 来检查是否有 NaN 或 Inf
                        w_l[~torch.isfinite(w_l)] = 1 / 3
                        w_v[~torch.isfinite(w_v)] = 1 / 3
                        w_a[~torch.isfinite(w_a)] = 1 / 3
                        loss_late = (w_l.unsqueeze(-1) * loss_l_late).mean() + (
                                w_v.unsqueeze(-1) * loss_v_late).mean() + (w_a.unsqueeze(-1) * loss_a_late).mean()
                        # share early loss
                        loss_task_share = self.criterion_re(output['logits_c_share'], labels)
                        consistency = consistency.unsqueeze(-1)
                        loss_all = loss_base + (consistency * loss_task_share).mean() + (
                                (1 - consistency) * loss_late).mean()

                        # pred
                        pre_share = output['logits_c_share']
                        pre_late = w_l.unsqueeze(-1) * output['logits_l_late'] + w_v.unsqueeze(-1) * output[
                            'logits_v_late'] + w_l.unsqueeze(-1) * output['logits_a_late']
                        #pre = output['logits_output_basic'] + pre_share + pre_late
                        pre = output['logits_output_basic'] + consistency.expand(-1, 7) * pre_share + 3 * (
                                    1 - consistency.expand(-1, 7)) * pre_late

                        _, pred = F.normalize(pre, dim=1).max(1)
                    else:
                        _ = feature_statistic.forward(features, labels, pred=pred, confidence=confidence,
                                                      modify_T=False)
                        loss_l_late = self.criterion(output['logits_l_late'], labels)
                        loss_v_late = self.criterion(output['logits_v_late'], labels)
                        loss_a_late = self.criterion(output['logits_a_late'], labels)
                        loss_task_share = self.criterion(output['logits_c_share'], labels)
                        loss_all = loss_basic + loss_task_share + loss_l_late + loss_v_late + loss_a_late
                    loss_all.backward()

                    # if self.args.grad_clip != -1.0:
                    #     nn.utils.clip_grad_value_(model.parameters(), self.args.grad_clip)
                    train_loss += loss_all.item()

                    y_pred.append(pred.cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(epochs, model, feature_statistic, dataloader['valid'], mode="VAL")
            test_results = self.do_test(epochs, model, feature_statistic, dataloader['test'], mode="TEST",
                                        analyze_k=True)

            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save each epoch model
            torch.save(model.state_dict(), './pt/' + str(epochs) + '.pth')
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                model_save_path = './pt/dmd.pth'
                torch.save(model.state_dict(), model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, epochs, model, feature_statistic, dataloader, mode="VAL", analyze_k=False):

        model.eval()
        y_pred, y_true = [], []
        consistency_list, cl, kl = [], [], []
        eval_loss = 0.0
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = (torch.round(labels.view(-1, 1)) + 3).long().squeeze()

                    w_l = 0.0
                    w_v = 0.0
                    w_a = 0.0
                    consistency = 0
                    output = model(text, audio, vision)
                    features = [output['feature_l'], output['feature_v'], output['feature_a']]
                    confidence, pred = F.normalize(output['logits_output_basic'], dim=1).max(1)

                    # loss
                    loss_basic = self.criterion(output['logits_output_basic'], labels)
                    if epochs > self.args.init_epoch:
                        loss_l_late = self.criterion_re(output['logits_l_late'], labels)
                        loss_v_late = self.criterion_re(output['logits_v_late'], labels)
                        loss_a_late = self.criterion_re(output['logits_a_late'], labels)
                        # base loss
                        l_typicality, v_typicality, a_typicality, consistency, consistency_label, cpcc_loss, loss_center = feature_statistic.forward(
                            features, labels, pred=pred, confidence=confidence, modify_T=True, append=False)
                        consistency_list.append(consistency.cpu())
                        loss_base = loss_basic + 1 * cpcc_loss + 0.01 * loss_center
                        # late loss
                        typicality_sum = l_typicality + v_typicality + a_typicality
                        w_l = l_typicality / typicality_sum + 1e-8
                        w_v = v_typicality / typicality_sum + 1e-8
                        w_a = a_typicality / typicality_sum + 1e-8
                        # 使用 torch.isfinite() 来检查是否有 NaN 或 Inf
                        w_l[~torch.isfinite(w_l)] = 1 / 3
                        w_v[~torch.isfinite(w_v)] = 1 / 3
                        w_a[~torch.isfinite(w_a)] = 1 / 3
                        loss_late = (w_l.unsqueeze(-1) * loss_l_late).mean() + (
                                w_v.unsqueeze(-1) * loss_v_late).mean() + (w_a.unsqueeze(-1) * loss_a_late).mean()
                        # share early loss
                        loss_task_share = self.criterion_re(output['logits_c_share'], labels)
                        consistency = consistency.unsqueeze(-1)
                        loss_all = loss_base + (consistency * loss_task_share).mean() + (
                                (1 - consistency) * loss_late).mean()
                        # pred
                        pre_share = output['logits_c_share']
                        pre_late = w_l.unsqueeze(-1) * output['logits_l_late'] + w_v.unsqueeze(-1) * output[
                            'logits_v_late'] + w_l.unsqueeze(-1) * output['logits_a_late']
                        #pre = output['logits_output_basic'] + pre_share + pre_late
                        pre = output['logits_output_basic'] + consistency.expand(-1, 7) * pre_share + (
                                    1 - consistency.expand(-1, 7)) * pre_late

                        _, pred = F.normalize(pre, dim=1).max(1)
                        if analyze_k:
                            cl.append(consistency_label.cpu())
                            ll = F.softmax(output['logits_l_late'], dim=-1).cpu()
                            lv = F.softmax(output['logits_v_late'], dim=-1).cpu()
                            la = F.softmax(output['logits_a_late'], dim=-1).cpu()
                            kl_lv = torch.sum(ll * torch.log(ll / lv), dim=-1)
                            kl_va = torch.sum(lv * torch.log(lv / la), dim=-1)
                            kl_al = torch.sum(la * torch.log(la / ll), dim=-1)
                            x = kl_lv + kl_va + kl_al
                            min_val = x.min()
                            max_val = x.max()
                            kl.append((max_val - x) / (max_val - min_val))

                    else:
                        _ = feature_statistic.forward(features, labels, pred=pred, confidence=confidence,
                                                      modify_T=False)
                        loss_l_late = self.criterion(output['logits_l_late'], labels)
                        loss_v_late = self.criterion(output['logits_v_late'], labels)
                        loss_a_late = self.criterion(output['logits_a_late'], labels)
                        loss_task_share = self.criterion(output['logits_c_share'], labels)
                        loss_all = loss_basic + loss_task_share + loss_l_late + loss_v_late + loss_a_late
                    eval_loss += loss_all.item()

                    y_pred.append(pred.cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        if epochs > self.args.init_epoch and analyze_k:
            self.analyze_consistency(pred, true, torch.cat(consistency_list), torch.cat(cl), torch.cat(kl), epochs,
                                     mode='acc')
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        return eval_results

    def analyze_consistency(self, y_pred, y_true, consistency, cl, kl, epoch, mode=None):
        if mode == 'acc':
            y_pred = y_pred.numpy()
            y_true = y_true.numpy()
            consistency = consistency.numpy()

            quantiles = np.percentile(consistency, [0, 25, 50, 55, 60, 70, 80, 90, 100])

            # 将 consistency 按照 quantiles 划分为不同的组
            # 使用 numpy.digitize 将数据分配到相应的组
            groups = np.digitize(consistency, quantiles, right=True)

            # 计算每个组的准确率
            accuracies_7 = []
            accuracies_2 = []
            for group in range(0, len(quantiles) - 1):
                # 获取属于当前组的样本索引
                group_indices = np.where(groups == group)[0]

                g_pred = y_pred[group_indices]
                g_true = y_true[group_indices]
                # 计算当前组的7class准确率
                correct_predictions_7 = np.sum(g_pred == g_true)
                accuracy_7 = correct_predictions_7 / len(group_indices)  # 当前组的准确率
                accuracies_7.append(accuracy_7)

                # 计算当前组的2class准确率
                non_zeros = np.array([i for i, e in enumerate(g_true) if e != 3])
                non_zeros_binary_truth = (g_true[non_zeros] > 3)
                non_zeros_binary_preds = (g_pred[non_zeros] > 3)

                non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
                accuracies_2.append(non_zeros_acc2)

            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams.update({'font.size': 17})
            # 可视化：绘制准确率与一致性组之间的关系
            plt.figure(figsize=(6, 3))
            plt.plot(range(1, len(quantiles)), accuracies_7, marker='o', linestyle='-', color='#219ebc')
            plt.xticks(range(2, len(quantiles), 2),
                       [f"{(i + 1) / 8:.2f}" for i in range(1, len(quantiles), 2)])
            plt.xlabel('Consistency Quantile')
            plt.ylabel('Accuracy 7 class')
            plt.grid(True)
            plt.savefig(f'/home/data2t1/xxx/MSA/MTL/log/consistency/ACC7_{epoch}.pdf', dpi=300, bbox_inches='tight',
                        pad_inches=0)
            plt.show()

            # 可视化：绘制准确率与一致性组之间的关系
            plt.figure(figsize=(6, 3))
            plt.plot(range(1, len(quantiles)), accuracies_2, marker='o', linestyle='-', color='#669bbc')
            plt.xticks(range(2, len(quantiles), 2),
                       [f"{(i + 1) / 8:.2f}" for i in range(1, len(quantiles), 2)])
            plt.xlabel('Consistency Quantile')
            plt.ylabel('Accuracy 2 class')
            plt.grid(True)
            plt.savefig(f'/home/data2t1/xxx/MSA/MTL/log/consistency/ACC2_{epoch}.pdf', dpi=300, bbox_inches='tight',
                        pad_inches=0)
            plt.show()
        elif mode == 'stat':
            consistency = consistency.numpy()
            cl = cl.numpy()
            plt.figure(figsize=(5, 3))
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams.update({'font.size': 14})
            sns.kdeplot(consistency, fill=True, label="Consistency", color='#edae49')
            sns.kdeplot(cl, fill=True, label="Consistency w/o typicalness", color='#00798c')
            sns.kdeplot(kl, fill=True, label="KL", color='#d1495b')

            plt.xlabel("Value")
            plt.ylabel("Density")

            # 显示图例
            plt.legend(loc='best')
            plt.savefig(f'/home/data2t1/xxx/MSA/MTL/log/consistency/con_kl_stat_{epoch}.pdf', dpi=300,
                        bbox_inches='tight',
                        pad_inches=0)
            plt.show()
