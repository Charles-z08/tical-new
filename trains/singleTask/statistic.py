import torch.nn as nn
import torch

from trains.singleTask.model.CPCCLoss import CPCCLoss
from trains.singleTask.model.geom import poincare, pmath


def standardize(T):
    mean = torch.mean(T)
    std = torch.std(T)
    return (T - mean) / std


# 计算标签距离（欧几里得距离）
def label_distance(label_a, label_v, label_l):
    return torch.sqrt(torch.sum((label_a - label_v) ** 2) +
                      torch.sum((label_v - label_l) ** 2) +
                      torch.sum((label_l - label_a) ** 2))


# 计算典型性差异（欧几里得距离）
def typicality_distance(T_l, T_v, T_a):
    return torch.sqrt((T_l - T_v) ** 2 +
                      (T_v - T_a) ** 2 +
                      (T_a - T_l) ** 2)


# 计算典型性差异（绝对差）
def typicality_distance_abs(T_l, T_v, T_a):
    return torch.abs(T_l - T_v) + \
        torch.abs(T_v - T_a) + \
        torch.abs(T_a - T_l)


def calculate_consistency(T_l, T_v, T_a, min_labels_l, min_labels_v, min_labels_a, p=4, k=0.5, t=1/5):
    # 通过方差和密度系数求标签差异,密度系数p越大就越能区分相似情感预测和相悖情感预测
    mu = (min_labels_l + min_labels_v + min_labels_a)/3
    term_l = torch.abs(min_labels_l - mu)
    term_v = torch.abs(min_labels_v - mu)
    term_a = torch.abs(min_labels_a - mu)
    d = ((term_l + term_v + term_a)/3) ** p

    # 计算指数部分,k系数调控总体一致性权重
    distance_part = torch.exp(-d * k)
    # 计算典型性部分,温度系数调控
    typicality_part = (T_l * T_v * T_a) ** t

    return torch.sqrt(typicality_part * distance_part), distance_part


def poincare_distance(u, v, epsilon=1e-6):
    """
    计算一个批次的 Poincaré 距离。u 形状为 [16, 2300]，v 形状为 [2300,]
    :param u: 第一批次特征，形状为 [16, 2300]
    :param v: 第二个特征向量，形状为 [2300,]
    :param epsilon: 防止除零错误的很小常数
    :return: 每个样本与 v 的双曲距离，形状为 [16]
    """
    # 计算 u 和 v 的范数
    u_norm = torch.norm(u, dim=1)  # u 的每行的范数
    v_norm = torch.norm(v)  # v 的范数（只有一个值）

    # 防止除以零，确保 1 - 范数的平方大于一个小的阈值
    u_norm_sq = u_norm ** 2
    v_norm_sq = v_norm ** 2

    # 如果 u_norm_sq 或 v_norm_sq 接近 1，我们使用一个小的阈值来避免除以零
    u_norm_sq = torch.clamp(u_norm_sq, max=1 - epsilon)  # 保证 u 的范数平方不超过 1 - epsilon
    v_norm_sq = torch.clamp(v_norm_sq, max=1 - epsilon)  # 保证 v 的范数平方不超过 1 - epsilon

    # 计算每个样本与 v 之间的欧几里得距离
    euclidean_dists = torch.norm(u - v, dim=1)  # 计算每行之间的欧几里得距离

    # 计算 Poincaré 距离
    poincare_dists = torch.acosh(
        1 + 2 * (euclidean_dists ** 2) / ((1 - u_norm_sq) * (1 - v_norm_sq) + epsilon)
    )

    return poincare_dists


class Feature_statistic(nn.Module):
    def __init__(self, confidence_thresh=0.8, T_max=100, T_k=10, args=None):
        super(Feature_statistic, self).__init__()
        self.features_l_list = []
        self.features_v_list = []
        self.features_a_list = []
        self.labels_list = []
        self.confidence_thresh = confidence_thresh
        self.standard = 1
        self.T_max = T_max
        self.T_k = T_k
        self.args = args
        self.ccpc_loss = CPCCLoss(args)
        self.poincare = True

    def append(self, feature_l, feature_v, feature_a, labels, features_list_length=2000):
        if self.args.features_list_length > 0:
            features_list_length = self.args.features_list_length
        feature_l = feature_l.view(labels.size(0), -1).tolist()
        feature_v = feature_v.view(labels.size(0), -1).tolist()
        feature_a = feature_a.view(labels.size(0), -1).tolist()
        labels = labels.view(-1).tolist()

        self.features_l_list.extend(feature_l)
        self.features_v_list.extend(feature_v)
        self.features_a_list.extend(feature_a)
        self.labels_list.extend(labels)
        if len(self.labels_list) > features_list_length:
            self.features_l_list = self.features_l_list[-features_list_length:]
            self.features_v_list = self.features_v_list[-features_list_length:]
            self.features_a_list = self.features_a_list[-features_list_length:]
            self.labels_list = self.labels_list[-features_list_length:]

    def calculate_cpcc_loss(self, feature_l, feature_v, feature_a, min_labels_l, min_labels_v, min_labels_a):
        L_l = self.ccpc_loss(feature_l, min_labels_l)
        L_v = self.ccpc_loss(feature_v, min_labels_v)
        L_a = self.ccpc_loss(feature_a, min_labels_a)
        cpcc_loss = L_l + L_v + L_a
        return cpcc_loss

    def forward(self, features, labels, pred, confidence=None, modify_T=False, append=True):
        """
        Args:
            features: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size)
            pred: prediction matrix with shape (batch_size, num_class).
        """
        labels = labels.unsqueeze(1)
        pred = pred.unsqueeze(1)
        confidence = confidence.unsqueeze(1)
        T_l = torch.ones_like(labels)
        T_v = torch.ones_like(labels)
        T_a = torch.ones_like(labels)
        C = torch.ones_like(labels)
        C_label = torch.ones_like(labels)
        min_labels_l, min_labels_v, L_a = None, None, None
        cpcc_loss = None

        correctly_classified = torch.zeros_like(labels)
        correctly_classified[(labels == pred) & (confidence > self.confidence_thresh)] = 1

        feature_l, feature_v, feature_a = features
        # print(len(self.features_l_list))
        # TODO: update or append features_list
        if modify_T and len(self.features_l_list) != 0:
            # 将队列中的特征扩展为与当前样本相同的形状
            features_l_tensors = torch.tensor(self.features_l_list).to(self.args.device)
            features_v_tensors = torch.tensor(self.features_v_list).to(self.args.device)
            features_a_tensors = torch.tensor(self.features_a_list).to(self.args.device)
            labels_tensors = torch.tensor(self.labels_list).to(self.args.device)

            # 计算输入特征与队列中特征的欧式距离
            # 遍历队列中的每个特征并计算输入特征与队列中特征的欧几里得距离
            # 计算文本特征队列
            min_distances_l = torch.full((feature_l.size(0),), float('inf')).to(self.args.device)
            min_indices_l = torch.full((feature_l.size(0),), -1).to(self.args.device)
            for i in range(features_l_tensors.size(0)):  # 遍历历史队列中的每个特征
                if self.poincare:
                    dist = poincare_distance(feature_l, features_l_tensors[i])
                else:
                    dist = torch.norm(feature_l - features_l_tensors[i], p=2, dim=1)  # 计算输入特征与当前队列特征的欧几里得距离
                # 更新最小距离和最小距离索引
                mask = dist < min_distances_l  # 找出哪些距离比当前最小距离小
                min_distances_l[mask] = dist[mask]  # 更新最小距离
                min_indices_l[mask] = i  # 更新最小距离对应的索引
            # 取出文本模态最近的样本标签值
            min_labels_l = labels_tensors[min_indices_l]
            # 计算视觉特征队列
            min_distances_v = torch.full((feature_v.size(0),), float('inf')).to(self.args.device)
            min_indices_v = torch.full((feature_v.size(0),), -1).to(self.args.device)
            for i in range(features_l_tensors.size(0)):  # 遍历历史队列中的每个特征
                if self.poincare:
                    dist = poincare_distance(feature_v, features_v_tensors[i])
                else:
                    dist = torch.norm(feature_v - features_v_tensors[i], p=2, dim=1)  # 计算输入特征与当前队列特征的欧几里得距离
                # 更新最小距离和最小距离索引
                mask = dist < min_distances_v  # 找出哪些距离比当前最小距离小
                min_distances_v[mask] = dist[mask]  # 更新最小距离
                min_indices_v[mask] = i  # 更新最小距离对应的索引
            # 取出视觉模态最近的样本标签值
            min_labels_v = labels_tensors[min_indices_v]
            # 计算语音特征队列
            min_distances_a = torch.full((feature_a.size(0),), float('inf')).to(self.args.device)
            min_indices_a = torch.full((feature_a.size(0),), -1).to(self.args.device)
            for i in range(features_a_tensors.size(0)):  # 遍历历史队列中的每个特征
                if self.poincare:
                    dist = poincare_distance(feature_a, features_a_tensors[i])
                else:
                    dist = torch.norm(feature_a - features_a_tensors[i], p=2, dim=1)
                # 更新最小距离和最小距离索引
                mask = dist < min_distances_a  # 找出哪些距离比当前最小距离小
                min_distances_a[mask] = dist[mask]  # 更新最小距离
                min_indices_a[mask] = i  # 更新最小距离对应的索引
            # 取出语音模态最近的样本标签值
            min_labels_a = labels_tensors[min_indices_a]

            # 根据距离计算三个模态的典型性T
            T_l = torch.exp(-self.T_k * min_distances_l)
            T_v = torch.exp(-self.T_k * min_distances_v)
            T_a = torch.exp(-self.T_k * min_distances_a)
            # 如果启用了T_scale，则进行归一化
            if self.standard == 1:
                T_l = 1 - (min_distances_l - min_distances_l.min()) / (
                        min_distances_l.max() - min_distances_l.min() + 1e-8)
                T_v = 1 - (min_distances_v - min_distances_v.min()) / (
                        min_distances_v.max() - min_distances_v.min() + 1e-8)
                T_a = 1 - (min_distances_a - min_distances_a.min()) / (
                        min_distances_a.max() - min_distances_a.min() + 1e-8)

            # TODO: 计算一致性
            # 输入：模态典型性（T_l,T_v,T_a）
            #      对应的标签（min_labels_l,min_labels_v,min_labels_a）
            C, C_label = calculate_consistency(T_l, T_v, T_a, min_labels_l, min_labels_v, min_labels_a)

            # TODO: 计算CPCC损失
            # 输入：模态典型性（feature_l,feature_v,feature_a,min_labels_l,min_labels_v,min_labels_a）
            #      对应的标签（L_l, L_v, L_a）
            cpcc_loss = self.calculate_cpcc_loss(feature_l, feature_v, feature_a, min_labels_l, min_labels_v,
                                                 min_labels_a)

        if append and (correctly_classified.eq(1.0).any()).item():
            correctly_classified_mask_l = correctly_classified.repeat(1, feature_l.size(1))
            correctly_classified_mask_v = correctly_classified.repeat(1, feature_v.size(1))
            correctly_classified_mask_a = correctly_classified.repeat(1, feature_a.size(1))
            self.append(feature_l[correctly_classified_mask_l == 1], feature_v[correctly_classified_mask_v == 1],
                        feature_a[correctly_classified_mask_a == 1], labels[correctly_classified == 1])
        L_c = torch.norm(pmath.poincare_mean(poincare.expmap0(feature_l), dim=0, c=1.0)) + torch.norm(
            pmath.poincare_mean(poincare.expmap0(feature_v), dim=0, c=1.0)) + torch.norm(
            pmath.poincare_mean(poincare.expmap0(feature_a), dim=0, c=1.0))
        return T_l.detach(), T_v.detach(), T_a.detach(), C.detach(), C_label.detach(), cpcc_loss, L_c
