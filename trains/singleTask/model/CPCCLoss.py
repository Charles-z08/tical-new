import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trains.singleTask.model.geom.poincare as poincare
import trains.singleTask.model.geom.pmath as pmath
import trains.singleTask.model.geom.nn as hypnn
from trains.singleTask.model.geom.pmath import dist_matrix

class CPCCLoss(nn.Module):
    def __init__(self, args):
        super(CPCCLoss, self).__init__()
        self.tree_dist = {
            ('root', 'negative'): 1,
            ('root', 'neutral'): 1,
            ('root', 'positive'): 1,
            ('root', 'extreme_negative'): 2,
            ('root', 'very_negative'): 2,
            ('root', 'neutral_negative'): 2,
            ('root', 'slightly_positive'): 2,
            ('root', 'moderately_positive'): 2,
            ('root', 'extreme_positive'): 2,
            ('negative', 'neutral'): 2,
            ('negative', 'positive'): 2,
            ('negative', 'extreme_negative'): 1,
            ('negative', 'very_negative'): 1,
            ('negative', 'neutral_negative'): 1,
            ('negative', 'slightly_positive'): 3,
            ('negative', 'moderately_positive'): 3,
            ('negative', 'extreme_positive'): 3,
            ('neutral', 'positive'): 2,
            ('neutral', 'extreme_negative'): 3,
            ('neutral', 'very_negative'): 3,
            ('neutral', 'neutral_negative'): 3,
            ('neutral', 'slightly_positive'): 3,
            ('neutral', 'moderately_positive'): 3,
            ('neutral', 'extreme_positive'): 3,
            ('positive', 'extreme_negative'): 3,
            ('positive', 'very_negative'): 3,
            ('positive', 'neutral_negative'): 3,
            ('positive', 'slightly_positive'): 1,
            ('positive', 'moderately_positive'): 1,
            ('positive', 'extreme_positive'): 1,
            ('extreme_negative', 'very_negative'): 2,
            ('extreme_negative', 'neutral_negative'): 2,
            ('extreme_negative', 'slightly_positive'): 4,
            ('extreme_negative', 'moderately_positive'): 4,
            ('extreme_negative', 'extreme_positive'): 4,
            ('very_negative', 'neutral_negative'): 2,
            ('very_negative', 'slightly_positive'): 4,
            ('very_negative', 'moderately_positive'): 4,
            ('very_negative', 'extreme_positive'): 4,
            ('neutral_negative', 'slightly_positive'): 4,
            ('neutral_negative', 'moderately_positive'): 4,
            ('neutral_negative', 'extreme_positive'): 4,
            ('slightly_positive', 'moderately_positive'): 2,
            ('slightly_positive', 'extreme_positive'): 2,
            ('moderately_positive', 'extreme_positive'): 2
        }

        self.label_map_str = np.array([
            ['extreme_negative', 'negative', 'root'],  # [-3, -2.5)
            ['very_negative', 'negative', 'root'],  # [-2.5, -1.5)
            ['neutral_negative', 'negative', 'root'],  # [-1.5, -0.5)
            ['', 'neutral', 'root'],  # [-0.5, 0.5) (中性情感)
            ['slightly_positive', 'positive', 'root'],  # [0.5, 1.5)
            ['moderately_positive', 'positive', 'root'],  # [1.5, 2.5)
            ['extreme_positive', 'positive', 'root']  # [2.5, 3)
        ])
        self.label_map_int, self.str2int, self.int2str = self.map_strings_to_integers(self.label_map_str)
        self.empty_int = self.str2int.get('', -1)  # 用于填充的空字符串
        self.tree_depth = len(self.label_map_str[0])  # 标签的深度
        self.distance_type = 'poincare'  # 距离类型，可能是'Poincare'等
        if self.distance_type == 'poincare_exp_c':
            self.to_hyperbolic = hypnn.ToPoincare(c=args.hyp_c, ball_dim=args.poincare_head_dim, riemannian=True,
                                                  clip_r=args.clip_r, train_c=args.train_c)
            self.dist_f = lambda x, y: dist_matrix(x, y, c=args.hyp_c)

    def forward(self, representations, targets_fine):
        # 映射到分类值
        emotion_labels = torch.tensor([self.regression_to_cls(val.item()) for val in targets_fine])
        label_map = torch.tensor(self.label_map_int, device=targets_fine.device)
        # 假设树有d层
        targets = label_map[emotion_labels]
        all_unique_int = [torch.unique(targets[:, col][targets[:, col] != self.empty_int]) for col in
                          range(targets.shape[1])]  # 获取每一层的唯一节点
        all_unique_str = []  # 展平后的所有唯一字符串节点
        target_mean_list = []  # 保存每一层的均值
        representations_poincare = None
        if self.distance_type == 'poincare_mean':
            representations_poincare = poincare.expmap0(representations)

        # 遍历每一层，计算每一层的均值
        for col, unique_values in enumerate(all_unique_int):
            for val in unique_values:
                if self.distance_type == 'poincare_mean':
                    column_mean = pmath.poincare_mean(torch.index_select(representations_poincare, 0,
                                                                         (targets[:, col] == val).nonzero(
                                                                             as_tuple=True)[0]), dim=0, c=1.0)
                else:
                    column_mean = torch.mean(
                        torch.index_select(representations, 0, (targets[:, col] == val).nonzero(as_tuple=True)[0]), 0)
                target_mean_list.append(column_mean)
                all_unique_str.append(self.int2str[val.item()])

        sorted_sums = torch.stack(target_mean_list, 0)

        # 根据距离类型计算节点间的距离
        if self.distance_type == 'l2':
            pairwise_dist = F.pdist(sorted_sums, p=2.0)  # 欧几里得距离
        elif self.distance_type == 'nl2':
            all_norms = torch.norm(sorted_sums, dim=1, p=2).unsqueeze(-1)
            pairwise_dist = F.pdist(all_norms, p=2.0)
        elif self.distance_type == 'l1':
            pairwise_dist = F.pdist(sorted_sums, p=1.0)
        elif self.distance_type == 'poincare':
            epsilon = 1e-5
            all_norms = torch.norm(sorted_sums, dim=1, p=2).unsqueeze(-1)
            normalized_sorted_sums = sorted_sums * (1 - epsilon) / all_norms
            all_normalized_norms = torch.norm(normalized_sorted_sums, dim=1, p=2)
            condensed_idx = torch.triu_indices(len(all_unique_str), len(all_unique_str), offset=1,
                                               device=sorted_sums.device)
            numerator_square = torch.sum((normalized_sorted_sums[None, :] - normalized_sorted_sums[:, None]) ** 2, -1)
            numerator = numerator_square[condensed_idx[0], condensed_idx[1]]
            denominator_square = ((1 - all_normalized_norms ** 2).reshape(-1, 1)) @ (
                    1 - all_normalized_norms ** 2).reshape(1, -1)
            denominator = denominator_square[condensed_idx[0], condensed_idx[1]]
            pairwise_dist = torch.acosh(1 + 2 * (numerator / denominator + epsilon))
        elif self.distance_type == 'poincare_exp':
            sorted_sums_exp = poincare.expmap0(sorted_sums)
            condensed_idx = torch.triu_indices(len(all_unique_str), len(all_unique_str), offset=1,
                                               device=sorted_sums.device)
            pairwise_dists_poincare_matrix = poincare.pairwise_distance(sorted_sums_exp)
            pairwise_dist = pairwise_dists_poincare_matrix[condensed_idx[0], condensed_idx[1]]
        elif self.distance_type == 'poincare_exp_c':
            sorted_sums_exp = self.to_hyperbolic(sorted_sums)
            condensed_idx = torch.triu_indices(len(all_unique_str), len(all_unique_str), offset=1,
                                               device=sorted_sums.device)
            pairwise_dists_poincare_matrix = self.dist_f(sorted_sums_exp, sorted_sums_exp)
            pairwise_dist = pairwise_dists_poincare_matrix[condensed_idx[0], condensed_idx[1]]

        # 计算树的真实距离
        tree_pairwise_dist = self.dT(all_unique_str, pairwise_dist.device)

        # 最小化CPCC损失
        res = 1 - torch.corrcoef(torch.stack([pairwise_dist, tree_pairwise_dist], 0))[0, 1]

        if torch.isnan(res):
            return torch.tensor(1, device=pairwise_dist.device)
        else:
            return res

    def map_strings_to_integers(self, string_array):
        string_to_int = {}
        int_to_string = {}
        current_id = 0

        int_array = np.zeros(string_array.shape, dtype=int)

        for i in range(string_array.shape[0]):
            for j in range(string_array.shape[1]):
                string = string_array[i, j]
                if string not in string_to_int:
                    string_to_int[string] = current_id
                    int_to_string[current_id] = string
                    current_id += 1
                int_array[i, j] = string_to_int[string]

        return int_array, string_to_int, int_to_string

    def dT(self, all_node, device):
        tree_pairwise_dist = []
        for i in range(len(all_node)):
            for j in range(i + 1, len(all_node)):
                if self.tree_dist.get((all_node[i], all_node[j])) is not None:
                    tree_pairwise_dist.append(self.tree_dist[(all_node[i], all_node[j])])
                else:
                    tree_pairwise_dist.append(self.tree_dist[(all_node[j], all_node[i])])
        return torch.tensor(tree_pairwise_dist, device=device)

    # 根据回归值生成标签映射
    def regression_to_cls(self, value):
        if value < -2.5:
            return 0  # 'extreme_negative'  [-3, -2.5)
        elif -2.5 <= value < -1.5:
            return 1  # 'very_negative'    [-2.5, -1.5)
        elif -1.5 <= value < -0.5:
            return 2  # 'neutral_negative'  [-1.5, -0.5)
        elif -0.5 <= value < 0.5:
            return 3  # 'neutral'           [-0.5, 0.5) (中性情感)
        elif 0.5 <= value < 1.5:
            return 4  # 'slightly_positive' [0.5, 1.5)
        elif 1.5 <= value < 2.5:
            return 5  # 'positive'          [1.5, 2.5)
        else:
            return 6  # 'extreme_positive'  [2.5, 3]


if __name__ == '__main__':
    args = None
    c = CPCCLoss(args)
    feas = torch.randn(16, 2300)
    labels = torch.Tensor([-2, -1, 0, 1, 2, 3])
    res = c(feas, labels)
