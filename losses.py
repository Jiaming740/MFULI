# # -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import pandas as pd


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=1.0, alpha=0.75, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict)
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class HMLC(nn.Module):
    def __init__(self, config, hierarchy_path, similarity='hierarchical_jaccard'):
        super(HMLC, self).__init__()
        self.temperature = config.temp
        self.config = config
        self.similarity_function = similarity
        self.hierarchy = self.load_hierarchy(hierarchy_path)
        self.sup_con_loss = SupConLoss(self.temperature)
        self.max_depth = max([self.hierarchy[label]['depth'] for label in self.hierarchy])
        self.all_labels = list(self.hierarchy.keys())

    @staticmethod
    def load_hierarchy(hierarchy_path):
        hierarchy_df = pd.read_csv(hierarchy_path, encoding='utf-8')
        hierarchy = {}

        def add_node(section, class_, subclass):
            if section not in hierarchy:
                hierarchy[section] = {'parent': None, 'depth': 1}
            if class_ not in hierarchy:
                hierarchy[class_] = {'parent': section, 'depth': 2}
            if subclass not in hierarchy:
                hierarchy[subclass] = {'parent': class_, 'depth': 3}

        for _, row in hierarchy_df.iterrows():
            section = row['Section']
            class_ = row['Class']
            subclass = row['Subclass']
            add_node(section, class_, subclass)
        return hierarchy

    def labels_to_one_hot(self, label_set):
        label_to_index = {label: idx for idx, label in enumerate(self.hierarchy.keys())}
        one_hot = torch.zeros(len(self.hierarchy), dtype=torch.float)
        for label in label_set:
            if label in label_to_index:
                one_hot[label_to_index[label]] = 1
        return one_hot

    def get_path_to_root(self, label):
        path = []
        while label is not None:
            path.append(label)
            label = self.hierarchy[label]['parent']
        return path

    def hierarchical_jaccard(self, set1, set2):
        set1_labels = [list(self.hierarchy.keys())[i] for i in range(len(set1)) if set1[i] == 1]
        set2_labels = [list(self.hierarchy.keys())[i] for i in range(len(set2)) if set2[i] == 1]

        def get_all_hierarchy_labels(label_set):
            all_labels = set()
            for label in label_set:
                path = self.get_path_to_root(label)
                all_labels.update(path)
            return all_labels

        set1_hierarchy_labels = get_all_hierarchy_labels(set1_labels)
        set2_hierarchy_labels = get_all_hierarchy_labels(set2_labels)

        intersection = len(set1_hierarchy_labels.intersection(set2_hierarchy_labels))
        union = len(set1_hierarchy_labels.union(set2_hierarchy_labels))

        if union == 0:
            return torch.tensor(0.0)
        else:
            return torch.tensor(intersection / union)

    def forward(self, features, labels):
        device = features.device
        features = features.to(device)
        size = len(labels)
        mask = torch.zeros((size, size), dtype=torch.float).to(device)

        one_hot_labels = torch.stack([self.labels_to_one_hot(label_set).to(device) for label_set in labels])

        for i in range(size):
            for j in range(size):
                mask[i][j] = self.hierarchical_jaccard(one_hot_labels[i], one_hot_labels[j])

        cumulative_loss = self.sup_con_loss(features, mask=mask)
        return cumulative_loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None):

        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, ...],'
                             'at least 2 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], -1)

        device = features.device  # 获取 features 所在设备
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1).to(device)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        anchor_feature = features
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, features.T),
            self.temperature).to(device)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.scatter(
            torch.ones_like(mask).to(device),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask  # 把mask对角线的数据设置为0
        exp_logits = torch.exp(logits) * logits_mask  # 计算除了对角线所有的概率
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        one = torch.ones_like(mask).to(device)
        mask_labels = torch.where(mask > 0, one, mask)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_labels.sum(1)+1e-12)
        loss = -mean_log_prob_pos.mean()

        return loss


# Test
config = type('Config', (object,), {'temp': 0.07})

hierarchy_path = "./data/Hierarchical_label.csv"
model = HMLC(config, hierarchy_path, similarity='hierarchical_jaccard')

