import torch
import torch.nn as nn
import pandas as pd
from functools import lru_cache


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
        self.label_to_index = {label: idx for idx, label in enumerate(self.all_labels)}

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

    @lru_cache(maxsize=None)
    def get_path_to_root(self, label):
        path = []
        while label is not None:
            path.append(label)
            label = self.hierarchy[label]['parent']
        return path

    def labels_to_one_hot(self, labels_batch):
        batch_size = len(labels_batch)
        one_hot = torch.zeros((batch_size, len(self.hierarchy)), dtype=torch.float)
        for i, label_set in enumerate(labels_batch):
            for label in label_set:
                if label in self.label_to_index:
                    one_hot[i, self.label_to_index[label]] = 1
        return one_hot

    def hierarchical_jaccard_batch(self, one_hot_labels):
        batch_size = one_hot_labels.size(0)

        hierarchical_paths = [set() for _ in range(batch_size)]
        for i in range(batch_size):
            labels = [self.all_labels[idx] for idx in torch.nonzero(one_hot_labels[i], as_tuple=True)[0]]
            for label in labels:
                hierarchical_paths[i].update(self.get_path_to_root(label))

        intersection_matrix = torch.zeros((batch_size, batch_size), dtype=torch.float)
        union_matrix = torch.zeros((batch_size, batch_size), dtype=torch.float)

        for i in range(batch_size):
            for j in range(i, batch_size):
                set1 = hierarchical_paths[i]
                set2 = hierarchical_paths[j]
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                similarity = intersection / union if union != 0 else 0.0
                intersection_matrix[i, j] = similarity
                intersection_matrix[j, i] = similarity

        return intersection_matrix

    def forward(self, features, labels):
        device = features.device

        one_hot_labels = self.labels_to_one_hot(labels).to(device)

        mask = self.hierarchical_jaccard_batch(one_hot_labels).to(device)

        cumulative_loss = self.sup_con_loss(features, mask=mask)
        return cumulative_loss


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None):
        if len(features.shape) < 2:
            raise ValueError('`features` needs to be [bsz, ...],'
                             'at least 2 dimensions are required')
        if len(features.shape) > 2:
            features = features.view(features.shape[0], -1)

        device = features.device
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
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        one = torch.ones_like(mask).to(device)
        mask_labels = torch.where(mask > 0, one, mask)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_labels.sum(1) + 1e-12)

        loss = -mean_log_prob_pos.mean()

        return loss


config = type('Config', (object,), {'temp': 0.07})
hierarchy_path = "./data/Hierarchical_label.csv"
model = HMLC(config, hierarchy_path, similarity='hierarchical_jaccard')


