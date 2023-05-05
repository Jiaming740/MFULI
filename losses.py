#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import torch
import torch.nn as nn


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=1.0, alpha=0.75, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict)  # sigmoide获取概率
        # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class HMLC(nn.Module):
    def __init__(self, config, layer_penalty=None):
        super(HMLC, self).__init__()
        self.temperature = config.temp
        self.config = config
        if not layer_penalty:
            self.layer_penalty = self.pow_2  # 2的n次幂
        else:
            self.layer_penalty = layer_penalty
        self.sup_con_loss = SupConLoss(self.temperature)

    def pow_2(self, value):
        return torch.pow(2, value)

    def forward(self, features, labels):
        labels = labels.cpu()
        features = features.cpu()
        mask = torch.matmul(labels, labels.T)
        mask = mask.type(torch.float)
        size = labels.shape[0]
        labels_count = labels.sum(-1)
        for i in range(size):
            for j in range(size):
                t_max = torch.max(labels_count[i], labels_count[j])
                mask[i][j] = mask[i][j]/t_max
        # mask = mask ^ torch.diag_embed(torch.diag(mask))  # 对角线定义为0
        cumulative_loss = self.sup_con_loss(features, mask=mask)
        cumulative_loss = (1/self.layer_penalty(torch.tensor(self.config.alpha).type(torch.float))) * cumulative_loss
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

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float()
        else:
            mask = mask.float()
        anchor_feature = features
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, features.T),
            self.temperature)
        # print('anchor_dot_contrast:',anchor_dot_contrast)
        # anchor_dot_contrast = anchor_dot_contrast - torch.diag_embed(torch.diag(anchor_dot_contrast))
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1),
            0
        )
        mask = mask * logits_mask  # 把mask对角线的数据设置为0
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask  # 计算除了对角线所有的概率
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        # print('log_prob:',log_prob)
        # compute mean of log-likelihood over positive,sum(1)每列求和
        one = torch.ones_like(mask)
        mask_labels = torch.where(mask > 0, one, mask)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_labels.sum(1)+1e-12)
        loss = -mean_log_prob_pos.mean()
        # print(loss)
        return loss
