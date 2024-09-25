import torch.nn as nn
import torch
from transformers import AutoModel
from losses import BCEFocalLoss, HMLC
from GCN import GraphConvolution
import math
import copy

class SelfAttention(nn.Module):
    def __init__(self, dropout=None):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values, mask=None):
        d = queries.shape[-1]
        scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(d)
        scores = scores.squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = self.softmax(scores)
        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights.unsqueeze(1), values).squeeze(1)
        return output, attention_weights


def get_attention(val_out, dep_embed, adj):
    batch_size, max_len, feat_dim = val_out.shape
    val_us = val_out.unsqueeze(dim=2)
    val_us = val_us.repeat(1, 1, max_len, 1)
    val_cat = torch.cat((val_us, dep_embed), -1)
    atten_expand = (val_cat.float() * val_cat.float().transpose(1, 2))
    attention_score = torch.sum(atten_expand, dim=-1)
    attention_score = attention_score / feat_dim ** 0.5
    # softmax
    exp_attention_score = torch.exp(attention_score)
    exp_attention_score = torch.mul(exp_attention_score.float(), adj.float())  # 对应点相乘
    # 归一化
    sum_attention_score = torch.sum(exp_attention_score, dim=-1).unsqueeze(dim=-1).repeat(1, 1, max_len)
    attention_score = torch.div(exp_attention_score, sum_attention_score + 1e-10)
    return attention_score


class ContrastBert(nn.Module):
    def __init__(self, config, similarity='jaccard'):
        super(ContrastBert, self).__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.pretrained_model_path)
        self.dropout = nn.Dropout(config.dropout_prob)
        if config.GCN:
            self.dep_type_embedding = nn.Embedding(config.dep_type_num, config.embedding_size, padding_idx=0)
            self.gcn_mode = GraphConvolution(config.embedding_size, config.embedding_size)
            self.gcn_layer = nn.ModuleList([copy.deepcopy(self.gcn_mode) for _ in range(config.num_gcn_layers)])
            self.fc_layer = nn.Linear(self.bert.config.to_dict()['hidden_size']*2, self.bert.config.to_dict()['hidden_size'])
        if config.loss_type == 'CE':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = BCEFocalLoss()
        self.contrastive_criterion = HMLC(config, hierarchy_path=config.hierarchy_path, similarity=similarity)
        self.fc = nn.Linear(self.bert.config.to_dict()['hidden_size'], config.num_labels)

    def forward(self, inputs, labels=None, labels_desc_ids=None, dep_type_matrix=None, hierarchy=None):
        raw_outputs = self.bert(**inputs)
        sequence_output = raw_outputs.last_hidden_state
        text_cls = sequence_output[:, 0, :]  # 取CLS的特征向量
        device = sequence_output.device
        if self.config.GCN and dep_type_matrix is not None:
            dep_type_embedding_outputs = self.dep_type_embedding(dep_type_matrix)
            dep_adj_matrix = torch.clamp(dep_type_matrix, 0, 1)
            words_output = sequence_output
            for i, gcn_layer_module in enumerate(self.gcn_layer):
                attention_score = get_attention(words_output, dep_type_embedding_outputs, dep_adj_matrix)
                words_output = gcn_layer_module(words_output, attention_score, dep_type_embedding_outputs)
            words_output = words_output.mean(dim=1)
            pooled_output = torch.cat([text_cls, words_output], dim=-1)
            pooled_output = self.fc_layer(pooled_output)
        else:
            pooled_output = text_cls

        logits = self.fc(self.dropout(pooled_output))
        if labels is not None:
            loss = self.criterion(logits, labels)
            if self.config.Contrast:
                contrastive_loss = self.contrastive_criterion(sequence_output.to(device), labels.to(device))
                loss += contrastive_loss
            if self.config.LabelEmbedding and labels_desc_ids is not None:
                labels_outputs = self.bert(**labels_desc_ids)
                labels_hiddens = labels_outputs.last_hidden_state
                labels_cls = labels_hiddens[:, 0, :]
                mask = inputs['attention_mask']
                attention = SelfAttention(dropout=0.2)
                output, attn = attention(sequence_output, labels_cls.unsqueeze(1).to(device), sequence_output, mask=mask)
                sample_logits = self.fc(self.dropout(output))
                sample_loss = self.criterion(sample_logits, labels)
                loss += 0.1 * sample_loss
            return logits, loss, pooled_output
        return logits, 0.0


def load_model(model, pre_model_path, device, strict=False):
    pretrained_dict = torch.load(pre_model_path, map_location=device)
    model_dict = model.state_dict()

    # 过滤掉不匹配的键
    filtered_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    not_loaded_keys = set(pretrained_dict.keys()) - set(filtered_pretrained_dict.keys())

    print(f"Keys not loaded: {not_loaded_keys}")

    # 更新现有的 state dict
    model_dict.update(filtered_pretrained_dict)

    # 加载新的 state dict
    model.load_state_dict(model_dict, strict=strict)



