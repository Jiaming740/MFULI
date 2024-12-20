import torch.nn as nn
import torch
from transformers import AutoModel
from losses import BCEFocalLoss, HMLC
import math

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
    def __init__(self, config, similarity='hierarchical_jaccard'):
        super(ContrastBert, self).__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.pretrained_model_path)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.criterion = nn.BCEWithLogitsLoss() if config.loss_type == 'CE' else BCEFocalLoss()
        self.contrastive_criterion = HMLC(config, hierarchy_path=config.hierarchy_path, similarity=similarity)
        self.fc = nn.Linear(self.bert.config.hidden_size, config.num_labels)

    def forward(self, inputs, labels=None, labels_desc_ids=None, hierarchy=None):
        device = inputs['input_ids'].device
        raw_outputs = self.bert(**inputs)
        sequence_output = raw_outputs.last_hidden_state
        text_cls = sequence_output[:, 0, :]

        logits = self.fc(self.dropout(text_cls))

        loss = 0.0
        if labels is not None:
            labels = labels.to(device)
            loss = self.criterion(logits, labels)
            if self.config.Contrast:
                contrastive_loss = self.contrastive_criterion(sequence_output, labels)
                loss += contrastive_loss
            if self.config.LabelEmbedding and labels_desc_ids is not None:
                labels_outputs = self.bert(**labels_desc_ids)
                labels_cls = labels_outputs.last_hidden_state[:, 0, :]
                mask = inputs['attention_mask']
                attention = SelfAttention(dropout=0.2)
                output, _ = attention(sequence_output, labels_cls.unsqueeze(1), sequence_output, mask=mask)
                sample_logits = self.fc(self.dropout(output))
                sample_loss = self.criterion(sample_logits, labels)
                loss += 0.1 * sample_loss
        return logits, loss, text_cls


def load_model(model, pre_model_path, device, strict=False):
    pretrained_dict = torch.load(pre_model_path, map_location=device)
    model_dict = model.state_dict()

    filtered_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    not_loaded_keys = set(pretrained_dict.keys()) - set(filtered_pretrained_dict.keys())

    if not_loaded_keys:
        print(f"Keys not loaded: {not_loaded_keys}")

    model_dict.update(filtered_pretrained_dict)
    model.load_state_dict(model_dict, strict=strict)



