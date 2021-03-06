# coding=utf-8

import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from transformers import BertPreTrainedModel, BertModel, ElectraPreTrainedModel, \
    ElectraModel
from modeling.util import sequence_mask, split_bert_sequence


class AttentivePooling(nn.Module):

    def __init__(self, input_size):
        super(AttentivePooling, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, input, mask):
        bsz, length, size = input.size()
        score = self.fc(input.contiguous().view(-1, size)).view(bsz, length)
        score = score.masked_fill((~mask), -float('inf'))
        prob = F.softmax(score, dim=-1)
        attn = torch.bmm(prob.unsqueeze(1), input)
        return attn


class TriLinear(nn.Module):

    def __init__(self, input_size):
        super(TriLinear, self).__init__()
        self.w1 = nn.Parameter(torch.FloatTensor(1, input_size))
        self.w2 = nn.Parameter(torch.FloatTensor(1, input_size))
        self.w3 = nn.Parameter(torch.FloatTensor(1, input_size))

        self.init_param()

    def forward(self, query, key):
        ndim = query.dim()
        q_logit = F.linear(query, self.w1)
        k_logit = F.linear(key, self.w2)

        shape = [1] * (ndim - 1) + [-1]
        dot_k = self.w3.view(shape) * key
        dot_logit = torch.matmul(query, torch.transpose(dot_k, -1, -2))

        logit = q_logit + torch.transpose(k_logit, -1, -2) + dot_logit
        return logit

    def init_param(self):
        init.normal_(self.w1, 0., 0.02)
        init.normal_(self.w2, 0., 0.02)
        init.normal_(self.w3, 0., 0.02)


class Attention(nn.Module):

    def __init__(self, sim):
        super(Attention, self).__init__()
        self.sim = sim

    def forward(self, query, key, value, query_mask=None, key_mask=None):
        ndim = query.dim()
        logit = self.sim(query, key)
        if query_mask is not None and key_mask is not None:
            mask = query_mask.unsqueeze(ndim - 1) * key_mask.unsqueeze(ndim - 2)
            logit = logit.masked_fill((~mask), -float('inf'))

        attn_weight = F.softmax(logit, dim=-1)
        if query_mask is not None and key_mask is not None:
            attn_weight = attn_weight.masked_fill((~mask), 0.)

        attn = torch.matmul(attn_weight, value)

        kq_weight = F.softmax(logit, dim=1)
        if query_mask is not None and key_mask is not None:
            kq_weight = kq_weight.masked_fill((~mask), 0.)

        co_weight = torch.matmul(attn_weight,
                                 torch.transpose(kq_weight, -1, -2))
        co_attn = torch.matmul(co_weight, query)

        return (attn, attn_weight), (co_attn, co_weight)


# class OCN(ElectraPreTrainedModel):
class OCN(nn.Module):

    def __init__(self, config):
        # self.electra = ElectraModel(config)
        super().__init__()

        self.attn_sim = TriLinear(config.hidden_size)
        self.attention = Attention(sim=self.attn_sim)
        self.attn_fc = nn.Linear(config.hidden_size * 3, config.hidden_size,
                                 bias=True)

        self.opt_attn_sim = TriLinear(config.hidden_size)
        self.opt_attention = Attention(sim=self.opt_attn_sim)
        self.comp_fc = nn.Linear(config.hidden_size * 7, config.hidden_size,
                                 bias=True)

        self.query_attentive_pooling = AttentivePooling(
            input_size=config.hidden_size)
        self.gate_fc = nn.Linear(config.hidden_size * 2, config.hidden_size,
                                 bias=True)

        self.opt_selfattn_sim = TriLinear(config.hidden_size)
        self.opt_self_attention = Attention(sim=self.opt_selfattn_sim)
        self.opt_selfattn_fc = nn.Linear(config.hidden_size * 4,
                                         config.hidden_size, bias=True)

        self.score_fc = nn.Linear(config.hidden_size, 1)
        self.hidden_size = config.hidden_size
        # self.aggregation_layer = torch.nn.Linear(2 * config.hidden_size,
        #                                          config.hidden_size)
        # self.aggregation_activation = torch.nn.Tanh()

    def _custom_pad(self, tensor_list, max_dim, dim=-1):
        new_list = []
        for tensor in tensor_list:
            pad_dim = [0, 0] * (-1-dim) + [0, max_dim - tensor.shape[1]]
            new_tensor = F.pad(tensor, pad_dim, "constant", 0)
            new_list.append(new_tensor)
        return torch.cat(new_list, dim=0)

    def forward(self, last_layer, attention_mask, all_doc_final_pos, num_label):

        bsz = attention_mask.size(0) // num_label
        last_layer = last_layer.view(bsz, num_label, -1, self.hidden_size)
        attention_mask = attention_mask.view(bsz, num_label, -1)
        all_doc_final_pos = all_doc_final_pos.view(bsz, num_label)

        doc_enc_batch_list = []
        opt_enc_batch_list = []
        doc_mask_batch_list = []
        opt_mask_batch_list = []
        minus_inf = float("-inf")
        start_index = 0
        for index_batch in range(bsz):
            doc_enc_label_list = []
            opt_enc_label_list = []
            doc_mask_label_list = []
            opt_mask_label_list = []
            label_final_pos = all_doc_final_pos[index_batch]
            for index_label in range(num_label):
                doc_final_pos = label_final_pos[index_label]
                single_doc_enc = last_layer[index_batch, index_label,
                                 0: doc_final_pos, :]
                single_opt_enc = last_layer[index_batch, index_label,
                                 doc_final_pos:, :]

                single_doc_mask = attention_mask[index_batch, index_label,
                                  0: doc_final_pos] > 0
                single_opt_mask = attention_mask[index_batch, index_label,
                                  doc_final_pos:] > 0
                doc_enc_label_list.append(single_doc_enc)
                opt_enc_label_list.append(single_opt_enc)
                doc_mask_label_list.append(single_doc_mask)
                opt_mask_label_list.append(single_opt_mask)
            doc_enc_batch_list.append(
                rnn_utils.pad_sequence(doc_enc_label_list, batch_first=True,
                                       padding_value=0))
            opt_enc_batch_list.append(
                rnn_utils.pad_sequence(opt_enc_label_list, batch_first=True,
                                       padding_value=0))
            doc_mask_batch_list.append(
                rnn_utils.pad_sequence(doc_mask_label_list, batch_first=True,
                                       padding_value=False))
            opt_mask_batch_list.append(
                rnn_utils.pad_sequence(opt_mask_label_list, batch_first=True,
                                       padding_value=False))
        max_doc_length = max([doc.shape[1] for doc in doc_enc_batch_list])
        max_opt_length = max([opt.shape[1] for opt in opt_enc_batch_list])
        doc_enc = self._custom_pad(doc_enc_batch_list, max_doc_length, -2).view(
            bsz * num_label, -1, self.hidden_size)
        doc_mask = self._custom_pad(doc_mask_batch_list, max_doc_length, -1).view(
            bsz * num_label, -1)
        opt_enc = self._custom_pad(opt_enc_batch_list, max_opt_length, -2).view(
            bsz, num_label, -1, self.hidden_size)
        opt_mask = self._custom_pad(opt_mask_batch_list, max_opt_length, -1).view(
            bsz, num_label, -1)
        # doc_enc.masked_fill()

        # Option Comparison
        correlation_list = []
        for i in range(num_label):
            cur_opt = opt_enc[:, i, :, :]
            cur_mask = opt_mask[:, i, :]

            comp_info = []
            for j in range(num_label):
                if j == i:
                    continue

                tmp_opt = opt_enc[:, j, :, :]
                tmp_mask = opt_mask[:, j, :]

                (attn, _), _ = self.opt_attention(cur_opt, tmp_opt, tmp_opt,
                                                  cur_mask, tmp_mask)
                comp_info.append(cur_opt * attn)
                comp_info.append(cur_opt - attn)

            correlation = torch.tanh(
                self.comp_fc(torch.cat([cur_opt] + comp_info, dim=-1)))
            correlation_list.append(correlation)

        correlation_list = [correlation.unsqueeze(1) for correlation in
                            correlation_list]
        opt_correlation = torch.cat(correlation_list, dim=1)

        opt_mask = opt_mask.view(bsz * num_label, -1)
        opt_enc = opt_enc.view(bsz * num_label, -1, self.hidden_size)
        _, _, dim2, dim3 = opt_correlation.shape
        opt_correlation = opt_correlation.view(-1, dim2, dim3)
        features = torch.cat([opt_enc,
                              opt_correlation], dim=-1)
        gate = torch.sigmoid(self.gate_fc(features))
        option = opt_enc * gate + opt_correlation * (1.0 - gate)
        # option = self.aggregation_layer(concat_encodings)

        (attn, _), (coattn, _) = self.attention(option, doc_enc, doc_enc,
                                                opt_mask, doc_mask)
        fusion = self.attn_fc(torch.cat((option, attn, coattn), -1))
        fusion = F.tanh(fusion)

        (attn, _), _ = self.opt_self_attention(fusion, fusion, fusion, opt_mask,
                                               opt_mask)
        fusion = self.opt_selfattn_fc(
            torch.cat((fusion, attn, fusion * attn, fusion - attn), -1))
        fusion = F.relu(fusion)

        fusion = fusion.masked_fill(
            (~opt_mask).unsqueeze(-1).expand(bsz * num_label, -1,
                                             self.hidden_size),
            -float('inf'))
        # 0)
        fusion, _ = fusion.max(dim=1)
        # 4, 768
        return fusion

        # logits = self.score_fc(fusion).view(bsz, num_label)
        #
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     loss = loss_fct(logits, labels)
        #     return loss, logits
        # else:
        #     return logits
