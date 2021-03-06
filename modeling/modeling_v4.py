from transformers import (ElectraModel, ElectraPreTrainedModel, BertModel, BertPreTrainedModel, RobertaModel)
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, Conv1d
import torch.nn.functional as F
import math
import torch.nn.utils.rnn as rnn_utils


BertLayerNorm = torch.nn.LayerNorm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ACT2FN = {"gelu": F.gelu, "relu": F.relu}

class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # (batch_size, seq_len, all_head_size) -> (batch_size, num_attention_heads, seq_len, attention_head_size)

    def forward(self, input_ids_a, input_ids_b, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(input_ids_a)
        mixed_key_layer = self.key(input_ids_b)
        mixed_value_layer = self.value(input_ids_b)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # (batch_size * num_choice, num_attention_heads, seq_len, attention_head_size) -> (batch_size * num_choice, seq_len, num_attention_heads, attention_head_size)

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        
        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids_a + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)


class GRUWithPadding(nn.Module):
    def __init__(self, config, num_rnn = 1):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = num_rnn
        self.biGRU = nn.GRU(config.hidden_size, config.hidden_size, self.num_layers, batch_first = True, bidirectional = True)

    def forward(self, inputs):
        batch_size = len(inputs)
        sorted_inputs = sorted(enumerate(inputs), key=lambda x: x[1].size(0), reverse = True)
        idx_inputs = [i[0] for i in sorted_inputs]
        inputs = [i[1] for i in sorted_inputs]
        inputs_lengths = [len(i[1]) for i in sorted_inputs]

        inputs = rnn_utils.pad_sequence(inputs, batch_first = True)
        inputs = rnn_utils.pack_padded_sequence(inputs, inputs_lengths, batch_first = True) #(batch_size, seq_len, hidden_size)

        h0 = torch.rand(2 * self.num_layers, batch_size, self.hidden_size).to(inputs.data.device) # (2, batch_size, hidden_size)
        # self.biGRU.flatten_parameters()
        out, _ = self.biGRU(inputs, h0) # (batch_size, 2, hidden_size )
        out_pad, out_len = rnn_utils.pad_packed_sequence(out, batch_first = True) # (batch_size, seq_len, 2 * hidden_size)

        _, idx2 = torch.sort(torch.tensor(idx_inputs))
        idx2 = idx2.to(out_pad.device)
        output = torch.index_select(out_pad, 0, idx2)
        out_len = out_len.to(out_pad.device)
        out_len = torch.index_select(out_len, 0, idx2)

        out_idx = (out_len - 1).unsqueeze(1).unsqueeze(2).repeat([1,1,self.hidden_size * 2])
        output = torch.gather(output, 1, out_idx).squeeze(1)

        return output 


class FuseLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear1 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.linear3 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.gate = nn.Sigmoid()

    def forward(self, orig, input1, input2):
        out1 = self.activation(self.linear1(torch.cat([orig, input1, orig - input1, orig * input1], dim = -1)))
        out2 = self.activation(self.linear2(torch.cat([orig, input2, orig - input2, orig * input2], dim = -1)))
        fuse_prob = self.gate(self.linear3(torch.cat([out1, out2], dim = -1)))

        return fuse_prob * input1 + (1 - fuse_prob) * input2


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
        torch.nn.init.normal_(self.w1, 0., 0.02)
        torch.nn.init.normal_(self.w2, 0., 0.02)
        torch.nn.init.normal_(self.w3, 0., 0.02)


class OptAttention(nn.Module):

    def __init__(self, sim):
        super(OptAttention, self).__init__()
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


class ElectraForMultipleChoicePlusV4(ElectraPreTrainedModel):
    def __init__(self, config, num_rnn = 1, num_decoupling = 1,):
        super().__init__(config)
        print("Using electra")
        self.electra = ElectraModel(config)
        self.num_decoupling = num_decoupling

        # ocn
        self.hidden_size = config.hidden_size
        self.opt_attn_sim = TriLinear(config.hidden_size)
        self.opt_attention = OptAttention(sim=self.opt_attn_sim)
        self.comp_fc = nn.Linear(config.hidden_size * 7, config.hidden_size,
                                 bias=True)
        self.gate_fc = nn.Linear(config.hidden_size * 2, config.hidden_size,
                                 bias=True)

        self.localMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])
        self.globalMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])
        self.SASelfMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])
        self.SACrossMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])

        self.fuse1 = FuseLayer(config)
        self.fuse2 = FuseLayer(config)
        
        self.gru1 = GRUWithPadding(config, num_rnn)
        self.gru2 = GRUWithPadding(config, num_rnn)

        self.pooler = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.pooler_2 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.pooler_activation_2 = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.classifier2 = nn.Linear(config.hidden_size, 2)
        # self.merger = torch.nn.Linear(3*config.hidden_size, config.hidden_size)
        # self.merger_activation = nn.Tanh()

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        sep_pos = None,
        position_ids=None,
        turn_ids = None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        doc_final_pos=None
    ):
        bsz = input_ids.shape[0]
        num_labels = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None 
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # (batch_size * choice, seq_len)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        sep_pos = sep_pos.view(-1, sep_pos.size(-1)) if sep_pos is not None else None
        turn_ids = turn_ids.view(-1, turn_ids.size(-1)) if turn_ids is not None else None

        turn_ids = turn_ids.unsqueeze(-1).repeat([1,1,turn_ids.size(1)])
        outputs = self.electra(input_ids, token_type_ids, attention_mask)
        last_layer = outputs['last_hidden_state']
        # option comparison
        doc_final_pos_ = doc_final_pos.view(-1)
        opt_enc = []
        opt_mask = []
        for idx in range(last_layer.size(0)):
            # print(doc_final_pos.size())
            single_doc_final_pos = doc_final_pos_[idx]
            single_opt_enc = last_layer[idx, single_doc_final_pos:, :]
            single_opt_mask = attention_mask[idx, single_doc_final_pos:] > 0
            opt_enc.append(single_opt_enc)
            opt_mask.append(single_opt_mask)
        opt_enc_t = rnn_utils.pad_sequence(opt_enc, batch_first=True, padding_value=0)
        opt_mask_t = rnn_utils.pad_sequence(opt_mask, batch_first=True, padding_value=0)

        correlation_list = []
        opt_enc_t = opt_enc_t.view(bsz, num_labels, -1, last_layer.shape[-1])
        opt_mask_t = opt_mask_t.view(bsz, num_labels, -1)

        for i in range(num_labels):
            cur_opt = opt_enc_t[:, i, :, :]
            cur_mask = opt_mask_t[:, i, :]

            comp_info = []
            for j in range(num_labels):
                if j == i:
                    continue

                tmp_opt = opt_enc_t[:, j, :, :]
                tmp_mask = opt_mask_t[:, j, :]

                (attn, _), _ = self.opt_attention(cur_opt, tmp_opt, tmp_opt,
                                                  cur_mask, tmp_mask)
                comp_info.append(cur_opt * attn)
                comp_info.append(cur_opt - attn)

            correlation = torch.tanh(self.comp_fc(torch.cat([cur_opt] + comp_info, dim=-1)))
            correlation_list.append(correlation)

        correlation_list = [correlation.unsqueeze(1) for correlation in
                            correlation_list]
        opt_correlation = torch.cat(correlation_list, dim=1)

        opt_mask_t = opt_mask_t.view(bsz * num_labels, -1)
        opt_enc_t = opt_enc_t.view(bsz * num_labels, -1, self.hidden_size)

        _, _, dim2, dim3 = opt_correlation.shape
        opt_correlation = opt_correlation.view(-1, dim2, dim3)
        features = torch.cat([opt_enc_t, opt_correlation], dim=-1)
        gate = torch.sigmoid(self.gate_fc(features))
        option = opt_enc_t * gate + opt_correlation * (1.0 - gate)
        
        # replace option
        max_length = last_layer.size(1)
        for idx in range(last_layer.size(0)):
            single_doc_final_pos = doc_final_pos_[idx]
            single_doc_mask = torch.ones(last_layer[idx].shape).to(device)
            single_doc_mask[:single_doc_final_pos, :] = 0
            single_doc_mask = single_doc_mask.bool()
            single_option_mask = ~single_doc_mask
            single_option = option[idx]
            padded_option = F.pad(single_option, (0, 0, single_doc_final_pos, max_length - single_option.shape[0] - single_doc_final_pos), "constant", 0)
            # padded_option = F.pad(single_option, (0, 0, max_length - single_option.shape[0], 0), "constant", 0)
            # assert(padded_option.shape[0] == last_layer[idx].shape[0])
            masked_option = padded_option.masked_fill(single_option_mask, 0)
            # masked_doc = last_layer[idx].masked_fill(single_doc_mask, 0)
            last_layer[idx] =  masked_option + last_layer[idx]

        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        
        #print("size of sequence_output:", sequence_output.size())
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # (batch_size * num_choice, 1, 1, seq_len)
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        

        # decoupling & fusing
        local_mask = torch.zeros_like(attention_mask, dtype = self.dtype)
        local_mask = local_mask.repeat((1,1,attention_mask.size(-1), 1)) #(batch_size * num_choice, 1, seq_len, seq_len)
        global_mask = torch.zeros_like(local_mask, dtype = self.dtype)
        sa_self_mask = torch.zeros_like(local_mask, dtype = self.dtype)
        sa_cross_mask = torch.zeros_like(local_mask, dtype = self.dtype)

        last_seps = []

        for i in range(input_ids.size(0)):
            last_sep = 1

            while last_sep < len(sep_pos[i]) and sep_pos[i][last_sep] != 0: 
                last_sep += 1
            
            last_sep = last_sep - 1
            last_seps.append(last_sep)

            local_mask[i, 0, turn_ids[i] == turn_ids[i].T] = 1.0
            local_mask[i, 0, :, (sep_pos[i][last_sep] + 1):] = 0

            sa_self_mask[i, 0, (turn_ids[i] % 2) == (turn_ids[i].T % 2)] = 1.0
            sa_self_mask[i, 0, :, (sep_pos[i][last_sep] + 1):] = 0

            global_mask[i, 0, :, :(sep_pos[i][last_sep] + 1)] = 1.0 - local_mask[i, 0, :, :(sep_pos[i][last_sep] + 1)]
            sa_cross_mask[i, 0, :, :(sep_pos[i][last_sep] + 1)] = 1.0 - sa_self_mask[i, 0, :, :(sep_pos[i][last_sep] + 1)]

        attention_mask = (1.0 - attention_mask) * -10000.0
        local_mask = (1.0 - local_mask) * -10000.0
        global_mask = (1.0 - global_mask) * -10000.0
        sa_self_mask = (1.0 - sa_self_mask) * -10000.0
        sa_cross_mask = (1.0 - sa_cross_mask) * -10000.0

        # outputs = self.electra(
        #     input_ids,
        #     attention_mask=attention_mask.squeeze(1),
        #     token_type_ids=token_type_ids,
        #     position_ids=position_ids,
        #     head_mask=head_mask,
        #     inputs_embeds=inputs_embeds,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        # )

        sequence_output = last_layer # (batch_size * num_choice, seq_len, hidden_size)


        local_word_level = self.localMHA[0](sequence_output, sequence_output, attention_mask = local_mask)[0]
        global_word_level = self.globalMHA[0](sequence_output, sequence_output, attention_mask = global_mask)[0]
        sa_self_word_level = self.SASelfMHA[0](sequence_output, sequence_output, attention_mask = sa_self_mask)[0]
        sa_cross_word_level = self.SACrossMHA[0](sequence_output, sequence_output, attention_mask = sa_cross_mask)[0]

        for t in range(1, self.num_decoupling):
            local_word_level = self.localMHA[t](local_word_level, local_word_level, attention_mask = local_mask)[0]
            global_word_level = self.globalMHA[t](global_word_level, global_word_level, attention_mask = global_mask)[0]
            sa_self_word_level = self.SASelfMHA[t](sa_self_word_level, sa_self_word_level, attention_mask = sa_self_mask)[0]
            sa_cross_word_level = self.SACrossMHA[t](sa_cross_word_level, sa_cross_word_level, attention_mask = sa_cross_mask)[0]

        context_word_level = self.fuse1(sequence_output, local_word_level, global_word_level)
        sa_word_level = self.fuse2(sequence_output, sa_self_word_level, sa_cross_word_level)

        new_batch = []

        context_utterance_level = []
        sa_utterance_level = []

        for i in range(sequence_output.size(0)):
            context_utterances = [torch.max(context_word_level[i, :(sep_pos[i][0] + 1)], dim = 0, keepdim = True)[0]]
            sa_utterances = [torch.max(sa_word_level[i, :(sep_pos[i][0] + 1)], dim = 0, keepdim = True)[0]]
            
            for j in range(1, last_seps[i] + 1):
                current_context_utter, _ = torch.max(context_word_level[i, (sep_pos[i][j-1] + 1):(sep_pos[i][j] + 1)], dim = 0, keepdim = True)
                current_sa_utter, _ = torch.max(sa_word_level[i, (sep_pos[i][j-1] + 1):(sep_pos[i][j] + 1)], dim = 0, keepdim = True)
                context_utterances.append(current_context_utter)
                sa_utterances.append(current_sa_utter)

            context_utterance_level.append(torch.cat(context_utterances, dim = 0)) # (batch_size, utterances, hidden_size)
            sa_utterance_level.append(torch.cat(sa_utterances, dim = 0))

        context_final_states = self.gru1(context_utterance_level) 
        sa_final_states = self.gru2(sa_utterance_level) # (batch_size * num_choice, 2 * hidden_size)

        # final_state = torch.cat((context_final_states, sa_final_states), 1)
        final_state = torch.cat((context_final_states, sa_final_states), 1)

        pooled_output = self.pooler_activation(self.pooler(final_state))
        # pooled_output = self.merger_activation(self.merger(torch.cat([correlation_output, pooled_output], dim=-1)))
        pooled_output = self.dropout(pooled_output)

        if num_labels > 2:
            logits = self.classifier(pooled_output)
        else:
            logits = self.classifier2(pooled_output)

        reshaped_logits = logits.view(-1, num_labels) if num_labels > 2 else logits

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs #(loss), reshaped_logits, (hidden_states), (attentions)
