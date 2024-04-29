from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .att_model import pack_wrapper, AttModel


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, rm):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.rm = rm

    def forward(self, src, tgt, src_mask, tgt_mask,imglabels):
        
        return self.decode(self.encode(src, src_mask,imglabels), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask,imglabels):
        return self.encoder(self.src_embed(src), src_mask,imglabels)

    def decode(self, hidden_states, src_mask, tgt, tgt_mask):
        memory = self.rm.init_memory(hidden_states.size(0)).to(hidden_states)
        memory = self.rm(self.tgt_embed(tgt), memory)
        return self.decoder(self.tgt_embed(tgt), hidden_states, src_mask, tgt_mask, memory)


from fusion_model import Build_MultiModel_ShareBackbone

import torchvision.transforms as transforms


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)


        #method 1
        self.conv_multi=nn.Conv1d(513,512, kernel_size=1)  # 输入通道数为196，输出通道数为127，卷积核大小为1

        #method2
        # self.conv_multi=nn.Conv1d(521,512, kernel_size=1)  # 输入通道数为196，输出通道数为127，卷积核大小为1

        #method 3 [batch_size,1, 9]
        # self.conv_multi1=nn.Conv1d(9,512, kernel_size=1)  # 输入通道数为196，输出通道数为127，卷积核大小为1
        # self.conv_multi2=nn.Conv1d(50,49, kernel_size=1)  # 输入通道数为196，输出通道数为127，卷积核大小为1

        # self.conv_multi3=nn.Conv1d(49,1, kernel_size=1)  
        # self.conv_multi4=nn.Conv1d(2,49, kernel_size=1)  


    def forward(self, imgs_fea, mask,imglabels):
        

        avg_fet_list=[]
        for item in imgs_fea:
        

            for layer in self.layers:
                x = layer(item, mask)
            avg_fet_list.append(x)
        

        avg_fc_feats = torch.stack(avg_fet_list, dim=1)
        summed_fc_feats = torch.sum(avg_fc_feats, dim=1)  # 逐个特征相加
        avg_fc_feats_mean = summed_fc_feats / len(avg_fet_list)  # 计算平均特征


        # #method 1 和 2
        feat = torch.cat([avg_fc_feats_mean, imglabels], dim=2)  # 按照第三个维度拼接，变成[batch_size, 49, 513]
        feat = feat.transpose(1, 2)  # 变成[batch_size, 513, 49]        
        feat = self.conv_multi(feat)  # 变成[batch_size, 512，49 ]
        feat = feat.transpose(1, 2)  # 变成[batch_size, 49,512]


        # #method 3
        # imglabels = self.conv_multi1(imglabels)  ## 变成[batch_size, 512, 1]
        # imglabels = imglabels.transpose(1, 2)  # 变成[batch_size, 1,512]
        # feat = torch.cat([avg_fc_feats_mean, imglabels], dim=1) # 按照第2个维度拼接，变成[batch_size, 50, 512]
        # feat = self.conv_multi2(feat)



        # #method 4   把avg_fc_feats_mean映射到1 512 然后拼成2 512 
        # imglabels = self.conv_multi1(imglabels)  ## 变成[batch_size, 512, 1]
        # imglabels = imglabels.transpose(1, 2)  # 变成[batch_size, 1,512]
        # avg_fc_feats_mean = self.conv_multi3(avg_fc_feats_mean)  ## 变成[batch_size, 1, 512]
        # feat = torch.cat([avg_fc_feats_mean, imglabels], dim=1) # 按照第2个维度拼接，变成[batch_size, 2, 512]
        # feat = self.conv_multi4(feat)




        return self.norm(feat)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask, memory)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout, rm_num_slots, rm_d_model):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ConditionalSublayerConnection(d_model, dropout, rm_num_slots, rm_d_model), 3)

    def forward(self, x, hidden_states, src_mask, tgt_mask, memory):
        m = hidden_states
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask), memory)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask), memory)
        return self.sublayer[2](x, self.feed_forward, memory)


class ConditionalSublayerConnection(nn.Module):
    def __init__(self, d_model, dropout, rm_num_slots, rm_d_model):
        super(ConditionalSublayerConnection, self).__init__()
        self.norm = ConditionalLayerNorm(d_model, rm_num_slots, rm_d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, memory):
        return x + self.dropout(sublayer(self.norm(x, memory)))


class ConditionalLayerNorm(nn.Module):
    def __init__(self, d_model, rm_num_slots, rm_d_model, eps=1e-6):
        super(ConditionalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.rm_d_model = rm_d_model
        self.rm_num_slots = rm_num_slots
        self.eps = eps

        self.mlp_gamma = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(rm_d_model, rm_d_model))

        self.mlp_beta = nn.Sequential(nn.Linear(rm_num_slots * rm_d_model, d_model),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(d_model, d_model))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x, memory):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        delta_gamma = self.mlp_gamma(memory)
        delta_beta = self.mlp_beta(memory)
        gamma_hat = self.gamma.clone()
        beta_hat = self.beta.clone()
        gamma_hat = torch.stack([gamma_hat] * x.size(0), dim=0)
        gamma_hat = torch.stack([gamma_hat] * x.size(1), dim=1)
        beta_hat = torch.stack([beta_hat] * x.size(0), dim=0)
        beta_hat = torch.stack([beta_hat] * x.size(1), dim=1)
        gamma_hat += delta_gamma
        beta_hat += delta_beta
        return gamma_hat * (x - mean) / (std + self.eps) + beta_hat


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RelationalMemory(nn.Module):

    def __init__(self, num_slots, d_model, num_heads=1):
        super(RelationalMemory, self).__init__()
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.d_model = d_model

        self.attn = MultiHeadedAttention(num_heads, d_model)
        self.mlp = nn.Sequential(nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU(),
                                 nn.Linear(self.d_model, self.d_model),
                                 nn.ReLU())

        self.W = nn.Linear(self.d_model, self.d_model * 2)
        self.U = nn.Linear(self.d_model, self.d_model * 2)

    def init_memory(self, batch_size):
        memory = torch.stack([torch.eye(self.num_slots)] * batch_size)
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((batch_size, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]

        return memory

    def forward_step(self, input, memory):
        memory = memory.reshape(-1, self.num_slots, self.d_model)
        q = memory
        k = torch.cat([memory, input.unsqueeze(1)], 1)
        v = torch.cat([memory, input.unsqueeze(1)], 1)
        next_memory = memory + self.attn(q, k, v)
        next_memory = next_memory + self.mlp(next_memory)

        gates = self.W(input.unsqueeze(1)) + self.U(torch.tanh(memory))
        gates = torch.split(gates, split_size_or_sections=self.d_model, dim=2)
        input_gate, forget_gate = gates
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)

        next_memory = input_gate * torch.tanh(next_memory) + forget_gate * memory
        next_memory = next_memory.reshape(-1, self.num_slots * self.d_model)

        return next_memory

    def forward(self, inputs, memory):
        outputs = []
        for i in range(inputs.shape[1]):
            memory = self.forward_step(inputs[:, i], memory)
            outputs.append(memory)
        outputs = torch.stack(outputs, dim=1)

        return outputs


class EncoderDecoder(AttModel):

    def make_model(self, tgt_vocab):
        c = copy.deepcopy
        



        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        rm = RelationalMemory(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads)
        model = Transformer(
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),



            Decoder(
                DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout, self.rm_num_slots, self.rm_d_model),
                4),
            lambda x: x,
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)),
            rm)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(EncoderDecoder, self).__init__(args, tokenizer)

        self.args = args



        self.device = torch.device(self.args.gpu_index if torch.cuda.is_available() else "cpu")




        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.rm_num_slots = args.rm_num_slots
        self.rm_num_heads = args.rm_num_heads
        self.rm_d_model = args.rm_d_model

        tgt_vocab = self.vocab_size + 1

        self.model = self.make_model(tgt_vocab)
        self.logit = nn.Linear(args.d_model, tgt_vocab)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats,imagelabels, att_masks):

        att_feats_resize, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats_resize, att_masks,imagelabels)




        avg_fc_feats = torch.stack(fc_feats, dim=1)
        summed_fc_feats = torch.sum(avg_fc_feats, dim=1)  # 逐个特征相加
        avg_fc_feats_mean = summed_fc_feats / len(fc_feats)  # 计算平均特征


        avg_att_feats = torch.stack(att_feats, dim=1)
        summed_att_feats = torch.sum(avg_att_feats, dim=1)  # 逐个特征相加
        avg_att_feats_mean = summed_att_feats / len(att_feats)  # 计算平均特征




        return avg_fc_feats_mean[..., :1], avg_att_feats_mean[..., :1], memory, att_masks

    # def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
    #     att_feats, att_masks = self.clip_att(att_feats, att_masks)
    #     att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

    #     if att_masks is None:
    #         att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
    #     att_masks = att_masks.unsqueeze(-2)

    #     if seq is not None:
    #         # crop the last one
    #         seq = seq[:, :-1]
    #         seq_mask = (seq.data > 0)
    #         seq_mask[:, 0] += True

    #         seq_mask = seq_mask.unsqueeze(-2)
    #         seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
    #     else:
    #         seq_mask = None

    #     return att_feats, seq, att_masks, seq_mask
    

    def _prepare_feature_forward(self, att_feats_list, att_masks=None, seq=None):
    # 对每个图片特征进行处理
        att_feats_processed = []
        for att_feats in att_feats_list:
            att_feats, att_masks = self.clip_att(att_feats, att_masks)
            att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
            att_feats_processed.append(att_feats)

        # 合并处理后的图片特征
        # att_feats = torch.cat(att_feats_processed, dim=0)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats_processed[0].shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return att_feats_processed, seq, att_masks, seq_mask


    def _forward(self, fc_feats, att_feats,imglabels ,seq,att_masks=None):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)



        out = self.model(att_feats, seq, att_masks, seq_mask,imglabels)
        
        
        
        
        
        
        outputs = F.log_softmax(self.logit(out), dim=-1)
        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        # if len(state) == 0:
        #     ys = it.unsqueeze(1)
        # else:
        #     ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)

        if len(state) == 0:
            ys = it.unsqueeze(1).to(self.device)
        else:
            ys = torch.cat([state[0][0].to(self.device), it.unsqueeze(1).to(self.device)], dim=1)

        # out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device))
        out = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(self.device))
        return out[:, -1], [ys.unsqueeze(0)]
