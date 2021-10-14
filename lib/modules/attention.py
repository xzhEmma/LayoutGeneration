#!/usr/bin/env python
# codes modified from
#   https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/attention.py
# and
#   https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/GlobalAttention.py


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class Attention(nn.Module):
    """
    Inputs: h_t, h_s, m_s
        - **h_t** (batch, tgt_len,./):
            tensor containing output hidden features from the decoder.
        - **h_s** (batch, src_len, src_dim):
            tensor containing output hidden features from the encoder.
        - **m_s** (batch, src_len):
            tensor containing the padding mask of the encoded input sequence.

    Outputs: context, scores
        - **context** (batch, tgt_len, src_dim):
            tensor containing the attended output features
        - **scores** (batch, tgt_len, src_len):
            tensor containing attention weights.

    Examples::

         >>> attention = Attention('general', 256, 128)
         >>> h_s = torch.randn(5, 3, 256)
         >>> h_t = torch.randn(5, 5, 128)
         >>> m_s = torch.randn(5, 3).random_(0, 2)
         >>> context, scores = attention(h_t, h_s, m_s)

    """
    def __init__(self, attn_type, src_dim, tgt_dim, config):
        super(Attention, self).__init__()
        self.cfg = config
        self.attn_type = attn_type.lower()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
      #  print('-----------------',src_dim,tgt_dim)
        # Luong Attention
        if self.attn_type == 'general':
            # general: :math:`score(H_j, q) = H_j^T W_a q`

            self.linear_in = nn.Linear(src_dim, tgt_dim, bias=False)
        # Bahdanau Attention
        elif self.attn_type == 'mlp':
            # mlp: :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`
            self.linear_context = nn.Linear(src_dim, tgt_dim, bias=False)
            self.linear_query = nn.Linear(tgt_dim, tgt_dim, bias=True)
            self.v_a = nn.Linear(tgt_dim, 1, bias=False)
        elif self.attn_type == 'dot':
            # dot product scoring requires the encoder and decoder hidden states have the same dimension
            assert(src_dim == tgt_dim)
        else:
            raise ValueError("Unsupported attention mechanism: {0}".format(attn_type))

        # mlp wants it with bias
        # out_bias = self.attn_type == "mlp"
        # self.linear_out = nn.Linear(src_dim+tgt_dim, tgt_dim, bias=out_bias)

        # self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def score(self, h_t, h_s):
        # h_t: (batch, tgt_len , tgt_dim )537图像的隐藏状态
        # h_s: (batch, src_len , src_dim 600)812文本词编码
        # scores: (batch, tgt_len, src_len)

        if self.attn_type == 'dot':
            scores = torch.bmm(h_t, h_s.transpose(1, 2))
        elif self.attn_type == 'general':
            energy = self.linear_in(h_s)
            scores = torch.bmm(h_t, energy.transpose(1, 2))
        elif self.attn_type == 'mlp':
            src_batch, src_len, src_dim = h_s.size()
            tgt_batch, tgt_len, tgt_dim = h_t.size()
            assert(src_batch == tgt_batch)

            wq = self.linear_query(h_t)
            wq = wq.view(tgt_batch, tgt_len, 1, tgt_dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, tgt_dim)

            uh = self.linear_context(h_s)
            uh = uh.view(src_batch, 1, src_len, tgt_dim)
            uh = uh.expand(src_batch, tgt_len, src_len, tgt_dim)

            wquh = self.tanh(wq + uh)

            scores = self.v_a(wquh).view(src_batch, tgt_len, src_len)

        return scores
        

    def forward(self, h_t, h_s, m_s,index,index_lens):#layout [4, 10, 512+61] text [4, 18, 512+300]
        src_batch, src_len, src_dim = h_s.size()
        scores = self.score(h_t, h_s) # B x tgt_len x src_len
        src_mask = m_s.view(src_batch, 1, src_len)
        scores = scores * src_mask
        scores = scores - 1e11 * (1.0 - src_mask)
        scores = self.softmax(scores.clamp(min=-1e10))
        # (batch, tgt_len, src_len) * (batch, src_len, src_dim) -> (batch, tgt_len, src_dim)
        context = torch.bmm(scores, h_s)
        # concat -> (batch, tgt_len, src_dim+tgt_dim)
        # combined = torch.cat((context, h_t), dim=-1)
        # # output -> (batch, tgt_len, tgt_dim)
        # output = self.linear_out(combined)
        # if self.attn_type in ["general", "dot"]:
        #     output = self.tanh(output)
        l=0 
        if self.cfg.where_attn >0:
            for i in range(index.shape[0]):
                for j,ind in enumerate(index[i,:index_lens[i],:]):
                    at = scores[i,:,ind[0]]
                    ida = np.argmax(at.cpu().detach().numpy())
                    l1 = scores[i,ida,ind[0]] - scores[i,ida,ind[1]]
                    bt = scores[i,:,ind[1]]
                    idb = np.argmax(bt.cpu().detach().numpy())
                    l2 = scores[i,idb,ind[1]] - scores[i,idb,ind[0]]
                    l = l+abs(l1)+abs(l2)
            return context, scores, l

        return context, scores


class LOC_Attention(nn.Module):
    def __init__(self, attn_type, src_dim, tgt_dim):
        super(LOC_Attention, self).__init__()
        self.attn_type = attn_type.lower()
        self.src_dim = src_dim
        self.tgt_dim = tgt_dim
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.linear_in = nn.Linear(src_dim, tgt_dim, bias=False)
        self.fc = nn.Linear(784,512)
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
    def forward(self, context,curr_outs):#10 28 28 512
        context = self.linear_in(context)
        #if self.cfg.sentence = 1:
        #     context_w = torch.sum(context,1)
        #     context_w = 
        # else:
        #     context_w = 
        s = torch.einsum("iaj,iajbd->iabd",[context,curr_outs])#12 10 28 28
        bsize, tlen, gh, gw = s.size()
        nsize = bsize * tlen
        s = torch.flatten(s,start_dim=2)#12 10 784
        s = s.view(nsize,gh*gw)
        s_map = F.softmax(s, dim=-1)
        s_map = s_map.view(bsize, tlen, gh*gw)#12 10 784
        s_map = s_map.unsqueeze(-1)
        s_map = s_map.expand(bsize, tlen, gh*gw, gh*gw)#12 10 784 784

        curr_outs_f = torch.flatten(curr_outs,start_dim=-2)#12 10 512 28 28 -> 12 10 512 28*28 
        A = torch.matmul(curr_outs_f.transpose(2,3),curr_outs_f)#[12, 10, 784, 784]
        curr_outs_u = self.fc(A*s_map) #12 10 784 512
        curr_outs_u = torch.reshape(curr_outs_u,(bsize,-1,28,28,512))
        curr_outs_u = curr_outs_u.transpose(-1,-2)
        curr_outs_u = curr_outs_u.transpose(-2,-3)
        l_map = torch.einsum("iaj,iajbd->iabd",[context,curr_outs_u])#12 10 28 28
        l_map = torch.flatten(l_map,start_dim=2)#12 10 784
        l_map = l_map.view(nsize,gh*gw)
        l_map = F.softmax(l_map, dim=-1)
        l_map = l_map.view(bsize, tlen, gh*gw)#12 10 784
        l_map = l_map.view(bsize, tlen, gh,gw)
        l_map = l_map.unsqueeze(2)
        return l_map



